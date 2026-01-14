# noisy_ml_simulator.py
"""
NoisyMLSimulator: a wrapper around StateVecSimTorch that records gates via CircuitBridge
and can run a noisy simulation using Qiskit Aer via NoiseExtractor.

Design goals:
- Defensive imports: importing this module should fail with a clear message if core
  dependencies are missing (StateVecSimTorch, CircuitBridge).
- Qiskit / NoiseExtractor is optional. If Aer isn't available, noisy path is disabled
  but the class still works for ideal simulations.
- Clear runtime errors for misuse (e.g., calling get_comparison before running forward).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np
from .qiskit_noise_extractor import NoiseExtractor
from .circuit_bridge import CircuitBridge
from .torch_state_vector_simulator import StateVecSimTorch

__all__ = ["NoisyMLSimulator"]

class NoisyMLSimulator(StateVecSimTorch):
    """
    Extended StateVecSimTorch that:
      - Records gates into a CircuitBridge as gates are applied.
      - Optionally simulates the recorded Qiskit circuit under a depolarizing noise
        model via NoiseExtractor (uses Qiskit Aer).
      - Provides a `get_comparison()` API returning ideal vs noisy metrics.

    Parameters
    ----------
    n_qubits, n_layers : int
        forwarded to StateVecSimTorch
    p_1qubit, p_2qubit : float
        depolarizing error probabilities for 1- and 2-qubit gates used to create
        a simple noise model (only used if NoiseExtractor is available).
    track_noise : bool
        enable/disable noisy simulation. If True but qiskit isn't installed, noisy
        path will be disabled and a warning printed.
    gpu, seed, **kwargs : passed to parent
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        p_1qubit: float = 0.001,
        p_2qubit: float = 0.01,
        track_noise: bool = True,
        gpu: bool = False,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, gpu=gpu, seed=seed, **kwargs)

        self.p_1qubit = float(p_1qubit)
        self.p_2qubit = float(p_2qubit)
        self.track_noise = bool(track_noise)

        self.circuit_bridge = CircuitBridge(n_qubits=self.n_qubits)

        self.noise_extractor: Optional[Any] = None
        if self.track_noise:
                self.noise_extractor = NoiseExtractor(p_1qubit=self.p_1qubit, p_2qubit=self.p_2qubit)
                
        self.ideal_states: list = []
        self.ideal_bloch_trajectory: list = []

   
    def forward(self, X):
        """
        Reset trackers and call parent forward which initializes `self.state`.
        Keep a copy (CPU, detached) of the initial state.
        """
        self.circuit_bridge.reset()
        self.ideal_states = []
        self.ideal_bloch_trajectory = []

        initial_state = super().forward(X)
        # store first snapshot (detached, CPU)
        self.ideal_states.append(initial_state.clone().detach().cpu())
        if self.n_qubits == 1:
            self.ideal_bloch_trajectory.append(self._state_to_bloch(initial_state))
        return initial_state

    def H(self, qubits, n_samples):
        """Apply Hadamard via parent, record gate and snapshot."""
        final_state = super().H(qubits, n_samples)
        self.circuit_bridge.record_gate("H", qubits)
        self.ideal_states.append(final_state.clone().detach().cpu())
        if self.n_qubits == 1:
            self.ideal_bloch_trajectory.append(self._state_to_bloch(final_state))
        return final_state

    def Rot(self, X):
        """Apply parameterized rot to all qubits and record per-qubit rotation angles."""
        final_state = super().Rot(X)
        try:
            angles_batch = X.detach().cpu().numpy()
            for q in range(self.n_qubits):
                angles = angles_batch[0, q].tolist()
                self.circuit_bridge.record_gate("ROT", [q], params={"angles": angles})
        except Exception:
            pass
        self.ideal_states.append(final_state.clone().detach().cpu())
        if self.n_qubits == 1:
            self.ideal_bloch_trajectory.append(self._state_to_bloch(final_state))
        return final_state

    def CNOT(self, qubit_map, n_samples):
        """Apply CNOTs, record them in canonical (control, target) pairs."""
        final_state = super().CNOT(qubit_map, n_samples)
        if qubit_map == "next" or qubit_map == "ring":
            qs = [(i, i + 1) for i in range(self.n_qubits - 1)]
            if qubit_map == "ring" and self.n_qubits > 2:
                qs.append((self.n_qubits - 1, 0))
        else:
            qs = qubit_map
        for pair in qs:
            self.circuit_bridge.record_gate("CNOT", list(pair))
        self.ideal_states.append(final_state.clone().detach().cpu())
        if self.n_qubits == 1:
            self.ideal_bloch_trajectory.append(self._state_to_bloch(final_state))
        return final_state

    # ( add similar wrappers for X,Y,Z, RX, RY, RZ, CZ, SWAP )


    def _statevector_to_density(self, state_vec_torch):
        """Assumes state_vec_torch shape (batch, dim). Returns density matrix for first sample (numpy)."""
        sv = state_vec_torch.detach().cpu().numpy()
        if sv.ndim == 2:
            psi = sv[0]
        else:
            psi = sv
        rho = np.outer(psi, np.conjugate(psi))
        return rho

    def _reduced_density(self, rho: np.ndarray, keep: int):
        """
        Partial trace to obtain reduced 1-qubit density matrix for qubit index `keep`.
        Qubit indexing: 0..n_qubits-1 (keeps semantics consistent with your StateVecSimTorch).
        """
        n = self.n_qubits
        rho_reshaped = rho.reshape([2] * n * 2)
        rows = list(range(0, n))
        cols = list(range(n, 2 * n))
        keep_row = keep
        keep_col = keep + n
        other_rows = [r for r in rows if r != keep_row]
        other_cols = [c for c in cols if c != keep_col]
        perm = [keep_row, keep_col] + other_rows + other_cols
        rho_perm = np.transpose(rho_reshaped, perm)
        k = n - 1
        rho_perm = rho_perm.reshape(2, 2, 2 ** k, 2 ** k)
        reduced = np.zeros((2, 2), dtype=rho.dtype)
        for i in range(2 ** k):
            reduced += rho_perm[:, :, i, i]
        return reduced

    def _density_to_bloch(self, rho_1q: np.ndarray):
        """Convert 2x2 density matrix to Bloch vector [x,y,z]."""
        assert rho_1q.shape == (2, 2)
        rho00 = rho_1q[0, 0]
        rho11 = rho_1q[1, 1]
        rho01 = rho_1q[0, 1]
        rx = 2.0 * np.real(rho01)
        ry = -2.0 * np.imag(rho01)
        rz = np.real(rho00 - rho11)
        return np.array([rx, ry, rz], dtype=float)

    def _state_to_bloch(self, state_torch):
        """Compute Bloch vector for qubit 0 from a torch statevector (batch x dim)."""
        rho = self._statevector_to_density(state_torch)
        reduced = self._reduced_density(rho, keep=0)
        return self._density_to_bloch(reduced)

    def get_comparison(self) -> Dict[str, Any]:
        """
        Returns:
          {
            "ideal": {"density_matrix": rho_ideal, "fidelity": 1.0, "purity": purity_ideal, "bloch_trajectory": [...]},
            "noisy": {"density_matrix": rho_noisy, "fidelity": fid, "purity": purity_noisy, "bloch": [...]),
            "qiskit_circuit": qc
          }
        Notes:
          - fidelity here is psi^† rho_noisy psi (ideal state is pure).
          - This implementation uses the last stored ideal state snapshot.
        """
        if len(self.ideal_states) == 0:
            raise RuntimeError("No ideal state recorded. Call forward(X) and apply gates before calling get_comparison().")

        ideal_state_torch = self.ideal_states[-1]
        rho_ideal = self._statevector_to_density(ideal_state_torch)

        qc = self.circuit_bridge.to_qiskit_circuit()
        if self.track_noise and self.noise_extractor is not None:
            rho_noisy = self.noise_extractor.simulate_circuit(qc)
        else:
            rho_noisy = deepcopy(rho_ideal)

        # compute fidelity (ideal pure psi vs rho_noisy)
        psi = ideal_state_torch.detach().cpu().numpy()[0]
        fid = float(np.real(np.vdot(psi, rho_noisy.dot(psi))))

        purity_ideal = float(np.real(np.trace(rho_ideal.dot(rho_ideal))))
        purity_noisy = float(np.real(np.trace(rho_noisy.dot(rho_noisy))))

        if self.n_qubits == 1:
            bloch_ideal = self._density_to_bloch(rho_ideal)
            bloch_noisy = self._density_to_bloch(rho_noisy)
        else:
            bloch_ideal = self._density_to_bloch(self._reduced_density(rho_ideal, 0))
            bloch_noisy = self._density_to_bloch(self._reduced_density(rho_noisy, 0))

        return {
            "ideal": {
                "density_matrix": rho_ideal,
                "fidelity": 1.0,
                "purity": purity_ideal,
                "bloch_trajectory": [b.tolist() for b in self.ideal_bloch_trajectory]
                if self.ideal_bloch_trajectory
                else [bloch_ideal.tolist()],
            },
            "noisy": {
                "density_matrix": rho_noisy,
                "fidelity": fid,
                "purity": purity_noisy,
                "bloch": bloch_noisy.tolist(),
            },
            "qiskit_circuit": qc,
        }
