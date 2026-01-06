from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional
import numpy as np
from app.backends.qiskit_noise_extractor import NoiseExtractor
from app.backends.circuit_bridge import CircuitBridge
from app.backends.torch_state_vector_simulator import StateVecSimTorch

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

    def _snapshot_and_record(self, final_state, gate_name: str, qubits: Sequence[int], params: Dict[str, Any] = None):
        """Record gate into circuit bridge and snapshot ideal state & bloch (if 1 qubit)."""
        if params is None:
            params = {}
        self.circuit_bridge.record_gate(gate_name, list(qubits), params=params)
        # store snapshot (detached CPU)
        self.ideal_states.append(final_state.clone().detach().cpu())
        if self.n_qubits == 1:
            try:
                self.ideal_bloch_trajectory.append(self._state_to_bloch(final_state))
            except Exception:
                # be robust if conversion fails
                pass

    def _extract_qubits_from_args(self, args, kwargs) -> List[int]:
        """
        Best-effort extraction of qubit indices from args/kwargs.
        Handles the common call patterns:
         - first positional arg is int (single qubit)
         - first positional arg is list/tuple of ints
         - qubit_map strings like "next", "ring"
         - 'qubits' keyword
        Returns list of ints (may be empty if can't infer).
        """
        # priority: kwargs['qubits']
        qs = kwargs.get("qubits", None)
        if qs is not None:
            return list(qs) if isinstance(qs, (list, tuple)) else [int(qs)]
        if len(args) >= 1:
            first = args[0]
            # qubit map strings used in your CNOT implementation
            if isinstance(first, str) and first in ("next", "ring"):
                if first == "next":
                    return [(i, i + 1) for i in range(self.n_qubits - 1)]
                else:  # ring
                    qs_pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
                    if self.n_qubits > 2:
                        qs_pairs.append((self.n_qubits - 1, 0))
                    return qs_pairs
            if isinstance(first, (list, tuple)):
                return list(first)
            try:
                # single int-like
                return [int(first)]
            except Exception:
                pass
        # fallback: empty
        return []

    # Generic single-qubit Pauli wrappers (accept variable signatures, forward to parent)
    def X(self, *args, **kwargs):
        final_state = super().X(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "X", qs)
        return final_state

    def Y(self, *args, **kwargs):
        final_state = super().Y(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "Y", qs)
        return final_state

    def Z(self, *args, **kwargs):
        final_state = super().Z(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "Z", qs)
        return final_state

    # Parameterized single-qubit rotations (RX/RY/RZ). We try to find 'theta' in args/kwargs.
    def RX(self, *args, **kwargs):
        final_state = super().RX(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        theta = kwargs.get("theta", None)
        # often theta could be second positional arg
        if theta is None and len(args) >= 2:
            theta = args[1]
        params = {"theta": float(theta)} if theta is not None else {}
        self._snapshot_and_record(final_state, "RX", qs, params=params)
        return final_state

    def RY(self, *args, **kwargs):
        final_state = super().RY(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        theta = kwargs.get("theta", None)
        if theta is None and len(args) >= 2:
            theta = args[1]
        params = {"theta": float(theta)} if theta is not None else {}
        self._snapshot_and_record(final_state, "RY", qs, params=params)
        return final_state

    def RZ(self, *args, **kwargs):
        final_state = super().RZ(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        theta = kwargs.get("theta", None)
        if theta is None and len(args) >= 2:
            theta = args[1]
        params = {"theta": float(theta)} if theta is not None else {}
        self._snapshot_and_record(final_state, "RZ", qs, params=params)
        return final_state

    # Controlled-Z (sometimes parent may expose CZ)
    def CZ(self, *args, **kwargs):
        final_state = super().CZ(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "CZ", qs)
        return final_state

    # Swap
    def SWAP(self, *args, **kwargs):
        final_state = super().SWAP(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "SWAP", qs)
        return final_state

    # S / SDG / T / TDG / SX / SXDG (phase / small-rotation gates)
    def S(self, *args, **kwargs):
        final_state = super().S(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "S", qs)
        return final_state

    def SDG(self, *args, **kwargs):
        final_state = super().SDG(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "SDG", qs)
        return final_state

    def T(self, *args, **kwargs):
        final_state = super().T(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "T", qs)
        return final_state

    def TDG(self, *args, **kwargs):
        final_state = super().TDG(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "TDG", qs)
        return final_state

    def SX(self, *args, **kwargs):
        final_state = super().SX(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "SX", qs)
        return final_state

    def SXDG(self, *args, **kwargs):
        final_state = super().SXDG(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "SXDG", qs)
        return final_state

    # Generic U / U3 wrapper: try to capture (theta, phi, lam) if provided
    def U(self, *args, **kwargs):
        final_state = super().U(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        # attempt to find angles in kwargs or positional args
        params = {}
        if "params" in kwargs:
            params = kwargs["params"]
        else:
            # common signature: (qubit, theta, phi, lam, ...)
            if len(args) >= 4:
                try:
                    params = {"theta": float(args[1]), "phi": float(args[2]), "lam": float(args[3])}
                except Exception:
                    params = {}
            else:
                # check kwargs for theta/phi/lam
                for key in ("theta", "phi", "lam", "lambda"):
                    if key in kwargs:
                        params[key] = float(kwargs[key])
        self._snapshot_and_record(final_state, "U", qs, params=params)
        return final_state

    # Controlled parameterized rotations (CRX/CRY/CRZ)
    def CRX(self, *args, **kwargs):
        final_state = super().CRX(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        theta = kwargs.get("theta", None)
        if theta is None and len(args) >= 2:
            theta = args[1]
        params = {"theta": float(theta)} if theta is not None else {}
        self._snapshot_and_record(final_state, "CRX", qs, params=params)
        return final_state

    def CRY(self, *args, **kwargs):
        final_state = super().CRY(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        theta = kwargs.get("theta", None)
        if theta is None and len(args) >= 2:
            theta = args[1]
        params = {"theta": float(theta)} if theta is not None else {}
        self._snapshot_and_record(final_state, "CRY", qs, params=params)
        return final_state

    def CRZ(self, *args, **kwargs):
        final_state = super().CRZ(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        theta = kwargs.get("theta", None)
        if theta is None and len(args) >= 2:
            theta = args[1]
        params = {"theta": float(theta)} if theta is not None else {}
        self._snapshot_and_record(final_state, "CRZ", qs, params=params)
        return final_state

    # Toffoli (CCX)
    def CCX(self, *args, **kwargs):
        final_state = super().CCX(*args, **kwargs)
        qs = self._extract_qubits_from_args(args, kwargs)
        self._snapshot_and_record(final_state, "CCX", qs)
        return final_state


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
