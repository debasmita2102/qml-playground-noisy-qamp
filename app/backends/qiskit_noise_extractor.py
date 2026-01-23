from __future__ import annotations

import os
from typing import Optional, Callable
import numpy as _np
import logging
import socket

_IMPORT_ERROR = None
try:
    from qiskit import QuantumCircuit
    import qiskit_aer.noise as noise
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
except Exception as e_a:
    try:
        from qiskit import QuantumCircuit
        from qiskit.providers.aer import AerSimulator
        from qiskit.providers.aer.noise import NoiseModel
        import qiskit.providers.aer.noise as noise
    except Exception as e_b:
        _IMPORT_ERROR = e_a if e_a else e_b
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None

__all__ = ["NoiseExtractor"]

class NoiseExtractor:
    """Build and run simple noise models (depolarizing and/or amplitude damping/phase damping)
    using Qiskit Aer.

    Multi-qubit numpy-channel helpers apply *local* single-qubit channels independently
    to a set of target qubits (default: all qubits). Target indexing uses the convention
    that qubit 0 is the least-significant / right-most tensor factor when forming kronecker
    products (i.e., ordering matches numpy.kron(..., q2, q1, q0)).
    """

    def __init__(
        self,
        p_1qubit: float = 0.001,
        p_2qubit: float = 0.01,
        error_kind: str = "depolarizing",
        p_amp: Optional[float] = None,
        p_phase: Optional[float] = None,
        ibm_backend_name: Optional[str] = None,
        provider_loader: Optional[Callable] = None,
        allow_fake_provider: Optional[bool] = None,
    ):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "qiskit / qiskit-aer not available: install 'qiskit-aer' and 'qiskit' to use NoiseExtractor"
                + f" (import error: {_IMPORT_ERROR})"
            )

        self.p_1qubit = float(p_1qubit)
        self.p_2qubit = float(p_2qubit)
        self.error_kind = str(error_kind).lower()
        self.p_amp = float(p_amp) if p_amp is not None else self.p_1qubit
        self.p_phase = float(p_phase) if p_phase is not None else self.p_1qubit
        self.ibm_backend_name = None if ibm_backend_name is None else (str(ibm_backend_name).strip() or None)
        self.ibm_backend_error: Optional[str] = None
        self.ibm_backend = None
        self._provider_loader = provider_loader
        # Allow fake provider fallback only when explicitly requested (env or argument).
        if allow_fake_provider is None:
            env_allow_fake = os.getenv("NOISE_EXTRACTOR_ALLOW_FAKE_PROVIDER")
            allow_fake_provider = str(env_allow_fake).lower() in ("1", "true", "yes", "on") if env_allow_fake is not None else False
        self.allow_fake_provider = bool(allow_fake_provider)

        # Build noise model & simulator
        self.noise_model = self._build_noise_model()
        self.simulator = AerSimulator(noise_model=self.noise_model, method="density_matrix")

        # Optionally override with a real IBM backend noise model
        if self.ibm_backend_name:
            try:
                self.load_ibm_backend(
                    self.ibm_backend_name,
                    provider_loader=self._provider_loader,
                    allow_fake_provider=self.allow_fake_provider,
                )
            except Exception as exc:
                self.ibm_backend_error = str(exc)
                _logger.warning(
                    "Falling back to synthetic noise model; IBM backend '%s' unavailable: %s",
                    self.ibm_backend_name,
                    exc,
                )

    @staticmethod
    def _kraus_amplitude_damping(gamma: float):
        k0 = _np.array([[1.0, 0.0], [0.0, _np.sqrt(max(0.0, 1.0 - gamma))]], dtype=_np.complex128)
        k1 = _np.array([[0.0, _np.sqrt(max(0.0, gamma))], [0.0, 0.0]], dtype=_np.complex128)
        return [k0, k1]

    @staticmethod
    def _kraus_phase_damping(p: float):
        I = _np.eye(2, dtype=_np.complex128)
        k0 = _np.sqrt(max(0.0, 1.0 - p)) * I
        k1 = _np.sqrt(max(0.0, p)) * _np.array([[1.0, 0.0], [0.0, 0.0]], dtype=_np.complex128)
        k2 = _np.sqrt(max(0.0, p)) * _np.array([[0.0, 0.0], [0.0, 1.0]], dtype=_np.complex128)
        return [k0, k1, k2]

    @staticmethod
    def _kraus_depolarizing(p: float):
        """Single-qubit Kraus operators for depolarizing channel with parameter p:
           E(rho) = (1-p) rho + p I/2. Equivalent Kraus set: {sqrt(1-3p/4) I, sqrt(p/4) X, sqrt(p/4) Y, sqrt(p/4) Z}
        """
        p = float(p)
        sq = lambda x: _np.sqrt(max(0.0, x))
        k0 = sq(1.0 - 3.0 * p / 4.0) * _np.eye(2, dtype=_np.complex128)
        kx = sq(p / 4.0) * _np.array([[0.0, 1.0], [1.0, 0.0]], dtype=_np.complex128)
        ky = sq(p / 4.0) * _np.array([[0.0, -1j], [1j, 0.0]], dtype=_np.complex128)
        kz = sq(p / 4.0) * _np.array([[1.0, 0.0], [0.0, -1.0]], dtype=_np.complex128)
        return [k0, kx, ky, kz]

    def _build_noise_model(self) -> NoiseModel:
        """Create a NoiseModel with errors applied to common 1- and 2-qubit gates.

        Single-qubit errors are chosen by `self.error_kind`. Two-qubit gates use depolarizing error.
        """
        single_q_error = None

        if self.error_kind == "amplitude_damping":
            try:
                single_q_error = noise.amplitude_damping_error(self.p_amp)
            except Exception:
                try:
                    ks = self._kraus_amplitude_damping(self.p_amp)
                    single_q_error = noise.kraus_error(ks)
                except Exception as e:
                    raise RuntimeError(f"Failed to construct amplitude damping error: {e}") from e
        elif self.error_kind == "phase_damping":
            try:
                single_q_error = noise.phase_damping_error(self.p_phase)
            except Exception:
                try:
                    ks = self._kraus_phase_damping(self.p_phase)
                    single_q_error = noise.kraus_error(ks)
                except Exception as e:
                    raise RuntimeError(f"Failed to construct phase damping error: {e}") from e
        else:
            try:
                single_q_error = noise.depolarizing_error(self.p_1qubit, 1)
            except Exception:
                ks = self._kraus_depolarizing(self.p_1qubit)
                single_q_error = noise.kraus_error(ks)

        try:
            err_2q = noise.depolarizing_error(self.p_2qubit, 2)
        except Exception as e:
            raise RuntimeError(f"Failed to create two-qubit depolarizing error: {e}") from e

        nm = NoiseModel()

        one_q_gates = [
            "u1",
            "u2",
            "u3",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
        ]
        two_q_gates = ["cx", "cz", "swap"]

        for g in one_q_gates:
            try:
                nm.add_all_qubit_quantum_error(single_q_error, g)
            except Exception:
                pass

        for g in two_q_gates:
            try:
                nm.add_all_qubit_quantum_error(err_2q, g)
            except Exception:
                pass

        return nm

    def build_noise_from_backend(self, backend) -> NoiseModel:
        """Build a NoiseModel from a real backend object (requires IBM provider backend)."""
        if _IMPORT_ERROR is not None:
            raise RuntimeError("qiskit not available")
        nm = NoiseModel.from_backend(backend)
        self.noise_model = nm
        self.simulator = AerSimulator(noise_model=self.noise_model, method="density_matrix")
        return nm

    @staticmethod
    def _validate_density_matrix(rho: _np.ndarray):
        if rho is None:
            raise ValueError("rho is None")
        if not isinstance(rho, _np.ndarray):
            raise TypeError("rho must be a numpy array")
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("rho must be a square 2D density matrix")
        dim = rho.shape[0]
        if (dim & (dim - 1)) != 0:
            raise ValueError("rho dimension must be a power of two (2^n x 2^n)")

    @staticmethod
    def _num_qubits_from_rho(rho: _np.ndarray) -> int:
        dim = rho.shape[0]
        n = int(round(_np.log2(dim)))
        return n

    @staticmethod
    def _full_kron_for_target(single_op: _np.ndarray, target: int, n_qubits: int) -> _np.ndarray:
        """
        Build full-system operator for single-qubit operator `single_op` acting on `target` qubit.
        Tensor ordering: (... ⊗ q2 ⊗ q1 ⊗ q0), so target=0 -> right-most factor.
        """

        ops = []
        for q in range(n_qubits - 1, -1, -1):
            if q == target:
                ops.append(single_op)
            else:
                ops.append(_np.eye(2, dtype=_np.complex128))
        full = ops[0]
        for op in ops[1:]:
            full = _np.kron(full, op)
        return full

    def _apply_local_kraus_channel_to_rho(self, rho: _np.ndarray, kraus_list, target: int) -> _np.ndarray:
        """
        Apply a single-qubit Kraus list to `target` qubit in full density `rho`.
        Returns updated rho' = sum_i (K_i_full) rho (K_i_full)^\dagger
        """

        n_qubits = self._num_qubits_from_rho(rho)
        rho_p = _np.zeros_like(rho, dtype=_np.complex128)
        for K in kraus_list:
            K_full = self._full_kron_for_target(K, target, n_qubits)
            rho_p = rho_p + K_full @ rho @ K_full.conj().T
        return _np.asarray(rho_p, dtype=_np.complex128)

    def apply_amplitude_damping_to_density_multi(self, rho: _np.ndarray, gamma: Optional[float] = None, targets=None) -> _np.ndarray:
        """
        Apply amplitude damping independently to each qubit in `targets`.
        - rho: full-system density matrix (2^n x 2^n)
        - gamma: damping probability (defaults to self.p_amp)
        - targets: iterable of qubit indices (little-endian). If None, apply to all qubits.
        """
        self._validate_density_matrix(rho)
        gamma = float(gamma) if gamma is not None else float(self.p_amp)
        n = self._num_qubits_from_rho(rho)
        if targets is None:
            targets = list(range(n))
        else:
            targets = list(targets)

        ks = self._kraus_amplitude_damping(gamma)
        rho_p = rho.copy()
        for t in targets:
            if t < 0 or t >= n:
                raise IndexError(f"target qubit {t} out of range for {n} qubits")
            rho_p = self._apply_local_kraus_channel_to_rho(rho_p, ks, int(t))
        return _np.asarray(rho_p, dtype=_np.complex128)

    def apply_phase_damping_to_density_multi(self, rho: _np.ndarray, p: Optional[float] = None, targets=None) -> _np.ndarray:
        """
        Apply phase damping independently to each qubit in `targets`.
        - p: dephasing probability (defaults to self.p_phase)
        """
        self._validate_density_matrix(rho)
        p = float(p) if p is not None else float(self.p_phase)
        n = self._num_qubits_from_rho(rho)
        if targets is None:
            targets = list(range(n))
        else:
            targets = list(targets)

        ks = self._kraus_phase_damping(p)
        rho_p = rho.copy()
        for t in targets:
            if t < 0 or t >= n:
                raise IndexError(f"target qubit {t} out of range for {n} qubits")
            rho_p = self._apply_local_kraus_channel_to_rho(rho_p, ks, int(t))
        return _np.asarray(rho_p, dtype=_np.complex128)

    def apply_depolarizing_to_density_multi(self, rho: _np.ndarray, p: Optional[float] = None, targets=None) -> _np.ndarray:
        """
        Apply single-qubit depolarizing channel independently to each qubit in `targets`.
        - p: depolarizing probability (defaults to self.p_1qubit)
        """
        self._validate_density_matrix(rho)
        p = float(p) if p is not None else float(self.p_1qubit)
        n = self._num_qubits_from_rho(rho)
        if targets is None:
            targets = list(range(n))
        else:
            targets = list(targets)

        ks = self._kraus_depolarizing(p)
        rho_p = rho.copy()
        for t in targets:
            if t < 0 or t >= n:
                raise IndexError(f"target qubit {t} out of range for {n} qubits")
            rho_p = self._apply_local_kraus_channel_to_rho(rho_p, ks, int(t))
        return _np.asarray(rho_p, dtype=_np.complex128)

    def apply_noise_to_density_multi(self, rho: _np.ndarray, targets=None, **kwargs) -> _np.ndarray:
        """
        Convenience dispatcher: apply `self.error_kind` channel to `rho`.
        kwargs passed to underlying method (e.g., gamma/p/p).
        """
        kind = self.error_kind
        if kind == "amplitude_damping":
            return self.apply_amplitude_damping_to_density_multi(rho, gamma=kwargs.get("gamma", None), targets=targets)
        if kind == "phase_damping":
            return self.apply_phase_damping_to_density_multi(rho, p=kwargs.get("p", None), targets=targets)
        return self.apply_depolarizing_to_density_multi(rho, p=kwargs.get("p", None), targets=targets)

    def simulate_circuit(self, qc: QuantumCircuit, shots: int = 1) -> _np.ndarray:
        """
        Short, modern simulate_circuit for Aer builds that store final state under
        result.data()['statevector'] or result.data()['density_matrix'] (Statevector /
        DensityMatrix objects). Returns a numpy.complex128 density matrix.
        """
        if _IMPORT_ERROR is not None:
            raise RuntimeError("qiskit / qiskit-aer not available")

        qc_run = qc.copy() if hasattr(qc, "copy") else qc

        method = ""
        try:
            opts = getattr(self.simulator, "options", None)
            if opts is not None:
                try:
                    method = str(opts.get("method", "")).lower()
                except Exception:
                    method = str(getattr(opts, "method", "")).lower()
        except Exception:
            method = ""

        if "density" in method:
            try:
                qc_run.save_density_matrix(label="density_matrix")
            except Exception:
                try:
                    qc_run.save_state_density_matrix(label="density_matrix")
                except Exception:
                    pass
        elif "state" in method:
            try:
                qc_run.save_statevector(label="statevector")
            except Exception:
                try:
                    qc_run.save_state(label="statevector")
                except Exception:
                    pass

        def _to_ndarray(obj):
            if obj is None:
                return None
            if isinstance(obj, _np.ndarray):
                return _np.asarray(obj, dtype=_np.complex128)
            if hasattr(obj, "data"):
                try:
                    return _np.asarray(getattr(obj, "data"), dtype=_np.complex128)
                except Exception:
                    pass
            if hasattr(obj, "to_matrix"):
                try:
                    return _np.asarray(obj.to_matrix(), dtype=_np.complex128)
                except Exception:
                    pass
            try:
                return _np.asarray(obj, dtype=_np.complex128)
            except Exception:
                return None

        try:
            job = self.simulator.run(qc_run, shots=shots)
            result = job.result()
        except Exception as e:
            err = str(e).lower()
            if "contains invalid instructions" in err or "invalid instruction" in err:
                raise RuntimeError(
                    f"Aer rejected a save instruction for simulator method '{method}'."
                    " Make sure to add a save matching the method (save_density_matrix "
                    "for density_matrix, save_statevector for statevector). Original error:\n"
                    + str(e)
                ) from e
            raise RuntimeError(f"AerSimulator failed to run circuit: {e}") from e

        data = {}
        try:
            data = result.data() or {}
        except Exception:
            data = {}

        val = None
        if "density_matrix" in data:
            val = data["density_matrix"]
            arr = _to_ndarray(val)
            if arr is None:
                raise RuntimeError("Failed to convert DensityMatrix object to numpy array.")
            return _np.asarray(arr, dtype=_np.complex128)

        if "statevector" in data:
            sv = data["statevector"]
            sv_arr = _to_ndarray(sv)
            if sv_arr is None:
                raise RuntimeError("Failed to convert Statevector object to numpy array.")
            rho = _np.outer(sv_arr, _np.conjugate(sv_arr))
            return _np.asarray(rho, dtype=_np.complex128)

        try:
            rs = getattr(result, "results", None)
            if rs and len(rs) > 0:
                d0 = getattr(rs[0], "data", None) or {}
                if isinstance(d0, dict):
                    if "density_matrix" in d0:
                        arr = _to_ndarray(d0["density_matrix"])
                        if arr is not None:
                            return _np.asarray(arr, dtype=_np.complex128)
                    if "statevector" in d0:
                        sv_arr = _to_ndarray(d0["statevector"])
                        if sv_arr is not None:
                            return _np.asarray(_np.outer(sv_arr, _np.conjugate(sv_arr)), dtype=_np.complex128)
        except Exception:
            pass

        try:
            if hasattr(result, "get_density_matrix"):
                dm = result.get_density_matrix()
                dm_arr = _to_ndarray(dm)
                if dm_arr is not None:
                    return _np.asarray(dm_arr, dtype=_np.complex128)
        except Exception:
            pass
        try:
            if hasattr(result, "get_statevector"):
                sv = result.get_statevector()
                sv_arr = _to_ndarray(sv)
                if sv_arr is not None:
                    return _np.asarray(_np.outer(sv_arr, _np.conjugate(sv_arr)), dtype=_np.complex128)
        except Exception:
            pass

        raise RuntimeError(
            "Simulator result did not contain 'density_matrix' or 'statevector' in result.data(). "
            "Make sure you add a compatible save instruction before running the circuit. "
            "Use inspector script to see available keys."
        )
