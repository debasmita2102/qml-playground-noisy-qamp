from typing import Optional
import numpy as _np

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

from qiskit_aer.noise import amplitude_damping_error, phase_damping_error, depolarizing_error
import os

def _gate_error_to_depol_p(g_err, n_qubits):
        if g_err is None:
            return 0.0
        if n_qubits == 1:
            p = 2.0 * float(g_err)
        elif n_qubits == 2:
            p = (4.0 / 3.0) * float(g_err)
        else:
            d = 2 ** n_qubits
            denom = 1.0 - 1.0 / d
            p = float(g_err) / denom if denom > 0 else float(g_err)
        return max(0.0, min(1.0, p))

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

        # Build noise model & simulator
        self.noise_model = self._build_noise_model()
        self.simulator = AerSimulator(noise_model=self.noise_model, method="density_matrix")

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
    
    def get_backend(self, backend_name: str):
        """Return an IBM backend using QiskitRuntimeService (preferred) or fallback to IBMQ."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum_platform",
            token= os.getenv("QISKIT_IBM_RUNTIME_API_TOKEN"),
            instance=os.getenv("CRN"))
            return service.backend(backend_name)
        except Exception as e:
            raise RuntimeError(f"Unable to load backend '{backend_name}'. Ensure IBM credentials are set.") from e

    def get_backend_calibration_simple(self, backend, qubits: Optional[list] = None,
                                       gate_name: str = "cx", refresh: bool = False, verbose: bool = False):
        """
        Simple extractor for T1/T2 and gate errors using Backend.properties(refresh=...).
        Assumes per-qubit accessors props.t1(q) / props.t2(q) and props.gate_error(...) exist.
        Returns: {"t1": [...], "t2": [...], "gate_errors": {(q,): val, (q0,q1): val, ('global',): val}}
        """
        props = backend.properties(refresh=refresh)

        n_qubits = getattr(backend, "num_qubits", None)
        if n_qubits is None:
            # minimal fallback: try to infer from properties dict (rare)
            pd = props.to_dict()
            n_qubits = len(pd.get("qubits", []))
        qubits = list(qubits) if qubits is not None else list(range(n_qubits))

        t1_list = []
        t2_list = []
        for q in qubits:
            try:
                v = props.t1(q)
                t1_list.append(float(v) if v is not None else None)
            except Exception:
                t1_list.append(None)
            try:
                v = props.t2(q)
                t2_list.append(float(v) if v is not None else None)
            except Exception:
                t2_list.append(None)

        gate_errors = {}
        ge_acc = getattr(props, "gate_error", None)
        if callable(ge_acc):
            try:
                ge = ge_acc(gate_name)
                if isinstance(ge, dict):
                    for k, v in ge.items():
                        if isinstance(k, str):
                            try:
                                k_parsed = tuple(int(x.strip()) for x in k.strip("()[]").split(",") if x.strip() != "")
                            except Exception:
                                k_parsed = (k,)
                        elif isinstance(k, (list, tuple)):
                            k_parsed = tuple(int(x) for x in k)
                        else:
                            k_parsed = (k,)
                        gate_errors[k_parsed] = float(v) if v is not None else None
                elif isinstance(ge, (int, float)):
                    gate_errors[("global",)] = float(ge)
            except TypeError:
                for i in qubits:
                    for j in qubits:
                        if i >= j:
                            continue
                        try:
                            v = ge_acc(gate_name, qubits=[i, j])
                            if v is not None:
                                gate_errors[(i, j)] = float(v)
                        except Exception:
                            pass
            except Exception:
                pass

        if verbose:
            print("Backend calibration (simple):")
            print(f"  qubits: {qubits}")
            print(f"  T1: {t1_list[:min(8, len(t1_list))]}")
            print(f"  T2: {t2_list[:min(8, len(t2_list))]}")
            print(f"  gate_errors (sample): {dict(list(gate_errors.items())[:8])}")

        return {"t1": t1_list, "t2": t2_list, "gate_errors": gate_errors}
    
    def build_noise_model_from_cal_simple(self, cal: dict, default_p1: float = 0.001, default_p2: float = 0.01,
                                      one_q_gates=None, two_q_gate="cx"):
        if one_q_gates is None:
            one_q_gates = ["x", "u1", "u2", "u3", "h", "rx", "ry", "rz"]

        nm = NoiseModel()
        ge = cal.get("gate_errors", {}) or {}

        per_q = {k[0]: v for k, v in ge.items() if isinstance(k, tuple) and len(k) == 1 and isinstance(k[0], int)}
        global_1q = ge.get(("global",), ge.get(("global_1q",), None))

        if per_q:
            for q, err in per_q.items():
                p1 = _gate_error_to_depol_p(err, 1) if err is not None else default_p1
                for g in one_q_gates:
                    try:
                        nm.add_quantum_error(depolarizing_error(p1, 1), g, qubits=[q])
                    except Exception:
                        pass
        else:
            p1 = _gate_error_to_depol_p(global_1q, 1) if global_1q is not None else default_p1
            for g in one_q_gates:
                try:
                    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), g)
                except Exception:
                    pass

        attached_2q = False
        for k, err in ge.items():
            if isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, int) for x in k):
                p2 = _gate_error_to_depol_p(err, 2) if err is not None else default_p2
                try:
                    nm.add_quantum_error(depolarizing_error(p2, 2), two_q_gate, qubits=list(k))
                    attached_2q = True
                except Exception:
                    pass

        if not attached_2q:
            try:
                nm.add_all_qubit_quantum_error(depolarizing_error(default_p2, 2), two_q_gate)
            except Exception:
                pass

        return nm
    # Build T1/T2 -> amplitude+phase damping NoiseModel (simple)
    def build_noise_model_from_t1t2(self, cal: dict,
                                    one_q_gate_time_s: float,
                                    two_q_gate_time_s: float = 200e-9,
                                    default_2q_err: float = 0.02,
                                    one_q_gates=None,
                                    two_q_gate="cx"):
        if one_q_gates is None:
            one_q_gates = ["x", "u1", "u2", "u3", "h", "rx", "ry", "rz"]

        nm = NoiseModel()
        ge = cal.get("gate_errors", {}) or {}
        t1_list = cal.get("t1") or []
        t2_list = cal.get("t2") or []
        n_qubits = max(len(t1_list), len(t2_list)) if (t1_list or t2_list) else 0

        def _safe(x):
            try:
                return float(x)
            except Exception:
                return None

        for q in range(n_qubits):
            T1 = _safe(t1_list[q]) if q < len(t1_list) else None
            T2 = _safe(t2_list[q]) if q < len(t2_list) else None

            p_amp = None
            if T1 and T1 > 0:
                p_amp = 1.0 - _np.exp(-one_q_gate_time_s / T1)
                p_amp = max(0.0, min(1.0, p_amp))

            p_phase = None
            if T2 and T2 > 0:
                if T1 and T1 > 0:
                    denom = (1.0 / T2) - (1.0 / (2.0 * T1))
                    if denom > 0:
                        Tphi = 1.0 / denom
                        p_phase = 1.0 - _np.exp(-one_q_gate_time_s / Tphi)
                        p_phase = max(0.0, min(1.0, p_phase))
                if p_phase is None:
                    p_phase = 1.0 - _np.exp(-one_q_gate_time_s / T2)
                    p_phase = max(0.0, min(1.0, p_phase))

            if p_amp is not None and amplitude_damping_error is not None:
                err_amp = amplitude_damping_error(p_amp)
                for g in one_q_gates:
                    try:
                        nm.add_quantum_error(err_amp, g, qubits=[q])
                    except Exception:
                        pass

            if p_phase is not None and phase_damping_error is not None:
                err_phase = phase_damping_error(p_phase)
                for g in one_q_gates:
                    try:
                        nm.add_quantum_error(err_phase, g, qubits=[q])
                    except Exception:
                        pass

        attached = False
        for k, v in ge.items():
            if isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, int) for x in k):
                try:
                    p2_param = float(v) if v is not None else default_2q_err
                except Exception:
                    p2_param = default_2q_err
                p2_depol = max(0.0, min(1.0, (4.0 / 3.0) * p2_param))
                try:
                    nm.add_quantum_error(depolarizing_error(p2_depol, 2), two_q_gate, qubits=list(k))
                    attached = True
                except Exception:
                    pass

        if not attached:
            try:
                nm.add_all_qubit_quantum_error(depolarizing_error(default_2q_err, 2), two_q_gate)
            except Exception:
                pass

        return nm


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
