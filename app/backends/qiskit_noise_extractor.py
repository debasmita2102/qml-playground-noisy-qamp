# noise_models/qiskit_noise_extractor.py
"""
Noise extractor: builds a simple depolarizing / amplitude-damping NoiseModel and runs circuits using AerSimulator.
This version is a bit more tolerant to Qiskit/Aer variations and returns numpy complex128 arrays.

Added options:
 - error_kind: "depolarizing" (default) | "amplitude_damping"
 - p_amp: amplitude damping probability for single-qubit amplitude damping errors
"""
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
        # fallback namespace used by some qiskit installs
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
    """Build and run simple noise models (depolarizing and/or amplitude damping) using Qiskit Aer.

    Parameters
    ----------
    p_1qubit : float
        Depolarizing error probability for 1-qubit gates (default 0.001). Used when error_kind includes depolarizing.
    p_2qubit : float
        Depolarizing error probability for 2-qubit gates (default 0.01).
    error_kind : str
        One of "depolarizing" (default) or "amplitude_damping".
        - "depolarizing": single-qubit gates get depolarizing_error(p_1qubit)
        - "amplitude_damping": single-qubit gates get amplitude_damping_error(p_amp)
    p_amp : Optional[float]
        Amplitude damping probability to use when error_kind == "amplitude_damping".
        If not provided, p_1qubit is used as damping probability.
    """

    def __init__(
        self,
        p_1qubit: float = 0.001,
        p_2qubit: float = 0.01,
        error_kind: str = "depolarizing",
        p_amp: Optional[float] = None,
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

        # build a simple noise model and an AerSimulator configured for density-matrix
        self.noise_model = self._build_noise_model()
        # create simulator; allow user to override options later if needed
        # we use density_matrix method by default so we can extract density matrices
        self.simulator = AerSimulator(noise_model=self.noise_model, method="density_matrix")

    def _build_noise_model(self) -> NoiseModel:
        """Create a NoiseModel with errors applied to common 1- and 2-qubit gates.

        Single-qubit errors are chosen by `self.error_kind`. Two-qubit gates use depolarizing error.
        """
        # build single-qubit error according to requested kind
        if self.error_kind == "amplitude_damping":
            try:
                err_1q = noise.amplitude_damping_error(self.p_amp)
            except Exception as e:
                # some qiskit-aer versions might use a different path for amplitude damping error
                raise RuntimeError(f"Failed to create amplitude damping error: {e}") from e
        else:
            # default: depolarizing single-qubit error
            err_1q = noise.depolarizing_error(self.p_1qubit, 1)

        # two-qubit depolarizing error (for CX/CZ/SWAP)
        err_2q = noise.depolarizing_error(self.p_2qubit, 2)

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

        # attach single-qubit error to all common single-qubit gates
        for g in one_q_gates:
            nm.add_all_qubit_quantum_error(err_1q, g)

        # attach two-qubit depolarizing errors to common two-qubit gates
        for g in two_q_gates:
            nm.add_all_qubit_quantum_error(err_2q, g)

        return nm

    def build_noise_from_backend(self, backend) -> NoiseModel:
        """Build a NoiseModel from a real backend object (requires IBM provider backend)."""
        if _IMPORT_ERROR is not None:
            raise RuntimeError("qiskit not available")
        nm = NoiseModel.from_backend(backend)
        self.noise_model = nm
        self.simulator = AerSimulator(noise_model=self.noise_model, method="density_matrix")
        return nm

    def simulate_circuit(self, qc: QuantumCircuit, shots: int = 1) -> _np.ndarray:
        """
        Short, modern simulate_circuit for Aer builds that store final state under
        result.data()['statevector'] or result.data()['density_matrix'] (Statevector /
        DensityMatrix objects). Returns a numpy.complex128 density matrix.
        """
        if _IMPORT_ERROR is not None:
            raise RuntimeError("qiskit / qiskit-aer not available")

        import numpy as _np  # keep local alias consistent with module

        # defensive copy
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

        # add compatible save instruction
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

        # helper to convert qiskit objects -> numpy arrays
        def _to_ndarray(obj):
            if obj is None:
                return None
            # raw numpy array
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

        # run and extract
        try:
            job = self.simulator.run(qc_run, shots=shots)
            result = job.result()
        except Exception as e:
            # give a clear hint if we tried an incompatible save
            err = str(e).lower()
            if "contains invalid instructions" in err or "invalid instruction" in err:
                raise RuntimeError(
                    f"Aer rejected a save instruction for simulator method '{method}'."
                    " Make sure to add a save matching the method (save_density_matrix "
                    "for density_matrix, save_statevector for statevector). Original error:\n"
                    + str(e)
                ) from e
            raise RuntimeError(f"AerSimulator failed to run circuit: {e}") from e

        # result.data() should contain the object (Statevector or DensityMatrix)
        data = {}
        try:
            data = result.data() or {}
        except Exception:
            data = {}

        # prefer density_matrix, else statevector (both under result.data() or results[0].data)
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

        # try results[0].data() as fallback
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

        # try getters as last resort
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

        # nothing found -> helpful error
        raise RuntimeError(
            "Simulator result did not contain 'density_matrix' or 'statevector' in result.data(). "
            "Make sure you add a compatible save instruction before running the circuit. "
            "Use inspector script to see available keys."
        )
