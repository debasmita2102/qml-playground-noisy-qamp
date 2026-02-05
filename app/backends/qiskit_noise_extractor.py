from __future__ import annotations

import os
import socket
import logging
from typing import Optional, Callable, Iterable

import numpy as np


try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        amplitude_damping_error,
        phase_damping_error,
    )
    import qiskit_aer.noise as noise
except Exception as e:
    raise RuntimeError(
        "NoiseExtractor requires qiskit and qiskit-aer "
        "(pip install qiskit qiskit-aer)"
    ) from e

__all__ = ["NoiseExtractor"]
_logger = logging.getLogger(__name__)


def _clip_prob(p):
    try:
        return max(0.0, min(1.0, float(p)))
    except Exception:
        return 0.0


def _gate_error_to_depol_p(g_err, n_qubits):
    if g_err is None:
        return 0.0
    g_err = float(g_err)
    if n_qubits == 1:
        p = 2.0 * g_err
    elif n_qubits == 2:
        p = (4.0 / 3.0) * g_err
    else:
        d = 2 ** n_qubits
        denom = 1.0 - 1.0 / d
        p = g_err / denom if denom > 0 else g_err
    return _clip_prob(p)



class NoiseExtractor:
    """
    Unified noise-model + density-matrix noise toolkit.

    Tensor convention:
      (... ⊗ q2 ⊗ q1 ⊗ q0)
    """


    def __init__(
        self,
        p_1qubit: float = 1e-3,
        p_2qubit: float = 1e-2,
        error_kind: str = "depolarizing",
        p_amp: Optional[float] = None,
        p_phase: Optional[float] = None,
        ibm_backend_name: Optional[str] = None,
        provider_loader: Optional[Callable] = None,
        allow_fake_provider: bool = False,
    ):
        self.p_1qubit = _clip_prob(p_1qubit)
        self.p_2qubit = _clip_prob(p_2qubit)
        self.error_kind = error_kind.lower()

        self.p_amp = _clip_prob(p_amp if p_amp is not None else self.p_1qubit)
        self.p_phase = _clip_prob(p_phase if p_phase is not None else self.p_1qubit)

        self.ibm_backend_name = ibm_backend_name
        self._provider_loader = provider_loader
        self.allow_fake_provider = allow_fake_provider

        self.ibm_backend = None
        self.ibm_backend_error = None

        # Default synthetic noise
        self.noise_model = self._build_synthetic_noise_model()
        self.simulator = AerSimulator(
            noise_model=self.noise_model,
            method="density_matrix",
        )

        # Optional backend override
        if self.ibm_backend_name:
            try:
                self.load_ibm_backend(self.ibm_backend_name)
            except Exception as exc:
                self.ibm_backend_error = str(exc)
                _logger.warning(
                    "IBM backend unavailable, using synthetic noise: %s", exc
                )

    
    @staticmethod
    def _kraus_amplitude_damping(gamma):
        g = _clip_prob(gamma)
        return [
            np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex),
            np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex),
        ]

    @staticmethod
    def _kraus_phase_damping(p):
        p = _clip_prob(p)
        return [
            np.sqrt(1 - p) * np.eye(2, dtype=complex),
            np.sqrt(p) * np.diag([1, 0]),
            np.sqrt(p) * np.diag([0, 1]),
        ]

    @staticmethod
    def _kraus_depolarizing(p):
        p = _clip_prob(p)
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        return [
            np.sqrt(1 - 3 * p / 4) * I,
            np.sqrt(p / 4) * X,
            np.sqrt(p / 4) * Y,
            np.sqrt(p / 4) * Z,
        ]

 
    def _build_synthetic_noise_model(self) -> NoiseModel:
        nm = NoiseModel()

        # 1Q error
        if self.error_kind == "amplitude_damping":
            try:
                err_1q = amplitude_damping_error(self.p_amp)
            except Exception:
                err_1q = noise.kraus_error(
                    self._kraus_amplitude_damping(self.p_amp)
                )

        elif self.error_kind == "phase_damping":
            try:
                err_1q = phase_damping_error(self.p_phase)
            except Exception:
                err_1q = noise.kraus_error(
                    self._kraus_phase_damping(self.p_phase)
                )

        else:
            try:
                err_1q = depolarizing_error(self.p_1qubit, 1)
            except Exception:
                err_1q = noise.kraus_error(
                    self._kraus_depolarizing(self.p_1qubit)
                )

        err_2q = depolarizing_error(self.p_2qubit, 2)

        one_q_gates = [
            "x", "y", "z", "h",
            "s", "sdg", "t", "tdg",
            "rx", "ry", "rz",
            "u1", "u2", "u3",
        ]
        two_q_gates = ["cx", "cz", "swap"]

        for g in one_q_gates:
            nm.add_all_qubit_quantum_error(err_1q, g)

        for g in two_q_gates:
            nm.add_all_qubit_quantum_error(err_2q, g)

        return nm

  
    @staticmethod
    def _ensure_ibm_dns():
        for host in ["quantum.cloud.ibm.com", "auth.quantum-computing.ibm.com"]:
            try:
                socket.gethostbyname(host)
                return
            except Exception:
                pass
        raise RuntimeError("IBM Quantum DNS resolution failed")
    

    def load_ibm_backend(
        self,
        backend_name: str,
        provider_loader: Optional[Callable] = None,
        allow_fake_provider: Optional[bool] = None,
    ):
        name = backend_name.strip()
        allow_fake_provider = (
            self.allow_fake_provider
            if allow_fake_provider is None
            else allow_fake_provider
        )

        provider = None
        errors = []

        if provider_loader or self._provider_loader:
            try:
                provider = (provider_loader or self._provider_loader)()
            except Exception as e:
                errors.append(e)

        if provider is None:
            try:
                self._ensure_ibm_dns()
                from qiskit_ibm_runtime import QiskitRuntimeService
                provider = QiskitRuntimeService()
            except Exception as e:
                errors.append(e)

        if provider is None and allow_fake_provider:
            from qiskit.providers.fake_provider import FakeProvider
            provider = FakeProvider()

        if provider is None:
            raise RuntimeError(f"Cannot load IBM backend: {errors}")

        if hasattr(provider, "get_backend"):
            backend = provider.get_backend(name)
        elif hasattr(provider, "backend"):
            backend = provider.backend(name)
        else:
            raise RuntimeError(
                "Provider does not expose get_backend() or backend()"
            )

        self.ibm_backend = backend
        self.ibm_backend_name = name

        self.noise_model = NoiseModel.from_backend(backend)
        self.simulator = AerSimulator(
            noise_model=self.noise_model,
            method="density_matrix",
        )
        return backend


    def get_backend_calibration_simple(
        self,
        backend,
        qubits: Optional[list] = None,
        gate_name: str = "cx",
        refresh: bool = False,
    ):
        props = backend.properties(refresh=refresh)
        n = backend.num_qubits
        qubits = list(range(n)) if qubits is None else list(qubits)

        t1 = [props.t1(q) for q in qubits]
        t2 = [props.t2(q) for q in qubits]

        gate_errors = {}
        try:
            ge = props.gate_error(gate_name)
            if isinstance(ge, dict):
                for k, v in ge.items():
                    gate_errors[tuple(k)] = float(v)
            else:
                gate_errors[("global",)] = float(ge)
        except Exception:
            pass

        return {"t1": t1, "t2": t2, "gate_errors": gate_errors}

    def build_noise_model_from_cal_simple(
        self,
        cal: dict,
        default_p1=1e-3,
        default_p2=1e-2,
        one_q_gates=None,
        two_q_gate="cx",
    ):
        if one_q_gates is None:
            one_q_gates = ["x", "h", "rx", "ry", "rz"]

        nm = NoiseModel()
        ge = cal.get("gate_errors", {}) or {}

        global_1q = ge.get(("global",), None)

        for k, v in ge.items():
            if k == ("global",):
                continue
            if isinstance(k, tuple) and len(k) == 1 and isinstance(k[0], int):
                p1 = _gate_error_to_depol_p(v, 1)
                for g in one_q_gates:
                    nm.add_quantum_error(
                        depolarizing_error(p1, 1),
                        g,
                        qubits=[k[0]],
                    )

        if global_1q is not None:
            p1 = _gate_error_to_depol_p(global_1q, 1)
        else:
            p1 = default_p1

        for g in one_q_gates:
            nm.add_all_qubit_quantum_error(
                depolarizing_error(p1, 1),
                g,
            )

        attached_2q = False
        for k, v in ge.items():
            if isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, int) for x in k):
                p2 = _gate_error_to_depol_p(v, 2)
                nm.add_quantum_error(
                    depolarizing_error(p2, 2),
                    two_q_gate,
                    qubits=list(k),
                )
                attached_2q = True

        if not attached_2q:
            nm.add_all_qubit_quantum_error(
                depolarizing_error(default_p2, 2),
                two_q_gate,
            )

        return nm


    def build_noise_model_from_t1t2(
        self,
        cal: dict,
        one_q_gate_time_s: float,
        two_q_gate_time_s: float = 200e-9,
        default_2q_err: float = 0.02,
        one_q_gates=None,
        two_q_gate="cx",
    ):
        if one_q_gates is None:
            one_q_gates = ["x", "h", "rx", "ry", "rz"]

        nm = NoiseModel()
        t1 = cal.get("t1", [])
        t2 = cal.get("t2", [])
        ge = cal.get("gate_errors", {})

        for q in range(len(t1)):
            T1, T2 = t1[q], t2[q]
            p_amp = (
                1 - np.exp(-one_q_gate_time_s / T1)
                if T1 and T1 > 0
                else None
            )
            p_phase = (
                1 - np.exp(-one_q_gate_time_s / T2)
                if T2 and T2 > 0
                else None
            )

            err = None
            if p_amp is not None:
                err = amplitude_damping_error(_clip_prob(p_amp))
            if p_phase is not None:
                ph = phase_damping_error(_clip_prob(p_phase))
                err = err.compose(ph) if err else ph

            if err:
                for g in one_q_gates:
                    nm.add_quantum_error(err, g, qubits=[q])

        for k, v in ge.items():
            if len(k) == 2:
                p2 = _gate_error_to_depol_p(v, 2)
                nm.add_quantum_error(
                    depolarizing_error(p2, 2),
                    two_q_gate,
                    qubits=list(k),
                )

        return nm

   
    @staticmethod
    def _validate_density_matrix(rho):
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("Invalid density matrix")

    @staticmethod
    def _num_qubits(rho):
        return int(np.log2(rho.shape[0]))

    @staticmethod
    def _full_kron(op, target, n):
        out = 1
        for q in reversed(range(n)):
            out = np.kron(out, op if q == target else np.eye(2))
        return out

    def _apply_kraus_local(self, rho, ks, target):
        n = self._num_qubits(rho)
        out = np.zeros_like(rho, dtype=complex)
        for K in ks:
            Kf = self._full_kron(K, target, n)
            out += Kf @ rho @ Kf.conj().T
        return out

    def apply_local_noise(self, rho, targets=None):
        self._validate_density_matrix(rho)
        n = self._num_qubits(rho)
        targets = range(n) if targets is None else targets

        if self.error_kind == "amplitude_damping":
            ks = self._kraus_amplitude_damping(self.p_amp)
        elif self.error_kind == "phase_damping":
            ks = self._kraus_phase_damping(self.p_phase)
        else:
            ks = self._kraus_depolarizing(self.p_1qubit)

        out = rho.copy()
        for t in targets:
            out = self._apply_kraus_local(out, ks, t)
        return out


    def simulate_circuit(self, qc: QuantumCircuit, shots: int = 1):
        qc = qc.copy()
        qc.save_density_matrix()
        result = self.simulator.run(qc, shots=shots).result()
        dm = result.data()["density_matrix"]
        return np.asarray(dm.data, dtype=complex)