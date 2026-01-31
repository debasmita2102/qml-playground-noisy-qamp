import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator

from app.backends.noisy_ml_simulator import NoiseExtractor

def _is_valid_density_matrix(rho, atol=1e-8):
    assert isinstance(rho, np.ndarray)
    assert rho.ndim == 2
    assert rho.shape[0] == rho.shape[1]

    assert np.allclose(rho, rho.conj().T, atol=atol)

    tr = np.trace(rho)
    assert np.allclose(tr, 1.0, atol=atol)

    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-10)


def _rebuild_simulator(ne: NoiseExtractor):
    ne.simulator = AerSimulator(
        noise_model=ne.noise_model,
        method="density_matrix",
    )


def test_noise_model_from_synthetic_calibration_1q():
    """
    1-qubit test:
    - Build synthetic T1/T2 calibration
    - Run a single H gate
    - Assert valid density matrix
    - Assert fidelity strictly < 1
    """

    cal = {
        "t1": [50e-6],
        "t2": [70e-6],
        "gate_errors": {("global",): 0.01},
    }

    ne = NoiseExtractor(p_1qubit=0.0, p_2qubit=0.0)

    ne.noise_model = ne.build_noise_model_from_t1t2(
        cal,
        one_q_gate_time_s=5e-6,   
    )

    _rebuild_simulator(ne)

    qc = QuantumCircuit(1)
    qc.h(0)

    rho_noisy = ne.simulate_circuit(qc)

    psi_ideal = Statevector.from_instruction(qc)
    rho_ideal = np.outer(psi_ideal.data, psi_ideal.data.conj())

    _is_valid_density_matrix(rho_noisy)

    fid = state_fidelity(rho_noisy, rho_ideal)
    assert fid < 1.0


def test_noise_model_from_synthetic_calibration_2q():
    """
    2-qubit test:
    - Synthetic calibration with 2Q gate error
    - Bell-state circuit
    - Assert valid density matrix
    - Assert fidelity strictly < 1
    """

    cal = {
        "t1": [40e-6, 45e-6],
        "t2": [60e-6, 55e-6],
        "gate_errors": {
            ("global",): 0.005,
            (0, 1): 0.02,
        },
    }

    ne = NoiseExtractor(p_1qubit=0.0, p_2qubit=0.0)

    ne.noise_model = ne.build_noise_model_from_t1t2(
        cal,
        one_q_gate_time_s=5e-6,    
        two_q_gate_time_s=5e-6,
    )

    _rebuild_simulator(ne)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    rho_noisy = ne.simulate_circuit(qc)

    psi_ideal = Statevector.from_instruction(qc)
    rho_ideal = np.outer(psi_ideal.data, psi_ideal.data.conj())

    _is_valid_density_matrix(rho_noisy)

    fid = state_fidelity(rho_noisy, rho_ideal)
    assert fid < 1.0
