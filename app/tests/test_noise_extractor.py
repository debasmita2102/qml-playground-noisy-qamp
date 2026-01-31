import numpy as np
import pytest
import sys
import os

from qiskit import QuantumCircuit

test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.backends.qiskit_noise_extractor import NoiseExtractor
from app.backends.noisy_ml_simulator import NoisyMLSimulator


def is_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> bool:
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape[0] != rho.shape[1]:
        return False
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if not np.allclose(np.trace(rho), 1.0, atol=tol):
        return False
    eigs = np.linalg.eigvals(rho)
    return np.all(np.real(eigs) > -tol)


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    if state1.ndim == 1:
        state1 = np.outer(state1, np.conj(state1))
    if state2.ndim == 1:
        state2 = np.outer(state2, np.conj(state2))

    eigvals, eigvecs = np.linalg.eigh(state1)
    eigvals = np.clip(eigvals, 0.0, None)

    sqrt_rho1 = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    M = sqrt_rho1 @ state2 @ sqrt_rho1

    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.clip(eigvals_M, 0.0, None)

    return float(np.sum(np.sqrt(eigvals_M)) ** 2)


def compute_purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


@pytest.fixture
def skip_if_no_qiskit():
    if QuantumCircuit is None:
        pytest.skip("Qiskit not available")
    return True


class TestNoiseExtractor:

    def test_simulate_single_qubit_circuit(self, skip_if_no_qiskit):
        ne = NoiseExtractor(p_1qubit=0.01, p_2qubit=0.02)
        qc = QuantumCircuit(1)
        qc.h(0)

        rho = ne.simulate_circuit(qc)
        assert rho.shape == (2, 2)
        assert is_density_matrix(rho)

    def test_simulate_two_qubit(self, skip_if_no_qiskit):
        ne = NoiseExtractor(p_1qubit=0.01, p_2qubit=0.02)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        rho = ne.simulate_circuit(qc)
        assert rho.shape == (4, 4)
        assert is_density_matrix(rho)

    def test_noise_reduces_purity(self, skip_if_no_qiskit):
        ne_noisy = NoiseExtractor(p_1qubit=0.1, p_2qubit=0.2)
        ne_ideal = NoiseExtractor(p_1qubit=0.0, p_2qubit=0.0)

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(np.pi / 4, 0)

        purity_noisy = compute_purity(ne_noisy.simulate_circuit(qc))
        purity_ideal = compute_purity(ne_ideal.simulate_circuit(qc))

        assert purity_ideal > 0.99
        assert purity_noisy < purity_ideal

    def test_noise_reduces_fidelity(self, skip_if_no_qiskit):
        ne_noisy = NoiseExtractor(p_1qubit=0.05, p_2qubit=0.1)
        ne_ideal = NoiseExtractor(p_1qubit=0.0, p_2qubit=0.0)

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rx(np.pi / 3, 0)

        fidelity = compute_fidelity(
            ne_ideal.simulate_circuit(qc),
            ne_noisy.simulate_circuit(qc),
        )

        assert 0.0 < fidelity < 1.0
        assert fidelity > 0.8

    def test_different_noise_types(self, skip_if_no_qiskit):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rx(np.pi / 4, 0)

        rho_depol = NoiseExtractor(0.1, error_kind="depolarizing").simulate_circuit(qc)
        rho_amp = NoiseExtractor(0.1, error_kind="amplitude_damping").simulate_circuit(qc)
        rho_phase = NoiseExtractor(0.1, error_kind="phase_damping").simulate_circuit(qc)

        assert not np.allclose(rho_depol, rho_amp)
        assert not np.allclose(rho_depol, rho_phase)
        assert not np.allclose(rho_amp, rho_phase)


class TestNoisyMLSimulator:

    def test_module_can_be_imported(self):
        assert NoisyMLSimulator is not None
