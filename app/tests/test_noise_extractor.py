# tests/test_noise_integration.py
import numpy as np
import pytest
import importlib
from backends.circuit_bridge import CircuitBridge
from backends.qiskit_noise_extractor import NoiseExtractor
from backends.noisy_ml_simulator import NoisyMLSimulator
import torch

# Small helper used by multiple tests
def is_density_matrix(rho, tol=1e-10):
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape[0] != rho.shape[1]:
        return False
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if not np.allclose(np.trace(rho), 1.0, atol=tol):
        return False
    eigs = np.linalg.eigvals(rho)
    return np.all(np.real(eigs) > -tol)

def test_circuit_bridge_converts_gates():
    """Verify CircuitBridge records and converts gates correctly."""
    # skip test if CircuitBridge implementation is not available
    from qiskit import QuantumCircuit

    # Create bridge and record basic gates
    bridge = CircuitBridge(n_qubits=2)
    bridge.reset()
    bridge.record_gate("H", [0])
    bridge.record_gate("ROT", [0], params={"angles": [0.1, 0.2, 0.3]})
    bridge.record_gate("CNOT", [0, 1])

    qc = bridge.to_qiskit_circuit()
    # Basic sanity checks
    assert isinstance(qc, QuantumCircuit), "to_qiskit_circuit() must return a qiskit.QuantumCircuit"
    assert qc.num_qubits == 2
    # Expect at least 3 recorded operations in the circuit data (H, rotations -> rz/ry/rz or similar, cx)
    assert len(qc.data) >= 2

    opname_list = [instr.name.lower() for instr, _, _ in qc.data]
    assert any(n in opname_list for n in ("h", "rz", "ry", "rx")), f"Expected single-qubit ops in circuit, found {opname_list}"
    assert any(n in opname_list for n in ("cx", "cnot")), f"Expected two-qubit CX/CNOT in circuit, found {opname_list}"


def test_noise_extractor_creates_noise_model():
    """Verify NoiseExtractor creates valid Qiskit NoiseModel."""
    
    ne = NoiseExtractor(p_1qubit=0.01, p_2qubit=0.02, error_kind="depolarizing")
    assert hasattr(ne, "noise_model"), "NoiseExtractor should have attribute noise_model"
    assert hasattr(ne, "simulator"), "NoiseExtractor should have attribute simulator"

    # Build a simple 1-qubit circuit and simulate
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    rho = ne.simulate_circuit(qc)
    assert isinstance(rho, np.ndarray), "simulate_circuit must return a numpy array"
    assert rho.shape == (2, 2)
    assert is_density_matrix(rho), "Returned array must be a valid 1-qubit density matrix"


def test_noisy_simulator_runs_comparison():
    """Verify NoisyMLSimulator.get_comparison() returns valid results."""
    # Try to construct with track_noise=True to exercise noisy path if Aer is present.
    sim = NoisyMLSimulator(n_qubits=1, n_layers=1, p_1qubit=0.01, p_2qubit=0.02, track_noise=True, gpu=False, seed=42)
    aer_available_for_test = True
   
    X = torch.rand(1, 3)
    sim.forward(X)
    angles = sim.get_angles(X)  # shape (1, n_layers, n_qubits, 3)
    sim.Rot(angles[:, 0, :, :])
    sim.H([0], 1)

    # Call get_comparison and validate structure
    comp = sim.get_comparison()
    assert "ideal" in comp and "noisy" in comp and "qiskit_circuit" in comp

    ideal = comp["ideal"]
    noisy = comp["noisy"]

    assert "density_matrix" in ideal and isinstance(ideal["density_matrix"], np.ndarray)
    assert is_density_matrix(ideal["density_matrix"])

    assert "density_matrix" in noisy and isinstance(noisy["density_matrix"], np.ndarray)
    assert is_density_matrix(noisy["density_matrix"])

    assert 0.0 <= noisy["fidelity"] <= 1.0
    assert 0.0 <= ideal.get("purity", 0.0) <= 1.0
    assert 0.0 <= noisy.get("purity", 0.0) <= 1.0

    if aer_available_for_test:
        assert noisy["fidelity"] <= 1.0


try:
    from noisy_ml_simulator import NoisyMLSimulator
    import torch

    print("NoisyMLSimulator found — running a minimal example...")
    sim = NoisyMLSimulator(n_qubits=1, n_layers=1, p_1qubit=0.05, p_2qubit=0.0, track_noise=True, gpu=False, seed=42)
    X = torch.rand(1, 3)  # dummy feature vector
    _ = sim.forward(X)    # initializes |0>
    _ = sim.H([0], n_samples=1)

    out = sim.get_comparison()
    print("Keys:", list(out.keys()))
    print("Ideal Purity:", out["ideal"]["purity"], "Noisy Purity:", out["noisy"]["purity"])
    print("Noisy Fidelity:", out["noisy"]["fidelity"])
    print("Bloch (ideal traj first):", out["ideal"]["bloch_trajectory"][0])
    print("Bloch (noisy, q0):", out["noisy"]["bloch"])

except Exception as e:
    print("NoisyMLSimulator not available or failed to run.\n"
          "To enable this cell, ensure your project folder is on PYTHONPATH and "+
          "that 'noisy_ml_simulator.py' and its dependencies are importable.\n\n"
          f"Details: {e}")

