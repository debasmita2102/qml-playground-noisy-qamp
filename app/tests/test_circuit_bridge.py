import numpy as np
import pytest
import sys
import os

from qiskit import QuantumCircuit

test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.backends.circuit_bridge import CircuitBridge


@pytest.fixture
def skip_if_no_qiskit():
    if QuantumCircuit is None:
        pytest.skip("Qiskit not available")
    return True


class TestCircuitBridge:

    def test_initialization(self):
        bridge = CircuitBridge(n_qubits=3)
        assert bridge.n_qubits == 3
        assert bridge.gates == []

    def test_reset(self):
        bridge = CircuitBridge(n_qubits=2)
        bridge.record_gate("H", [0])
        bridge.record_gate("X", [1])
        assert len(bridge.gates) == 2
        bridge.reset()
        assert len(bridge.gates) == 0

    def test_single_qubit_pauli_gates(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=2)
        bridge.reset()
        bridge.record_gate("X", [0])
        bridge.record_gate("Y", [1])
        bridge.record_gate("Z", [0])
        bridge.record_gate("PAULIX", [1])

        qc = bridge.to_qiskit_circuit()
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2

        opnames = [instr.name.lower() for instr, _, _ in qc.data]
        assert opnames.count("x") == 2
        assert opnames.count("y") == 1
        assert opnames.count("z") == 1

    def test_hadamard_gate(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=3)
        bridge.reset()
        bridge.record_gate("H", [0, 1, 2])

        qc = bridge.to_qiskit_circuit()
        opnames = [instr.name.lower() for instr, _, _ in qc.data]
        assert opnames.count("h") == 3

    def test_rotation_gates(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=2)
        bridge.reset()
        bridge.record_gate("RX", [0], params={"theta": np.pi / 4})
        bridge.record_gate("RY", [1], params={"theta": np.pi / 2})
        bridge.record_gate("RZ", [0], params={"theta": np.pi})

        qc = bridge.to_qiskit_circuit()
        opnames = [instr.name.lower() for instr, _, _ in qc.data]
        assert "rx" in opnames
        assert "ry" in opnames
        assert "rz" in opnames

    def test_general_rotation_gate(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=1)
        bridge.reset()

        angles = [0.1, 0.2, 0.3]
        bridge.record_gate("ROT", [0], params={"angles": angles})
        bridge.record_gate("ROTATION", [0], params={"angles": angles})

        qc = bridge.to_qiskit_circuit()
        opnames = [instr.name.lower() for instr, _, _ in qc.data]

        assert opnames.count("rz") == 4
        assert opnames.count("ry") == 2

    def test_cnot_gate_various_formats(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=3)
        bridge.reset()
        bridge.record_gate("CNOT", [0, 1])
        bridge.record_gate("CX", [[1, 2], [2, 0]])

        qc = bridge.to_qiskit_circuit()
        opnames = [instr.name.lower() for instr, _, _ in qc.data]
        assert opnames.count("cx") == 3

    def test_cz_and_swap_gates(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=3)
        bridge.reset()
        bridge.record_gate("CZ", [0, 1])
        bridge.record_gate("SWAP", [1, 2])

        qc = bridge.to_qiskit_circuit()
        opnames = [instr.name.lower() for instr, _, _ in qc.data]
        assert "cz" in opnames
        assert "swap" in opnames

    def test_unknown_gate_handling(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=2)
        bridge.reset()
        bridge.record_gate("H", [0])
        bridge.record_gate("UNKNOWN_GATE", [0])
        bridge.record_gate("X", [1])

        qc = bridge.to_qiskit_circuit()
        assert len(qc.data) == 2

    def test_empty_circuit(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=2)
        bridge.reset()

        qc = bridge.to_qiskit_circuit()
        assert qc.num_qubits == 2
        assert len(qc.data) == 0

    def test_complex_circuit_sequence(self, skip_if_no_qiskit):
        bridge = CircuitBridge(n_qubits=3)
        bridge.reset()

        bridge.record_gate("H", [0, 1, 2])
        bridge.record_gate("ROT", [0], params={"angles": [0.1, 0.2, 0.3]})
        bridge.record_gate("CNOT", [0, 1])
        bridge.record_gate("RZ", [2], params={"theta": np.pi / 4})
        bridge.record_gate("CNOT", [1, 2])
        bridge.record_gate("X", [0])
        bridge.record_gate("CZ", [0, 2])

        qc = bridge.to_qiskit_circuit()
        assert qc.num_qubits == 3
        assert len(qc.data) >= 11
