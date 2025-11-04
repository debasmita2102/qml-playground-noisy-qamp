
# --------------------------------------------------------
# circuit_bridge.py
# --------------------------------------------------------

from typing import List, Dict, Any
from qiskit import QuantumCircuit
__all__ = ["CircuitBridge"]
class CircuitBridge:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.reset()

    def reset(self):
        self.gates = []

    def record_gate(self, name: str, qubits: List[int], params: Dict[str, Any] = None):
        if params is None:
            params = {}
        self.gates.append({"name": name, "qubits": list(qubits), "params": params})

    def to_qiskit_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for g in self.gates:
            name = g["name"].upper()
            qs = g["qubits"]
            p = g["params"]

            if name == "H":
                for q in qs:
                    qc.h(q)
            elif name in ("X", "PAULIX"):
                for q in qs:
                    qc.x(q)
            elif name in ("Y", "PAULIY"):
                for q in qs:
                    qc.y(q)
            elif name in ("Z", "PAULIZ"):
                for q in qs:
                    qc.z(q)
            elif name == "RX":
                theta = float(p.get("theta", 0.0))
                for q in qs:
                    qc.rx(theta, q)
            elif name == "RY":
                theta = float(p.get("theta", 0.0))
                for q in qs:
                    qc.ry(theta, q)
            elif name == "RZ":
                theta = float(p.get("theta", 0.0))
                for q in qs:
                    qc.rz(theta, q)
            elif name in ("ROT", "ROTATION"):
                angles = p.get("angles", None)
                if angles is None:
                    continue
                phi, theta, omega = float(angles[0]), float(angles[1]), float(angles[2])
                for q in qs:
                    qc.rz(phi, q)
                    qc.ry(theta, q)
                    qc.rz(omega, q)
            elif name in ("CNOT", "CX"):
                if len(qs) == 2 and isinstance(qs[0], int):
                    qc.cx(qs[0], qs[1])
                else:
                    for pair in qs:
                        qc.cx(pair[0], pair[1])
            elif name == "CZ":
                if len(qs) == 2 and isinstance(qs[0], int):
                    qc.cz(qs[0], qs[1])
                else:
                    for pair in qs:
                        qc.cz(pair[0], pair[1])
            elif name == "SWAP":
                if len(qs) == 2 and isinstance(qs[0], int):
                    qc.swap(qs[0], qs[1])
                else:
                    for pair in qs:
                        qc.swap(pair[0], pair[1])
            else:
                # unknown gate: skip
                continue

        return qc

