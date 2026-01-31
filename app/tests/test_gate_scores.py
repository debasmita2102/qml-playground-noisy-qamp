import numpy as np
from qiskit import QuantumCircuit

from app.backends.gate_scores import gate_scores_via_extractor

def test_gate_scores_via_extractor_basic():
    # ------------------
    # Tiny test circuit
    # ------------------
    qc = QuantumCircuit(1)
    qc.ry(0.3, 0)
    qc.rx(0.2, 0)

    scores = gate_scores_via_extractor(
        noise_type="depolarizing",
        depolarizing_probability=0.2,
        damping_rate=None,
        ibm_backend_name=None,
        quantum_circuit=qc,
    )

    assert isinstance(scores, list)
    assert len(scores) > 0

    for entry in scores:
        assert "gate" in entry
        assert "count" in entry
        assert "score" in entry

        assert isinstance(entry["gate"], str)
        assert isinstance(entry["count"], int)
        assert isinstance(entry["score"], float)

        assert entry["count"] > 0
        assert entry["score"] >= 0.0

    total_score = sum(e["score"] for e in scores)

    assert total_score > 0.0
    assert total_score < 1.0
