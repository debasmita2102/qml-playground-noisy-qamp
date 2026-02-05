from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, state_fidelity
import numpy as np
from app.backends.qiskit_noise_extractor import NoiseExtractor


__all__ = ["gate_scores_via_extractor"]

def gate_scores_via_extractor(
    *,
    noise_type: str,
    depolarizing_probability: float | None,
    damping_rate: float | None,
    ibm_backend_name: str | None,
    quantum_circuit,
):
    """
    Returns:
    [
      {"gate": "ry", "count": 3, "score": 0.23},
      ...
    ]
    """


    extractor = NoiseExtractor(
        error_kind=noise_type,
        p_1qubit=depolarizing_probability,
        p_amp=damping_rate,
    )

    ideal_qc = quantum_circuit.copy()
    ideal_qc.save_density_matrix()

    ideal_sim = AerSimulator(method="density_matrix")
    ideal_result = ideal_sim.run(ideal_qc).result()

    rho_ideal = DensityMatrix(
        ideal_result.data(0)["density_matrix"]
    )


    if ibm_backend_name:
        backend = extractor.get_backend(ibm_backend_name)
        cal = extractor.get_backend_calibration_simple(backend)
        noise_model = extractor.build_noise_model_from_cal_simple(cal)
    else:
        noise_model = extractor.build_noise_model_from_cal_simple(
            {"gate_errors": {}}
        )

    noisy_qc = quantum_circuit.copy()
    noisy_qc.save_density_matrix()

    noisy_sim = AerSimulator(
        noise_model=noise_model,
        method="density_matrix",
    )
    noisy_result = noisy_sim.run(noisy_qc).result()

    rho_noisy = DensityMatrix(
        noisy_result.data(0)["density_matrix"]
    )

    fid = float(state_fidelity(rho_ideal, rho_noisy))
    total_damage = 1.0 - fid

    gate_counts = {}
    for inst, _, _ in quantum_circuit.data:
        gate_counts[inst.name] = gate_counts.get(inst.name, 0) + 1

    total_gates = sum(gate_counts.values()) or 1

    scores = []
    for gate, count in gate_counts.items():
        scores.append({
            "gate": gate,
            "count": count,
            "score": total_damage * (count / total_gates),
        })

    return scores
