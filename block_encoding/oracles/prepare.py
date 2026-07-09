import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.quantum_info import SparsePauliOp

from block_encoding.oracles.binary_tree import BinaryAmplitudeTree


def build_prepare_oracle(matrix: SparsePauliOp) -> tuple[Gate, float]:
    """
    Prepare the PREPARE oracle for a matrix given as a linear combinaison of Pauli strings.
    """
    weights = list(np.abs(np.asarray(matrix.coeffs)))

    tree = BinaryAmplitudeTree(weights)
    rotation_angles = tree.calculate_rotation_angles()

    num_qubits = len(rotation_angles)
    circuit = QuantumCircuit(num_qubits, name="PREPARE")

    ctrl_qubits = []
    for i, angles in enumerate(rotation_angles):
        target = num_qubits - 1 - i
        if i == 0:
            circuit.ry(angles[0], target)
        else:
            for j, angle in enumerate(angles):
                ctrl_state = np.array([int(b) for b in f"{j:0{i}b}"], dtype=int)
                zero_bits = [ctrl_qubits[k] for k in np.where(ctrl_state == 0)[0]]
                if zero_bits:
                    circuit.x(zero_bits)
                circuit.mcry(angle, ctrl_qubits, target)
                if zero_bits:
                    circuit.x(zero_bits)
        ctrl_qubits.append(target)

    return circuit.to_gate(label="PREPARE"), tree.s
