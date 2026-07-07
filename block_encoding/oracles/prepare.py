import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from block_encoding import BinaryAmplitudeTree


def build_prepare_oracle(matrix: SparsePauliOp):
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
        if i == 0:
            circuit.ry(angles[0], i)
        else:
            for j, angle in enumerate(angles):
                ctrl_state = np.array(list(f"{j:0{i}b}")[::-1], dtype=int)
                zero_bits = np.where(ctrl_state == 0)[0]
                if zero_bits.size > 0:
                    circuit.x(zero_bits)
                circuit.mcry(angle, ctrl_qubits, i)
                if zero_bits.size > 0:
                    circuit.x(zero_bits)
        ctrl_qubits.append(i)

    return circuit
