import cmath
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate
from qiskit.quantum_info import Pauli, SparsePauliOp


def pauli_rotation_circuit(pauli: Pauli) -> Gate:
    """
    Construct the `QuantumCircuit` for the rotation of a single Pauli string `e^(-i * Pauli)`.

    Args:
        pauli (Pauli): The Pauli string to evolve

    Returns:
        QuantumCircuit: The circuit of the Pauli rotation
    """
    nb_qubits = len(pauli)
    circuit = QuantumCircuit(nb_qubits)

    active_qubits = np.nonzero(np.logical_or(pauli.x, pauli.z))[0]
    if len(active_qubits) == 0:
        return circuit

    where_y = np.nonzero(np.logical_and(pauli.x, pauli.z))[0]
    where_x = np.nonzero(pauli.x)[0]

    if len(where_y) > 0:
        circuit.sdg(where_y)
    if len(where_x) > 0:
        circuit.h(where_x)

    for j in range(len(active_qubits) - 1):
        circuit.cx(active_qubits[j], active_qubits[j + 1])

    circuit_reversed = circuit.inverse()
    circuit.z(active_qubits[-1])
    circuit.compose(circuit_reversed, inplace=True)

    return circuit.to_gate(label=f"{pauli}")


def apply_phase(num_qubit_index: int, index: int, phase: float) -> Gate:
    """
    Apply a phase only if the register is in the good state. Uses one ancilla qubit.
    """
    circuit = QuantumCircuit(num_qubit_index + 1)
    ctrl_state = f"{index:0{num_qubit_index}b}"

    ctrl_qubits = list(range(num_qubit_index))

    circuit.mcx(ctrl_qubits, num_qubit_index, ctrl_state=ctrl_state)
    circuit.p(phase, num_qubit_index)
    circuit.mcx(ctrl_qubits, num_qubit_index, ctrl_state=ctrl_state)

    return circuit.to_gate(label=f"Phase {phase}")


def build_select_oracle(matrix: SparsePauliOp):
    """
    Prepare the SELECT oracle given a linear combinaison of unitaries (LCU) in order block-encode.
    """
    num_qubit_index = len(f"{len(matrix.coeffs):b}")
    num_qubits_unitary = matrix.num_qubits

    index_reg = QuantumRegister(num_qubit_index, "j")
    psi_reg = QuantumRegister(num_qubits_unitary)
    ancilla_reg = QuantumRegister(1, "a")

    circuit = QuantumCircuit(index_reg, psi_reg, ancilla_reg)

    for i, (coeff, pauli) in enumerate(zip(matrix.coeffs, matrix.paulis)):
        _, phase = cmath.polar(coeff)

        # Appliquer la phase si nécéssaire.
        if phase != 0:
            phase_gate = apply_phase(num_qubit_index, i, phase)
            circuit.compose(phase_gate, index_reg[:] + ancilla_reg[:], inplace=True)

        # Appliquer la rotation de la chaine de Pauli
        pauli_rotation = pauli_rotation_circuit(pauli)
        controlled_pauli_rotation = pauli_rotation.control(
            num_ctrl_qubits=num_qubit_index, ctrl_state=f"{i:0{num_qubit_index}b}"
        )
        circuit.compose(
            controlled_pauli_rotation, index_reg[:] + psi_reg[:], inplace=True
        )

    return circuit
