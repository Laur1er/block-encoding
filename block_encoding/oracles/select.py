import cmath
import math
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate
from qiskit.quantum_info import Pauli, SparsePauliOp


def diagonalise_using_pauli_gadjets(pauli: Pauli) -> Gate:
    """
    Construct the `QuantumCircuit` for the rotation of a single Pauli string.

    Args:
        pauli (Pauli): The Pauli string to evolve

    Returns:
        Gate: The gate of the Pauli rotation
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

    return (circuit, active_qubits[-1])


def apply_phase(num_qubit_index: int, index: int, phase: float) -> Gate:
    """
    Apply the phase of the coefficient only if the register is in the good state.
    """
    circuit = QuantumCircuit(num_qubit_index)

    ctrl_qubits = list(range(num_qubit_index))

    mask = np.where(
        np.array(list(f"{index:0{num_qubit_index}b}")[::-1]).astype(int) == 0
    )[0]
    if mask.size > 0:
        circuit.x(mask)
    circuit.mcp(phase, ctrl_qubits[:-1], ctrl_qubits[-1])
    if mask.size > 0:
        circuit.x(mask)

    return circuit.to_gate(label=f"Phase {phase:.2f}")


def build_select_oracle(matrix: SparsePauliOp):
    """
    Prepare the SELECT oracle given a linear combinaison of unitaries (LCU) in order block-encode.
    """
    num_qubit_index = (
        math.ceil(math.log2(len(matrix.coeffs))) if len(matrix.coeffs) > 1 else 0
    )
    num_qubits_unitary = matrix.num_qubits

    index_reg = QuantumRegister(num_qubit_index, "j")
    state_reg = QuantumRegister(num_qubits_unitary)

    circuit = QuantumCircuit(index_reg, state_reg)

    for i, (coeff, pauli) in enumerate(zip(matrix.coeffs, matrix.paulis)):
        _, phase = cmath.polar(coeff)

        # Appliquer la phase si nécéssaire.
        if phase != 0:
            phase_gate = apply_phase(num_qubit_index, i, phase)
            circuit.compose(phase_gate, index_reg[:], inplace=True)

        # Appliquer la rotation de la chaine de Pauli
        diagonalisation, active_qubit = diagonalise_using_pauli_gadjets(pauli)
        circuit.compose(diagonalisation, state_reg[:], inplace=True)

        # Mask car pas ctrl_state
        mask = np.where(
            np.array(list(f"{i:0{num_qubit_index}b}")[::-1]).astype(int) == 0
        )[0]
        if mask.size > 0:
            circuit.x(index_reg[mask.tolist()])

        circuit.mcp(
            np.pi,
            index_reg[:],
            state_reg[active_qubit],
        )
        if mask.size > 0:
            circuit.x(index_reg[mask.tolist()])

        circuit.compose(diagonalisation.inverse(), state_reg[:], inplace=True)

    return circuit
