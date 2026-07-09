import numpy as np
import math

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp

from block_encoding.oracles.prepare import build_prepare_oracle
from block_encoding.oracles.select import build_select_oracle


def encode_LCU(lcu: SparsePauliOp) -> tuple[QuantumCircuit, float]:
    """
    Encode a matrix in the form of a Linear Combination of Unitaries (LCU) into a QuantumCircuit using PREPARE and SELECT oracles.
    """
    num_qubit_index = (
        math.ceil(math.log2(len(lcu.coeffs))) if len(lcu.coeffs) > 1 else 0
    )
    num_qubits_unitary = lcu.num_qubits

    index_reg = QuantumRegister(num_qubit_index, "j")
    psi_reg = QuantumRegister(num_qubits_unitary)
    ancilla_reg = QuantumRegister(1, "a")

    circuit = QuantumCircuit(psi_reg, index_reg, ancilla_reg)

    prepare_oracle, factor = build_prepare_oracle(lcu)
    select_oracle = build_select_oracle(lcu)

    circuit.compose(prepare_oracle, index_reg, inplace=True)
    circuit.compose(
        select_oracle, index_reg[:] + psi_reg[:] + ancilla_reg[:], inplace=True
    )
    circuit.compose(prepare_oracle.inverse(), index_reg, inplace=True)

    return circuit, factor
