import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.quantum_info import SparsePauliOp

from block_encoding import encode_LCU


def build_qsvt_circuit(angles: list[float], matrix: SparsePauliOp):
    """
    Build the circuit.
    """
    circuit, alpha = encode_LCU(matrix)
    index_qreg = circuit.qregs[0]

    for angle in angles:

        circuit.x(index_qreg)
        circuit.mcp(angle, index_qreg[:-1], index_qreg[-1])
        circuit.x(index_qreg)

    return circuit, alpha
