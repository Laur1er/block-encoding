import pytest
import numpy as np

from qiskit.quantum_info import SparsePauliOp, Operator
from block_encoding import encode_LCU


@pytest.mark.parametrize(
    "matrix",
    [
        SparsePauliOp(coeffs=[1, 2, 3, 4, 5], data=["XXX", "XYX", "XZZ", "IZZ", "ZYZ"]),
        SparsePauliOp(coeffs=[2, 5, 12, -3], data=["XX", "ZI", "ZZ", "YY"]),
        SparsePauliOp(
            coeffs=[-3, 7, 0.5, 21, 13, 0.3],
            data=["XXX", "XYX", "XZZ", "IZZ", "ZYZ", "ZZZ"],
        ),
        SparsePauliOp(
            coeffs=[-0.05, 1.75, -0.5, -2.75, 10, 0.3, 2.43, 0.5, -6],
            data=["XXX", "XYX", "XZZ", "IZZ", "ZYZ", "ZZZ", "XIX", "YIY", "YYY"],
        ),
        SparsePauliOp(
            coeffs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            data=["XXX", "XYX", "XZZ", "IZZ", "ZYZ", "XII", "XXI", "IIY", "IXI", "IIZ"],
        ),
    ],
)
def test_block_encoding(matrix: SparsePauliOp):

    h, l = matrix.to_matrix().shape
    circuit, s = encode_LCU(matrix)
    matrix_circuit = Operator(circuit).data

    assert np.allclose(matrix.to_matrix() / s, matrix_circuit[:h, :l])
