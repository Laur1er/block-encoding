import pytest
import numpy as np

from qiskit.quantum_info import Statevector, SparsePauliOp

from block_encoding import build_prepare_oracle


@pytest.mark.parametrize(
    "matrices",
    [
        ([1, 2, 3, 4, 5], ["XXX", "XYX", "XZZ", "IZZ", "ZYZ"]),
        ([2, 5, 12, -3], ["XX", "ZI", "ZZ", "YY"]),
        ([-3, 7, 0.5, 21, 13, 0.3], ["XXX", "XYX", "XZZ", "IZZ", "ZYZ", "ZZZ"]),
        (
            [-0.05, 1.75, -0.5, -2.75, 10, 0.3, 2.43, 0.5, -6],
            ["XXX", "XYX", "XZZ", "IZZ", "ZYZ", "ZZZ", "XIX", "YIY", "YYY"],
        ),
        (
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ["XXX", "XYX", "XZZ", "IZZ", "ZYZ", "XII", "XXI", "IIY", "IXI", "IIZ"],
        ),
    ],
)
def test_prepare_oracle(matrices: tuple):
    coeffs, paulis = matrices
    matrix = SparsePauliOp(paulis, coeffs)
    circuit = build_prepare_oracle(matrix)

    state_vector = Statevector(circuit).data

    total = sum(np.abs(coeffs))
    expected = np.sqrt(np.abs(np.array(coeffs)) / total)

    padded_expected = np.zeros(2**circuit.num_qubits)
    padded_expected[: len(expected)] = expected

    assert np.allclose(np.abs(state_vector), padded_expected)
