import pytest
import numpy as np

from qiskit.quantum_info import Statevector, SparsePauliOp

from block_encoding import build_prepare_oracle


@pytest.mark.parametrize(
    "matrices", [([1, 2, 3, 4, 5], ["XXX", "XYX", "XZZ", "IZZ", "ZYZ"])]
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

    print("ICIIIIIIIIIII")
    print("BONNE VALEUR")
    print(padded_expected)
    print("STATEVECTOR")
    print(state_vector)
    print("ICIIIIIIIIIII")

    assert np.allclose(np.abs(state_vector), padded_expected)
