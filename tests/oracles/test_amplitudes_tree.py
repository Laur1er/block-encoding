import pytest
import numpy as np
from itertools import chain

from block_encoding import BinaryAmplitudeTree


@pytest.mark.parametrize(
    "coeffs_and_answers",
    [
        (
            [2, 3, 4, 5, 6],
            [
                [2 * np.arccos(np.sqrt(14 / 20))],
                [2 * np.arccos(np.sqrt(5 / 14)), 2 * np.arccos(1)],
                [
                    2 * np.arccos(np.sqrt(2 / 5)),
                    2 * np.arccos(np.sqrt(4 / 9)),
                    2 * np.arccos(np.sqrt(1)),
                    0,
                ],
            ],
        ),
    ],
)
def test_angle_computing(coeffs_and_answers: tuple):
    coeffs, answer = coeffs_and_answers
    tree = BinaryAmplitudeTree(coeffs)
    angles = tree.calculate_rotation_angles()

    flat_angles = np.array(list(chain.from_iterable(angles)))
    flat_answer = np.array(list(chain.from_iterable(answer)))

    assert np.allclose(flat_angles, flat_answer)
