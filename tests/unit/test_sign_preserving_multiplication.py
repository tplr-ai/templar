from neurons.validator import sign_preserving_multiplication
import pytest


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (3, 4, 12),  # Both positive
        (-3, 4, -12),  # One negative
        (3, -4, -12),  # One negative
        (-3, -4, -12),  # Both negative
        (0, 5, 0),  # Zero case
        (5, 0, 0),  # Zero case
        (0, 0, 0),  # Zero case
        (1, 1, 1),  # Identity
        (-1, 1, -1),  # Negative identity
        (1, -1, -1),  # Negative identity
        (-1, -1, -1),  # Negative squared
    ],
)
def test_sign_preserving_multiplication(a, b, expected):
    assert sign_preserving_multiplication(a, b) == expected
