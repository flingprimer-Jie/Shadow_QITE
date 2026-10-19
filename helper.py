"""Minimal circuit helpers for the experimental single-qubit Shadow-QITE demo.

These helpers intentionally do not implement general multi-qubit QITE.
"""

import numpy as np
from pyquil.gates import H, RX, RY, RZ


_PAULI_ROTATIONS = {"X": RX, "Y": RY, "Z": RZ}


def ansatz(program, qbits):
    """Prepare |+> on each selected qubit, giving a simple initial ansatz."""
    for qubit in qbits:
        program += H(qubit)
    return program


def propagate(program, alist, qbits):
    """Append previously computed single-qubit Pauli rotations.

    Each entry in ``alist`` is (axis, angle_in_radians, logical_qubit_index).
    """
    for axis, angle, index in alist:
        if axis not in _PAULI_ROTATIONS:
            raise ValueError(f"Unsupported rotation axis: {axis!r}")
        if not 0 <= index < len(qbits):
            raise ValueError(f"Logical qubit index {index} is out of range")
        program += _PAULI_ROTATIONS[axis](float(angle), qbits[index])
    return program


def update_alist(sigma_expectation, alist, db, delta, legacy_hm):
    """Append a first-order, regularized single-qubit imaginary-time step.

    Solve the 4x4 linear system in the {I, X, Y, Z} Pauli basis. The resulting
    Pauli coefficients specify small unitary rotations whose first-order
    change approximates normalized imaginary-time evolution for a pure qubit.

    ``legacy_hm`` matches the interface used in qite_shadow.py:
        [ [[pauli_index], [coefficient]], ... ]
    where 1=X, 2=Y, 3=Z. ``db`` is the imaginary-time step and ``delta`` is
    nonnegative Tikhonov regularization (to handle shot noise / singularity).

    This is an educational prototype, not a general multi-qubit QITE solver.
    """
    if db <= 0:
        raise ValueError("Imaginary-time step db must be positive")
    if delta < 0:
        raise ValueError("Regularization delta must be nonnegative")

    measured = np.asarray(sigma_expectation, dtype=complex)
    if measured.shape != (4,):
        raise ValueError("Expected 4 Pauli expectation values [I, X, Y, Z]")
    bloch = np.real(measured[1:]).astype(float)
    if not np.all(np.isfinite(bloch)):
        raise ValueError("Non-finite Pauli expectation value")

    # Classical-shadow shot noise can put a measured Bloch vector outside
    # the physical Bloch ball; project it back before constructing S.
    norm = np.linalg.norm(bloch)
    if norm > 1.0:
        bloch /= norm

    h = np.zeros(3, dtype=float)
    for pauli_indices, coefficients in legacy_hm:
        if len(pauli_indices) != len(coefficients):
            raise ValueError("Pauli indices and coefficients have different lengths")
        for p, coefficient in zip(pauli_indices, coefficients):
            if p not in (1, 2, 3):
                raise ValueError(f"Unsupported single-qubit Pauli index: {p}")
            h[p - 1] += float(np.real(coefficient))

    # Re <sigma_i sigma_j> in the ordered [I, X, Y, Z] basis.
    overlap = np.eye(4)
    overlap[0, 1:] = bloch
    overlap[1:, 0] = bloch
    # Im <sigma_i H> is the driving term for the QITE linear system.
    driving = np.concatenate(([0.0], np.cross(h, bloch)))
    matrix = overlap + float(delta) * np.eye(4)
    coeffs = np.linalg.lstsq(matrix, driving, rcond=None)[0]

    # exp(-i db * a.sigma) = rotations by 2*db*a around Pauli axes.
    for axis, coefficient in zip("XYZ", coeffs[1:]):
        angle = 2.0 * float(db) * float(coefficient)
        if abs(angle) > 1e-12:
            alist.append((axis, angle, 0))
    return alist
