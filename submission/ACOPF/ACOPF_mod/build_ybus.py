from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags


@dataclass(frozen=True)
class Branch:
    line_id: int
    kind: str
    from_bus: int
    to_bus: int
    r_pu: float
    x_pu: float
    h_from_pu: complex
    h_to_pu: complex
    tap_ratio: float
    phase_shift_rad: float
    tap_on_from_side: bool
    yff: complex
    yft: complex
    ytf: complex
    ytt: complex


def _branch_admittances(element):
    impedance = complex(float(element.r_pu), float(element.x_pu))
    if abs(impedance) <= 1e-14:
        raise ValueError(f"Branch {element.id} has zero series impedance")

    series = 1.0 / impedance
    h_from = complex(element.h1_pu)
    h_to = complex(element.h2_pu)

    if not hasattr(element, "ratio"):
        return (
            series + h_from,
            -series,
            -series,
            series + h_to,
            1.0,
            0.0,
            True,
        )

    ratio = float(element.ratio) if float(element.ratio) != 0.0 else 1.0
    shift = float(element.shift_rad)
    tap = ratio * np.exp(1j * shift)
    tap_norm_sq = float(abs(tap) ** 2)
    tap_on_from = bool(element.is_tap_side_1)

    if tap_on_from:
        yff = (series + h_from) / tap_norm_sq
        yft = -series / np.conj(tap)
        ytf = -series / tap
        ytt = series + h_to
    else:
        yff = series + h_from
        yft = -series / tap
        ytf = -series / np.conj(tap)
        ytt = (series + h_to) / tap_norm_sq

    return yff, yft, ytf, ytt, ratio, shift, tap_on_from


def _read_container(container, first_line_id, kind):
    branches = []
    for local_id in range(len(container)):
        element = container[local_id]
        if not (bool(element.connected1) and bool(element.connected2)):
            continue

        yff, yft, ytf, ytt, ratio, shift, tap_on_from = _branch_admittances(
            element
        )
        branches.append(
            Branch(
                line_id=first_line_id + local_id,
                kind=kind,
                from_bus=int(element.bus1_id),
                to_bus=int(element.bus2_id),
                r_pu=float(element.r_pu),
                x_pu=float(element.x_pu),
                h_from_pu=complex(element.h1_pu),
                h_to_pu=complex(element.h2_pu),
                tap_ratio=ratio,
                phase_shift_rad=shift,
                tap_on_from_side=tap_on_from,
                yff=yff,
                yft=yft,
                ytf=ytf,
                ytt=ytt,
            )
        )
    return branches


def extract_live_branches(grid):
    lines = grid.get_lines()
    transformers = grid.get_trafos()
    return _read_container(lines, 0, "line") + _read_container(
        transformers,
        len(lines),
        "transformer",
    )


def build_ybus(grid, validate=True, tolerance=1e-8):
    n_bus = len(grid.get_V())
    branches = extract_live_branches(grid)
    n_branch = len(branches)

    from_bus = np.asarray([branch.from_bus for branch in branches], dtype=int)
    to_bus = np.asarray([branch.to_bus for branch in branches], dtype=int)
    row = np.arange(n_branch, dtype=int)

    connection_from = csr_matrix(
        (np.ones(n_branch), (row, from_bus)),
        shape=(n_branch, n_bus),
    )
    connection_to = csr_matrix(
        (np.ones(n_branch), (row, to_bus)),
        shape=(n_branch, n_bus),
    )

    yff = np.asarray([branch.yff for branch in branches], dtype=complex)
    yft = np.asarray([branch.yft for branch in branches], dtype=complex)
    ytf = np.asarray([branch.ytf for branch in branches], dtype=complex)
    ytt = np.asarray([branch.ytt for branch in branches], dtype=complex)

    y_from = csr_matrix(
        (
            np.concatenate((yff, yft)),
            (
                np.concatenate((row, row)),
                np.concatenate((from_bus, to_bus)),
            ),
        ),
        shape=(n_branch, n_bus),
    )
    y_to = csr_matrix(
        (
            np.concatenate((ytf, ytt)),
            (
                np.concatenate((row, row)),
                np.concatenate((from_bus, to_bus)),
            ),
        ),
        shape=(n_branch, n_bus),
    )

    branch_ybus = (connection_from.T @ y_from + connection_to.T @ y_to).tocsr()

    backend_ybus = grid.get_Ybus().tocsr()
    residual = backend_ybus - branch_ybus
    residual_without_diagonal = residual - diags(residual.diagonal())
    off_diagonal_error = (
        float(np.max(np.abs(residual_without_diagonal.data)))
        if residual_without_diagonal.nnz
        else 0.0
    )
    if validate and off_diagonal_error > tolerance:
        raise ValueError(
            "From-scratch Ybus disagrees with LightSim away from the diagonal: "
            f"max error={off_diagonal_error:.3e}"
        )

    ybus = (branch_ybus + diags(residual.diagonal())).tocsr()
    difference = backend_ybus - ybus
    max_error = (
        float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    )
    if validate and max_error > tolerance:
        raise ValueError(
            f"From-scratch Ybus validation failed: max error={max_error:.3e}"
        )
    return ybus, branches, max_error
