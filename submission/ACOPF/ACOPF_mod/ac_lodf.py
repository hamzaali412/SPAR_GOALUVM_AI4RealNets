import numpy as np

from ACOPF.ACOPF_mod.live_grid import live_grid
from ACOPF.ACOPF_mod import obs_network
from ACOPF.ACOPF_mod.data_extraction import OBSERVATION_NETWORK


def _line_ac_blocks(env, obs=None):
    grid = obs_network.model_grid(obs)
    if obs is not None and OBSERVATION_NETWORK:
        cls = type(env)
        topo = np.asarray(obs.topo_vect, dtype=int)
        status = np.asarray(obs.line_status, dtype=bool)
        ls_lines, ls_trafos = grid.get_lines(), grid.get_trafos()
        blocks = []
        for line_id in range(len(ls_lines) + len(ls_trafos)):
            element = (
                ls_lines[line_id]
                if line_id < len(ls_lines)
                else ls_trafos[line_id - len(ls_lines)]
            )
            bus1 = obs_network.global_bus(
                cls,
                int(cls.line_or_to_subid[line_id]),
                int(topo[int(cls.line_or_pos_topo_vect[line_id])]),
            )
            bus2 = obs_network.global_bus(
                cls,
                int(cls.line_ex_to_subid[line_id]),
                int(topo[int(cls.line_ex_pos_topo_vect[line_id])]),
            )
            blocks.append({
                "bus1": bus1, "bus2": bus2,
                "y11": complex(element.yac_11), "y12": complex(element.yac_12),
                "y21": complex(element.yac_21), "y22": complex(element.yac_22),
                "connected": bool(status[line_id]) and bus1 >= 0 and bus2 >= 0,
            })
        return blocks
    ls_lines, ls_trafos = grid.get_lines(), grid.get_trafos()
    blocks = []
    for i in range(len(ls_lines)):
        ln = ls_lines[i]
        blocks.append({
            "bus1": int(ln.bus1_id), "bus2": int(ln.bus2_id),
            "y11": complex(ln.yac_11), "y12": complex(ln.yac_12),
            "y21": complex(ln.yac_21), "y22": complex(ln.yac_22),
            "connected": bool(ln.connected1) and bool(ln.connected2),
        })
    for i in range(len(ls_trafos)):
        tr = ls_trafos[i]
        blocks.append({
            "bus1": int(tr.bus1_id), "bus2": int(tr.bus2_id),
            "y11": complex(tr.yac_11), "y12": complex(tr.yac_12),
            "y21": complex(tr.yac_21), "y22": complex(tr.yac_22),
            "connected": bool(tr.connected1) and bool(tr.connected2),
        })
    return blocks


def ac_lodf(env, obs, outage_lines, monitor_lines=None):
    if OBSERVATION_NETWORK:
        return _ac_lodf(env, obs, outage_lines, monitor_lines)
    with live_grid(env, obs):
        return _ac_lodf(env, obs, outage_lines, monitor_lines)


def _ac_lodf(env, obs, outage_lines, monitor_lines=None):
    if isinstance(outage_lines, (int, np.integer)):
        outage_lines = [int(outage_lines)]
    outage_lines = [int(l) for l in outage_lines]

    grid = obs_network.model_grid(obs)
    try:
        base_mva = float(grid.get_sn_mva())
    except Exception:
        base_mva = 100.0
    if OBSERVATION_NETWORK:
        network = obs_network.observation_network(env, obs, grid)
        Ybus = network["ybus"]
        V = network["voltage"]
        active = np.where(network["active"])[0]
    else:
        Ybus = grid.get_Ybus()
        V = np.asarray(grid.get_V())
        active = np.where(np.asarray(grid.get_bus_status(), dtype=bool))[0]
    pos = {int(b): k for k, b in enumerate(active)}

    Y_active = np.asarray(Ybus[np.ix_(active, active)].todense())
    Zbus = np.linalg.pinv(Y_active)

    blocks = _line_ac_blocks(env, obs)
    for l in outage_lines:
        if not blocks[l]["connected"]:
            raise ValueError(f"line {l} is already disconnected -- nothing to remove")

    K = len(outage_lines)
    term_bus = []
    I_pre = np.zeros(2 * K, dtype=complex)
    for k, l in enumerate(outage_lines):
        b = blocks[l]
        term_bus += [pos[b["bus1"]], pos[b["bus2"]]]
        I_pre[2 * k] = b["y11"] * V[b["bus1"]] + b["y12"] * V[b["bus2"]]
        I_pre[2 * k + 1] = b["y21"] * V[b["bus1"]] + b["y22"] * V[b["bus2"]]

    S = np.zeros((2 * K, 2 * K), dtype=complex)
    for k, l in enumerate(outage_lines):
        b = blocks[l]
        i_red, j_red = pos[b["bus1"]], pos[b["bus2"]]
        for m, node in enumerate(term_bus):
            dV_i, dV_j = Zbus[i_red, node], Zbus[j_red, node]
            S[2 * k, m] = b["y11"] * dV_i + b["y12"] * dV_j
            S[2 * k + 1, m] = b["y21"] * dV_i + b["y22"] * dV_j

    T = np.eye(2 * K, dtype=complex) - S
    gamma = np.linalg.solve(T, I_pre)

    dV = np.zeros(len(V), dtype=complex)
    dV[active] = Zbus[:, term_bus] @ gamma

    if monitor_lines is None:
        monitor_lines = [l for l, b in enumerate(blocks)
                         if b["connected"] and l not in outage_lines]

    out = {}
    for m in monitor_lines:
        b = blocks[m]
        i_glob, j_glob = b["bus1"], b["bus2"]
        I_pre_m = b["y11"] * V[i_glob] + b["y12"] * V[j_glob]
        I_post_m = I_pre_m + (b["y11"] * dV[i_glob] + b["y12"] * dV[j_glob])
        s_pre = float(abs(V[i_glob]) * abs(I_pre_m))
        s_post = float(abs(V[i_glob] + dV[i_glob]) * abs(I_post_m))
        ratio = s_post / s_pre if s_pre > 1e-9 else 1.0
        out[m] = {
            "s_pre_mw": s_pre,
            "s_post_mw": s_post,
            "delta_mw": s_post - s_pre,
            "rho_pre": float(obs.rho[m]),
            "rho_post": float(obs.rho[m]) * ratio,
            "p_pre_mw": float((V[i_glob] * np.conj(I_pre_m)).real * base_mva),
            "p_post_mw": float(((V[i_glob] + dV[i_glob]) * np.conj(I_post_m)).real * base_mva),
        }

    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1]["delta_mw"])))
