"""Fly brain in DOOM: a FlyWire connectome subcircuit as an integer spiking network (doomformer.spiking.IntNet).

Pieces (all reimplemented from the published equations, no code copied from the GPL-2.0 reference repo):

  * `Brain`  -- float full-brain reference: the Shiu et al. 2024 (Nature) leaky integrate-and-fire model on the
                FlyWire v783 connectome (138,639 neurons, 15.1M neuron pairs), dt 0.1 ms, exact exponential integration.
  * `select_subcircuit` -- choose <= 200 neurons that carry sugar -> MN9 (feeding) and bitter -> suppression.
  * `integerize` -- scale the subcircuit into an `IntNet` (the exact integer neuron DOOM executes).
  * `bernoulli_inputs`, `run_int` -- seeded input trains and the IntNet run used for validation.

Data: FlyWire connectome (Dorkenwald et al. 2024; Schlegel et al. 2024), v783 materialization, as packaged by
Shiu et al. 2024 / Eon Systems. See docs/fly_circuit.md for licence and attribution.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from doomformer.spiking import IntNet, V_MAX

DATA = Path(__file__).resolve().parent.parent / "build" / "fly"

# ---- Shiu et al. 2024 LIF parameters (model.py `default_params`) -----------------------------------------------
V0, V_RST, V_TH = -52.0, -52.0, -45.0     # mV
T_MBR, TAU = 20.0, 5.0                    # ms: membrane and synaptic time constants
T_RFC, T_DLY = 2.2, 1.8                   # ms: refractory period, synaptic delay
W_SYN = 0.275                             # mV per synapse (added to the synaptic variable g)
F_POI = 250                               # Poisson-input weight multiplier (v jumps by W_SYN * F_POI = 68.75 mV)
DT = 0.1                                  # ms, simulation step

# ---- FlyWire root ids (v783), from the Shiu et al. example notebook -------------------------------------------
MN9_LEFT, MN9_RIGHT = 720575940660219265, 720575940618238523     # proboscis motor neurons (feeding)
GIANT_FIBER = [720575940622838154, 720575940632499757]            # escape
P9 = [720575940627652358, 720575940635872101]
SUGAR_GRNS = [
    720575940616885538, 720575940630233916, 720575940639332736, 720575940632889389, 720575940617000768,
    720575940632425919, 720575940637568838, 720575940629176663, 720575940621502051, 720575940638202345,
    720575940612670570, 720575940611875570, 720575940621754367, 720575940633143833, 720575940613601698,
    720575940630797113, 720575940639198653, 720575940639259967, 720575940624963786, 720575940640649691,
    720575940610788069, 720575940623172843, 720575940628853239
]

BITTER_GRNS = [
    720575940619072513, 720575940646212996, 720575940622298631, 720575940642088333, 720575940627692048,
    720575940617239197, 720575940618682526, 720575940604714528, 720575940603266592, 720575940604027168,
    720575940619197093, 720575940610259370, 720575940627578156, 720575940629481516, 720575940618887217,
    720575940614281266, 720575940634859188, 720575940645743412, 720575940637742911, 720575940617094208,
    720575940629416318, 720575940630195909, 720575940615641798, 720575940638312262, 720575940624310345,
    720575940621778381, 720575940619659861, 720575940629146711, 720575940625750105, 720575940610483162,
    720575940610481370, 720575940602353632, 720575940610773090, 720575940617433830, 720575940628962407,
    720575940626287336, 720575940623183083, 720575940618025199, 720575940619028208, 720575940621864060,
    720575940613061118, 720575940621008895
]

LC_4S = [
    720575940605598892, 720575940611134833, 720575940612580977, 720575940613256863, 720575940613260959,
    720575940614914107, 720575940615462587, 720575940617176321, 720575940617266722, 720575940618807105,
    720575940620795728, 720575940622108001, 720575940624017251, 720575940625038090, 720575940625934973,
    720575940625991043, 720575940626605200, 720575940626626895, 720575940628454522, 720575940628462340,
    720575940630851036, 720575940638496720, 720575940603637438, 720575940610522009, 720575940612093351,
    720575940612323025, 720575940612380723, 720575940612498129, 720575940612518055, 720575940612968421,
    720575940613609484, 720575940613638041, 720575940614572742, 720575940614582946, 720575940615053580,
    720575940615127227, 720575940615232217, 720575940615575007, 720575940616066705, 720575940616713355,
    720575940617026260, 720575940617348379, 720575940618002644, 720575940618234704, 720575940618234715,
    720575940618266459, 720575940618267227, 720575940618275520, 720575940618312606, 720575940618676440,
    720575940618709158, 720575940618723749, 720575940619397542, 720575940620314221, 720575940620314612,
    720575940620731380, 720575940620903551, 720575940621145821, 720575940621522458, 720575940621753579,
    720575940622330582, 720575940622531767, 720575940622939836, 720575940624111763, 720575940624790781,
    720575940624856762, 720575940625841351, 720575940625845447, 720575940625906702, 720575940625932421,
    720575940626553596, 720575940626916936, 720575940627519107, 720575940628064260, 720575940628081541,
    720575940628419527, 720575940628518400, 720575940628599895, 720575940628606713, 720575940628699560,
    720575940628891863, 720575940629753807, 720575940629964591, 720575940630154660, 720575940630484495,
    720575940630998339, 720575940631032657, 720575940631338271, 720575940632475449, 720575940632715234,
    720575940632769180, 720575940633013355, 720575940633218863, 720575940633580384, 720575940634517856,
    720575940635835967, 720575940636957006, 720575940638456227, 720575940639817947, 720575940640612480,
    720575940641213824, 720575940645821316, 720575940649229433, 720575940652611745
]


class LIF:
    """Shiu LIF on a sparse signed synapse-count matrix (float reference). Stored CSC over presynaptic columns."""

    def __init__(self, counts: sp.spmatrix):
        """counts: [post, pre] signed integer synapse counts."""
        c = sp.csc_matrix(counts)
        c.sort_indices()
        self.n = c.shape[0]
        self.indptr = c.indptr.astype(np.int64)
        self.post = c.indices.astype(np.int32)
        self.count = c.data.astype(np.int32)
        self.w_mv = self.count.astype(np.float64) * W_SYN

    def counts_matrix(self) -> sp.csc_matrix:
        """Signed synapse counts as CSC, [post, pre]."""
        return sp.csc_matrix((self.count, self.post, self.indptr), shape=(self.n, self.n))

    def simulate(self, inputs: list[tuple[np.ndarray, float]], t_ms: float, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Run the full brain. inputs: [(neuron indices, Poisson rate in Hz), ...] (each neuron gets an independent
        Poisson train of rate r; an event adds W_SYN*F_POI mV to v, which always fires it; input neurons have no
        refractory period). Returns (neuron index, step) of every spike (step = time / DT)."""
        rng = np.random.default_rng(seed)
        n, steps = self.n, int(round(t_ms / DT))
        a, b = np.exp(-DT / TAU), np.exp(-DT / T_MBR)
        kg = TAU / (TAU - T_MBR) * (a - b)              # u(t+dt) = u b + g * kg   (exact for the g -> v cascade)
        D = int(round(T_DLY / DT))                      # delay in steps
        R = D + 1
        n_rfc = int(round(T_RFC / DT))
        buf = np.zeros((R, n))
        v = np.full(n, V0)
        g = np.zeros(n)
        in_idx = np.concatenate([i for i, _ in inputs]) if inputs else np.zeros(0, np.int64)
        in_p = np.concatenate([np.full(len(i), r * DT * 1e-3) for i, r in inputs]) if inputs else np.zeros(0)
        is_in = np.zeros(n, bool)
        is_in[in_idx] = True
        refr: deque = deque()                           # (step, idx) of recent spikes still refractory
        out_i, out_t = [], []
        for t in range(steps):
            slot = buf[t % R]
            g += slot
            slot[:] = 0.0
            while refr and refr[0][0] <= t - 1 - n_rfc:
                refr.popleft()
            ridx = np.concatenate([x for _, x in refr]) if refr else None
            if ridx is not None:
                v_keep, g_keep = v[ridx], g[ridx]
            u = v - V0
            u *= b
            u += g * kg
            u += V0
            g *= a
            v = u
            if ridx is not None:
                v[ridx], g[ridx] = v_keep, g_keep
            if len(in_idx):
                hit = in_idx[rng.random(len(in_idx)) < in_p]
                v[hit] += W_SYN * F_POI
            spk = np.flatnonzero(v > V_TH)
            if len(spk):
                v[spk] = V_RST
                g[spk] = 0.0
                out_i.append(spk.astype(np.int32))
                out_t.append(np.full(len(spk), t, np.int32))
                r_spk = spk[~is_in[spk]]
                if len(r_spk):
                    refr.append((t, r_spk))
                st, en = self.indptr[spk], self.indptr[spk + 1]
                ln = en - st
                tot = int(ln.sum())
                if tot:
                    pos = np.repeat(st - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(tot)
                    buf[(t + D) % R] += np.bincount(self.post[pos], weights=self.w_mv[pos], minlength=n)
        if not out_i:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        return np.concatenate(out_i), np.concatenate(out_t)


class Brain(LIF):
    """Full FlyWire brain: 138,639 neurons, 15.1M neuron pairs (v783)."""

    def __init__(self, data_dir: Path = DATA):
        data_dir = Path(data_dir)
        self.ids = pd.read_csv(data_dir / "2025_Completeness_783.csv", index_col=0).index.values.astype(np.int64)
        self.id2idx = {int(i): k for k, i in enumerate(self.ids)}
        df = pd.read_parquet(data_dir / "2025_Connectivity_783.parquet",
                             columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"])
        cnt = (df["Connectivity"].values * df["Excitatory"].values).astype(np.int32)
        super().__init__(sp.csc_matrix((cnt, (df["Postsynaptic_Index"].values, df["Presynaptic_Index"].values)),
                                       shape=(len(self.ids),) * 2))

    def idx(self, flyids) -> np.ndarray:
        return np.array([self.id2idx[int(i)] for i in flyids], np.int64)


# antennal grooming: aDN1 descending neurons and the Johnston's organ (JO) sensory neurons of the notebook
ADN1_RIGHT, ADN1_LEFT = 720575940616185531, 720575940624319124
ADN1 = [ADN1_LEFT, ADN1_RIGHT]
JO_EV = [
    720575940645106376, 720575940615272415, 720575940619869120, 720575940620257345, 720575940620382889,
    720575940630834683, 720575940632449619, 720575940634020508, 720575940605530302, 720575940607140035,
    720575940608742409, 720575940615590843, 720575940620410177, 720575940621870618, 720575940622344170,
    720575940623298559, 720575940626042149, 720575940627379333, 720575940630080071, 720575940632128031,
    720575940632307527, 720575940634820703
]

JO_EDC = [
    720575940606154370, 720575940605919334, 720575940608884931, 720575940616655989, 720575940620543110,
    720575940622937528, 720575940624799290, 720575940626565455, 720575940627941431, 720575940627977457,
    720575940628160617, 720575940629188251, 720575940641921421
]

JO_EDM = [
    720575940615972027, 720575940618941037, 720575940619729835, 720575940627282279, 720575940628903247,
    720575940604122982, 720575940609486690, 720575940609541917, 720575940610018266, 720575940611061526,
    720575940611273395, 720575940611684787, 720575940614060829, 720575940616040587, 720575940618599872,
    720575940618684481, 720575940619663239, 720575940619932654, 720575940620919578, 720575940621218729,
    720575940622271684, 720575940622638276, 720575940623312828, 720575940625797617, 720575940625962568,
    720575940626309438, 720575940626666066, 720575940627109991, 720575940628101126, 720575940628978450,
    720575940629055721, 720575940629650997, 720575940629985900, 720575940630992557, 720575940637054835,
    720575940637084762, 720575940638664437, 720575940646927668, 720575940646929204, 720575940659131009
]

JO_EDP = [
    720575940609522461, 720575940610261346, 720575940613641915, 720575940615469785, 720575940616589878,
    720575940616951124, 720575940619479979, 720575940621218985, 720575940628444667, 720575940634634606,
    720575940640753267, 720575940650244342
]

JO_EVL = [
    720575940615573597, 720575940615848788, 720575940619083349, 720575940621397417, 720575940621625597,
    720575940622283912, 720575940627049731, 720575940629022149, 720575940630122015, 720575940630564179,
    720575940633153375, 720575940637410869, 720575940638681845, 720575940621033477, 720575940621776410,
    720575940621815690, 720575940622234211, 720575940622635817, 720575940623897096, 720575940626148354,
    720575940626540821, 720575940628258715, 720575940629743063, 720575940630202624, 720575940630544967,
    720575940633553820, 720575940644036644
]

JO_EVM = [
    720575940602132509, 720575940602506208, 720575940610759634, 720575940614188149, 720575940615809349,
    720575940615976891, 720575940619341105, 720575940621092534, 720575940622419165, 720575940622449388,
    720575940623108134, 720575940624981436, 720575940628192055, 720575940630059847, 720575940632767383,
    720575940639296189, 720575940645466500, 720575940611783464, 720575940612307478, 720575940612960552,
    720575940614351477, 720575940617212134, 720575940617434086, 720575940618130334, 720575940620249734,
    720575940620940276, 720575940621010352, 720575940621729757, 720575940623437547, 720575940624546062,
    720575940624686268, 720575940625054647, 720575940625605905, 720575940626795909, 720575940627585688,
    720575940630020111, 720575940632175268, 720575940634073183, 720575940634891700, 720575940637012196,
    720575940637243504, 720575940639339392, 720575940659426177
]

JO_EVP = [
    720575940620444654, 720575940631866508, 720575940607853833, 720575940611088563, 720575940612773374,
    720575940613221928, 720575940615024543, 720575940615986459, 720575940617811013, 720575940618467195,
    720575940621442224, 720575940622199977, 720575940624915230, 720575940625559358, 720575940627104649,
    720575940627314088, 720575940633058989, 720575940636335735
]

JO_CA = [
    720575940605800369, 720575940608784579, 720575940618135109, 720575940626719101, 720575940629296185,
    720575940636137591, 720575940602720940, 720575940610079857, 720575940614427195, 720575940616501787,
    720575940617156445, 720575940625909962, 720575940626241369, 720575940629105658, 720575940629138959,
    720575940636559534, 720575940641372661
]

JO_CL = [
    720575940626135548, 720575940627751567, 720575940604753437, 720575940613971485, 720575940614835362,
    720575940623399059, 720575940630319671, 720575940639082062
]

JO_CM = [
    720575940607386307, 720575940634512992, 720575940614035485, 720575940618901424, 720575940630070343,
    720575940633443353, 720575940635058612, 720575940637632419, 720575940625626000
]

JO_GROUPS = {"JO_EV": JO_EV, "JO_EDC": JO_EDC, "JO_EDM": JO_EDM, "JO_EDP": JO_EDP, "JO_EVL": JO_EVL, "JO_EVM": JO_EVM, "JO_EVP": JO_EVP, "JO_CA": JO_CA, "JO_CL": JO_CL, "JO_CM": JO_CM}
ALL_JOS = [i for g in JO_GROUPS.values() for i in g]


# ---- experiments: the sensory conditions of Shiu et al. (example notebook) ----------------------------------------
RATE_HZ = 200.0
# condition -> {input group: Poisson rate in Hz}; GRN groups are the sugar / bitter gustatory receptor neurons
CONDITIONS = {
    "none": {},
    "sugar": {"sugar": RATE_HZ},
    "sugar+bitter": {"sugar": RATE_HZ, "bitter": RATE_HZ},
    "bitter": {"bitter": RATE_HZ},
}


def run_float(brain: LIF, groups: dict[str, np.ndarray], cond: dict[str, float], t_ms: float, seed: int):
    """One full-brain float trial; returns (neuron, step) spike arrays."""
    return brain.simulate([(groups[k], r) for k, r in cond.items()], t_ms, seed)


def spike_counts(spk, n: int, t0_ms: float = 0.0, t1_ms: float = np.inf) -> np.ndarray:
    """Per-neuron spike counts of a float trial inside [t0, t1) ms."""
    i, t = spk
    keep = (t >= round(t0_ms / DT)) & (t < round(min(t1_ms, 1e12) / DT))
    return np.bincount(i[keep], minlength=n)


# ---- subcircuit selection -------------------------------------------------------------------------------------
def influence(C: sp.csr_matrix, rates: dict[str, np.ndarray], grn_drive: dict[str, np.ndarray],
              targets: np.ndarray, depth: int = 5) -> np.ndarray:
    """Activity-weighted backward influence on `targets` (local indices) over the candidate graph.

    C[post, pre] signed synapse counts among candidates. In every condition the share of postsynaptic neuron j's
    activity-weighted synaptic input that comes from i is d_ji = |C_ji| r_i / (sum_k |C_jk| r_k + GRN drive_j);
    influence(MN9) = 1, influence_{l+1} = d^T influence_l, accumulated over `depth` hops (sum over paths of
    products of shares). The score of a neuron is its maximum over conditions, so neurons that matter only for the
    bitter (suppression) path are kept as well as the sugar path."""
    score = np.zeros(C.shape[0])
    absC = abs(C)
    for k, r in rates.items():
        Wd = absC.multiply(r[None, :]).tocsr()
        den = np.asarray(Wd.sum(1)).ravel() + grn_drive[k]
        D = sp.diags(1.0 / np.maximum(den, 1e-9)) @ Wd
        s = np.zeros(C.shape[0])
        s[targets] = 1.0
        acc = s.copy()
        for _ in range(depth):
            s = D.T @ s
            acc += s
        score = np.maximum(score, acc)
    return score


def select_subcircuit(brain: Brain, rates: dict[str, np.ndarray], groups: dict[str, np.ndarray],
                      n_max: int = 200, nnz_max: int = 4000, min_rate: float = 0.5, depth: int = 5) -> dict:
    """Pick <= n_max simulated neurons (sensory GRNs become input lines, not neurons) and prune weak synapses.

    rates[cond]: mean full-brain firing rate (Hz) of every neuron in that condition (float reference).
    Criterion: candidates = neurons (not the GRNs) with rate >= min_rate Hz in at least one condition (sugar,
    sugar+bitter, bitter); rank them by `influence` on MN9 (left+right); keep the n_max - 0 best (MN9 always in).
    Pruning: drop synapses with |count| < c_min, c_min the smallest integer for which nnz(W)+nnz(U) <= nnz_max."""
    mn = brain.idx([MN9_LEFT, MN9_RIGHT])
    grn_idx = np.concatenate([groups["sugar"], groups["bitter"]])
    C = brain.counts_matrix().tocsr()
    cand_mask = np.zeros(brain.n, bool)
    for r in rates.values():
        cand_mask |= r >= min_rate
    cand_mask[grn_idx] = False
    cand_mask[mn] = True
    cand = np.flatnonzero(cand_mask)
    Cc = C[cand][:, cand].astype(float).tocsr()
    drive = {}
    for k, cond in CONDITIONS.items():
        if k == "none":
            continue
        gv = np.zeros(len(grn_idx))
        for grp, hz in cond.items():
            gv[np.isin(grn_idx, groups[grp])] = hz
        drive[k] = np.asarray(abs(C[cand][:, grn_idx]).multiply(gv[None, :]).sum(1)).ravel()
    score = influence(Cc, {k: rates[k][cand] for k in drive}, drive, np.searchsorted(cand, mn), depth)
    order = np.argsort(-score, kind="stable")
    chosen = set(order[:n_max].tolist())
    chosen |= set(np.searchsorted(cand, mn).tolist())
    chosen = sorted(chosen, key=lambda i: -score[i])[:n_max]
    sel = np.sort(cand[chosen])
    Wfull = sp.coo_matrix(C[sel][:, sel])
    Ufull = sp.coo_matrix(C[sel][:, grn_idx])
    mag = np.concatenate([abs(Wfull.data), abs(Ufull.data)])
    c_min = 1
    while (mag >= c_min).sum() > nnz_max:
        c_min += 1
    kw = abs(Wfull.data) >= c_min
    ku = abs(Ufull.data) >= c_min
    W = sp.coo_matrix((Wfull.data[kw], (Wfull.row[kw], Wfull.col[kw])), shape=Wfull.shape).toarray()
    U = sp.coo_matrix((Ufull.data[ku], (Ufull.row[ku], Ufull.col[ku])), shape=Ufull.shape).toarray()
    used = np.flatnonzero((U != 0).any(0))
    info = dict(n_candidates=int(len(cand)), c_min=int(c_min),
                edges_total=int(len(mag)), edges_kept=int(kw.sum() + ku.sum()),
                syn_mass_total=int(mag.sum()), syn_mass_kept=int(abs(Wfull.data[kw]).sum() + abs(Ufull.data[ku]).sum()),
                score_min_kept=float(min(score[chosen])), min_rate=min_rate, depth=depth)
    return dict(sel=sel, grn=grn_idx[used], grn_group=np.array(["sugar"] * len(groups["sugar"])
                + ["bitter"] * len(groups["bitter"]))[used], W=W, U=U[:, used], info=info,
                score=dict(zip(cand.tolist(), score.tolist())))


def subnet_lif(circ: dict) -> tuple[LIF, np.ndarray, dict[str, np.ndarray]]:
    """Float LIF of the pruned subcircuit alone: neurons = selected neurons + the used GRNs (as Poisson-driven
    neurons). Returns (net, local index of selected neurons, {group: local GRN indices})."""
    n, m = len(circ["sel"]), len(circ["grn"])
    M = np.zeros((n + m, n + m), np.int32)
    M[:n, :n] = circ["W"]
    M[:n, n:] = circ["U"]
    loc = {g: n + np.flatnonzero(circ["grn_group"] == g) for g in ("sugar", "bitter")}
    return LIF(sp.csc_matrix(M)), np.arange(n), loc


# ---- integerization --------------------------------------------------------------------------------------------
def int_scale(theta: int) -> float:
    """Integer weight units per synapse count at unit gain.

    A spike through `c` synapses adds c * W_SYN mV to g; g decays with TAU and drives v with 1/T_MBR, so its time
    integral moves v by c * W_SYN * TAU / T_MBR mV (0.06875 mV per synapse). The integer neuron has threshold `theta`
    for the (V_TH - V_RST) = 7 mV gap, i.e. theta / 7 units per mV."""
    return theta / (V_TH - V_RST) * W_SYN * TAU / T_MBR


def input_rates(circ: dict, cond: str, rates_hz: np.ndarray) -> np.ndarray:
    """Presynaptic firing rates (Hz) seen by the subcircuit in a condition: [selected neurons, used GRN lines]."""
    x = np.array([CONDITIONS[cond].get(g, 0.0) for g in circ["grn_group"]])
    return np.concatenate([rates_hz, x])


def fit_gains(circ: dict, rates: dict[str, np.ndarray], dt_ms: float, theta: int, leak: int,
              f_min: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Open-loop per-postsynaptic-neuron gains (rate-matched synaptic scaling): (s_exc, s_inh, damp), arrays (N,) and damp = 1.

    The integer neuron is a linear-threshold unit: with mean input u units/step and linear leak L it fires at
    (u - L) / theta spikes/step. For each neuron j and condition k in which the float full-brain model fires it at
    f_jk >= f_min Hz, the unit-gain drive is d_jk = int_scale * dt * (sum_i c_ji r_ik + sum_m u_jm x_mk) (counts
    times presynaptic float rates, signed), and the required drive is t_jk = theta * f_jk * dt + L. s_exc[j] is the
    least-squares gain through the origin, sum t d / sum d^2 over the conditions with d_jk > 0. This keeps the
    synapse counts' relative sizes inside each postsynaptic neuron and replaces the float neuron's saturation
    (which the integer neuron lacks: with unit gain, hubs receive many times theta per step and overflow) by one
    scalar per neuron. Neurons with no active condition get the median gain. s_inh: see `inh_gains`."""
    sc = int_scale(theta) * dt_ms * 1e-3
    A = np.hstack([circ["W"], circ["U"]]).astype(float)
    num = np.zeros(A.shape[0])
    den = np.zeros(A.shape[0])
    for cond in ("sugar", "sugar+bitter", "bitter"):
        d = sc * A @ input_rates(circ, cond, rates[cond])
        f = np.minimum(rates[cond] * dt_ms * 1e-3, 0.95)
        t = theta * f + leak
        ok = (rates[cond] >= f_min) & (d > 0)
        num += np.where(ok, t * d, 0.0)
        den += np.where(ok, d * d, 0.0)
    s_exc = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    s_exc[np.isnan(s_exc)] = np.nanmedian(s_exc)
    return s_exc, inh_gains(circ, rates, s_exc, dt_ms, theta, leak), 1.0


def inh_gains(circ: dict, rates: dict[str, np.ndarray], s_exc: np.ndarray, dt_ms: float, theta: int,
              leak: int, T: int = 120, v_frac: float = 0.6) -> np.ndarray:
    """Inhibitory gain s_inh <= s_exc. Inhibition has no floor in the float model either, but the linear leak is
    only L per step, so a strongly inhibited neuron drifts towards -infinity (|v| bound!): s_inh is lowered until
    the mean drift s_exc * E - s_inh * I + L >= -v_frac * V_MAX / T in every condition (E, I: unit-gain
    excitatory / inhibitory drive at the float rates), so a T-step trial from rest stays above -v_frac * V_MAX."""
    sc = int_scale(theta) * dt_ms * 1e-3
    A = np.hstack([circ["W"], circ["U"]]).astype(float)
    s_inh = s_exc.copy()
    for cond in ("sugar", "sugar+bitter", "bitter"):
        r = input_rates(circ, cond, rates[cond])
        E = sc * np.maximum(A, 0) @ r
        I = sc * np.maximum(-A, 0) @ r
        lim = (s_exc * E + leak + v_frac * V_MAX / T) / np.maximum(I, 1e-12)
        s_inh = np.minimum(s_inh, np.where(I > 0, lim, np.inf))
    return s_inh


def _counts_clamped(net: IntNet, cond: str, T: int, seeds) -> tuple[np.ndarray, np.ndarray]:
    """Mean spike counts over seeds and max positive v of every neuron of the integer net, with v clamped to
    +-V_MAX instead of raising (calibration only)."""
    out = np.zeros(net.n)
    vmax = np.zeros(net.n)
    for sd in seeds:
        X = bernoulli_inputs(net, cond, T, sd)
        v = np.zeros(net.n, np.int64)
        for x in X:
            sp_ = (v >= net.theta).astype(np.int64)
            v = np.clip(v - net.theta * sp_ - net.leak * (v >= net.leak) + net.leak * (v <= -net.leak) + net.W @ sp_ + net.U @ x,
                        -V_MAX, V_MAX)
            out += sp_
            vmax = np.maximum(vmax, v)
    return out / len(seeds), vmax


def loop_gain(net: IntNet, rates: dict[str, np.ndarray], f_min: float = 1.0) -> float:
    """Largest spectral radius of the rate-map Jacobian W_act / theta (neurons that fire >= f_min Hz in the float
    model of a condition; linear-threshold units: d rate_j / d rate_i = W_ji / theta), over the three conditions.
    > 1 means the integer net's recurrent excitation runs away (the float neuron's refractory saturation, which
    cuts its small-signal gain to (1 - f * t_rfc)^2, is absent)."""
    rho = 0.0
    for cond in ("sugar", "sugar+bitter", "bitter"):
        act = rates[cond] >= f_min
        rho = max(rho, float(np.abs(np.linalg.eigvals(net.W[np.ix_(act, act)] / net.theta)).max()))
    return rho


def calibrate_gains(brain: Brain, circ: dict, rates: dict[str, np.ndarray], ref: dict[str, np.ndarray],
                    dt_ms: float, theta: int, lam: float, cap: int, T: int, seeds, iters: int = 30,
                    eta: float = 0.3, rho_max: float = 0.9, v_frac: float = 0.35) -> tuple[tuple[np.ndarray, np.ndarray, float], list]:
    """Closed-loop refinement on calibration seeds: the integer net is run on Bernoulli inputs and every neuron's
    excitatory gain is nudged by eta/3 times the summed log ratio of the float and integer spike counts (+eps
    smoothing) in the sugar, sugar+bitter and bitter conditions, minus a penalty for any neuron whose positive v
    exceeds v_frac * V_MAX (runaway); s_inh follows via `inh_gains`; the recurrent
    weights are damped by one global factor so that `loop_gain` <= rho_max (stability of the rate map).
    ref[cond]: mean float spike count of the selected neurons in the T*dt window.
    Returns (gains, history of (mean |log ratio|, damp))."""
    leak = int(round(lam * theta * dt_ms / T_MBR))
    s_exc, s_inh, _ = fit_gains(circ, rates, dt_ms, theta, leak)
    eps = T * dt_ms / 120.0                      # count smoothing proportional to the window (1 spike per 120 ms)
    hist = []
    for _ in range(iters):
        rho = loop_gain(integerize(brain, circ, rates, dt_ms, theta, lam, cap, gains=(s_exc, s_inh, 1.0)), rates)
        damp = min(1.0, rho_max / rho)
        net = integerize(brain, circ, rates, dt_ms, theta, lam, cap, gains=(s_exc, s_inh, damp))
        lr = np.zeros(net.n)
        vpos = np.zeros(net.n)
        for cond in ("sugar", "sugar+bitter", "bitter"):
            cnt, vm = _counts_clamped(net, cond, T, seeds)
            lr += np.log((ref[cond] + eps) / (cnt + eps))
            vpos = np.maximum(vpos, vm)
        hist.append((float(np.abs(lr).mean() / 3), damp))
        lr -= 3.0 * np.clip(vpos / (v_frac * V_MAX) - 1.0, 0.0, 2.0)      # runaway penalty
        s_exc = s_exc * np.exp(np.clip(eta * lr / 3, -0.7, 0.7))
        s_inh = inh_gains(circ, rates, s_exc, dt_ms, theta, leak)
    rho = loop_gain(integerize(brain, circ, rates, dt_ms, theta, lam, cap, gains=(s_exc, s_inh, 1.0)), rates)
    return (s_exc, s_inh, min(1.0, rho_max / rho)), hist


def log_error(net: IntNet, ref: dict[str, np.ndarray], T: int, seeds) -> tuple[float, int]:
    """Mean |log((int count + eps) / (float count + eps))|, eps = T * dt / 120 ms, over neurons and the three stimulated conditions, and the
    max |v| seen (raises OverflowError through IntNet.run if the contract is violated)."""
    err, vmax = [], 0
    for cond in ("sugar", "sugar+bitter", "bitter"):
        cnt = np.zeros(net.n)
        for sd in seeds:
            V, S = run_int(net, cond, T, sd)
            cnt += S.sum(0)
            vmax = max(vmax, int(np.abs(V).max()))
        eps = T * net.meta["dt_ms"] / 120.0
        err.append(np.abs(np.log((ref[cond] + eps) / (cnt / len(seeds) + eps))).mean())
    return float(np.mean(err)), vmax


def search_integerization(brain: Brain, circ: dict, rates: dict[str, np.ndarray], ref_fn, T: int = 120,
                          dts=(2.0,), thetas=(200,), lams=(0.25, 0.5), rho_maxs=(0.9, 1.2), cap: int = 200,
                          cal_seeds=range(200, 206), sel_seeds=range(300, 308), v_ok: float = 0.8,
                          robust_seeds=range(400, 440)):
    """Grid over (dt, theta, lam, rho_max): calibrate gains (calibration seeds) and score on separate selection
    seeds. A candidate is valid if no overflow and max |v| <= v_ok * V_MAX on the selection seeds, sugar MN9 fires
    and sugar+bitter MN9 fires < 10% of sugar. Best = lowest `log_error` among valid (ties: larger damp).
    ref_fn(window_ms) -> {cond: mean float counts of the selected neurons in [0, window_ms)}.
    Returns (best IntNet, table rows)."""
    rows, cands = [], []
    for dt in dts:
        ref = ref_fn(T * dt)
        for theta in thetas:
            for lam in lams:
                for rho_max in rho_maxs:
                    gains, hist = calibrate_gains(brain, circ, rates, ref, dt, theta, lam, cap, T, cal_seeds,
                                                  rho_max=rho_max)
                    net = integerize(brain, circ, rates, dt, theta, lam, cap, gains=gains)
                    row = dict(dt=dt, theta=theta, lam=lam, rho_max=rho_max, damp=float(gains[2]), valid=False,
                               err=float("nan"), vmax=-1, mn9_sugar=float("nan"), mn9_sb=float("nan"))
                    try:
                        err, vmax = log_error(net, ref, T, sel_seeds)
                        mn = [net.meta["outputs"]["MN9_left"], net.meta["outputs"]["MN9_right"]]
                        cn = {c: np.mean([run_int(net, c, T, sd)[1][:, mn].sum() for sd in sel_seeds])
                              for c in ("sugar", "sugar+bitter")}
                        row.update(err=err, vmax=vmax, mn9_sugar=float(cn["sugar"]), mn9_sb=float(cn["sugar+bitter"]))
                        fm = float(ref["sugar"][mn].sum())
                        row["valid"] = bool(vmax <= v_ok * V_MAX and 0.5 * fm <= cn["sugar"] <= 2.0 * fm
                                            and cn["sugar+bitter"] < 0.1 * cn["sugar"])
                        if row["valid"]:
                            row["valid"] = _rests(net, T, list(sel_seeds)[:3])
                    except OverflowError:
                        pass
                    rows.append(row)
                    if row["valid"]:
                        cands.append((row, net))
    return _pick_robust(cands, rows, T, robust_seeds)


def _rests(net: IntNet, T: int, seeds) -> bool:
    """After the stimulus every neuron must come to a silent fixed point (`rest_test`), in all stimulated conditions."""
    return rest_failures(net, T, seeds) == 0


def _pick_robust(cands, rows, T: int, robust_seeds):
    """Among valid candidates (sorted by log error) take the first whose rest test passes on `robust_seeds` in all
    three stimulated conditions; if none does, the one with the fewest failures. Records rest_fail in the row.
    Returns (net, rows, row) or (None, rows, None)."""
    if not cands:
        return None, rows, None
    best = None
    for row, net in sorted(cands, key=lambda rn: (rn[0]["err"], -rn[0].get("damp", 0.0))):
        row["rest_fail"] = rest_failures(net, T, robust_seeds)
        row["rest_trials"] = 3 * len(robust_seeds)
        if best is None or row["rest_fail"] < best[0]["rest_fail"]:
            best = (row, net)
        if row["rest_fail"] == 0:
            best = (row, net)
            break
    return best[1], rows, best[0]


def search_plain(brain: Brain, circ: dict, rates: dict[str, np.ndarray], ref_fn, T: int = 120,
                 dts=(1.0, 2.0), thetas=(100, 200, 300), lams=(0.15, 0.25, 0.35),
                 gains=(0.27, 0.3, 0.33, 0.36, 0.4, 0.44, 0.48, 0.53), inh_ratios=(1.0, 0.7, 0.5),
                 cap: int = 200, sel_seeds=range(300, 308), v_ok: float = 0.8, robust_seeds=range(400, 440)):
    """Plainest integerization: weight = clip(round(count * g * int_scale(theta)), +-cap) with ONE global gain g
    for excitatory synapses and g * inh_ratio for inhibitory ones; no per-neuron gains, no damping. Grid over
    (dt, theta, lam, g, inh_ratio); a candidate is valid if no overflow, max |v| <= v_ok * V_MAX on the selection
    seeds, sugar MN9 spike count (L+R) within [0.5, 2] x the float full-brain count and sugar+bitter MN9 < 10% of
    sugar, and the net comes to a silent fixed point after the input stops (`rest_test`); best = lowest `log_error`.
    Returns (best IntNet or None, table rows, best row or None)."""
    n = len(circ["sel"])
    rows, cands = [], []
    for dt in dts:
        ref = ref_fn(T * dt)
        for theta in thetas:
            for lam in lams:
                for g in gains:
                    for ratio in inh_ratios:
                        gn = (np.full(n, g), np.full(n, g * ratio), 1.0)
                        net = integerize(brain, circ, rates, dt, theta, lam, cap, gains=gn)
                        row = dict(dt=dt, theta=theta, lam=lam, gain=g, inh_ratio=ratio, valid=False,
                                   err=float("nan"), vmax=-1, mn9_sugar=float("nan"), mn9_sb=float("nan"))
                        try:
                            err, vmax = log_error(net, ref, T, sel_seeds)
                            mn = [net.meta["outputs"]["MN9_left"], net.meta["outputs"]["MN9_right"]]
                            cn = {c: np.mean([run_int(net, c, T, sd)[1][:, mn].sum() for sd in sel_seeds])
                                  for c in ("sugar", "sugar+bitter")}
                            row.update(err=err, vmax=vmax, mn9_sugar=float(cn["sugar"]), mn9_sb=float(cn["sugar+bitter"]))
                            fm = float(ref["sugar"][mn].sum())
                            row["valid"] = bool(vmax <= v_ok * V_MAX and 0.5 * fm <= cn["sugar"] <= 2.0 * fm
                                                and cn["sugar+bitter"] < 0.1 * cn["sugar"])
                            if row["valid"]:
                                row["valid"] = _rests(net, T, list(sel_seeds)[:3])
                        except OverflowError:
                            pass
                        rows.append(row)
                        if row["valid"]:
                            cands.append((row, net))
    return _pick_robust(cands, rows, T, robust_seeds)


def name_of(flyid: int) -> str:
    return {MN9_LEFT: "MN9_left", MN9_RIGHT: "MN9_right", GIANT_FIBER[0]: "GF_1", GIANT_FIBER[1]: "GF_2",
            ADN1_LEFT: "aDN1_left", ADN1_RIGHT: "aDN1_right"}.get(int(flyid), str(int(flyid)))


def integerize(brain: Brain, circ: dict, rates: dict[str, np.ndarray], dt_ms: float = 2.0, theta: int = 200,
               lam: float = 1.0, cap: int = 200, gains: tuple[np.ndarray, np.ndarray, float] | None = None,
               meta_extra: dict | None = None) -> IntNet:
    """Scale the pruned subcircuit into an IntNet.

    weight_ji = clip(round(count_ji * s_j * int_scale(theta)), +-cap) (sign from the FlyWire neurotransmitter call,
    already in the count; s_j = s_exc[j] for excitatory, s_inh[j] for inhibitory synapses, from `fit_gains`); leak = round(lam * theta * dt_ms / T_MBR)
    (the exponential leak at one membrane time constant, taken at threshold, as a linear leak); one time step =
    dt_ms of fly time (synaptic delay of one step ~ 1.8 ms). rates[cond]: steady-state float rates (Hz) of the
    selected neurons."""
    sc = int_scale(theta)
    leak = int(round(lam * theta * dt_ms / T_MBR))
    s_exc, s_inh, damp = fit_gains(circ, rates, dt_ms, theta, leak) if gains is None else gains
    q = lambda M, d: np.clip(np.rint(np.where(M > 0, M * s_exc[:, None], M * s_inh[:, None]) * sc * d),
                             -cap, cap).astype(np.int64)
    names = [name_of(brain.ids[i]) for i in circ["sel"]]
    inputs = [f"{g}_GRN_{int(brain.ids[i])}" for g, i in zip(circ["grn_group"], circ["grn"])]
    meta = dict(
        source="FlyWire connectome v783 (Dorkenwald et al. 2024, Nature 634:124; Schlegel et al. 2024, Nature "
               "634:139) as packaged for the Shiu et al. 2024 (Nature, doi:10.1038/s41586-024-07763-9) whole-brain "
               "LIF model by github.com/eonsystemspbc/fly-brain (2025_Connectivity_783.parquet)",
        licence="FlyWire public-release data (v783) is CC BY-NC 4.0 (non-commercial, attribution): cite Dorkenwald "
                "et al. 2024, Nature 634:124 and Schlegel et al. 2024, Nature 634:139 (FlyWire), and Shiu et al. "
                "2024 for the LIF model. This file is a derived, pruned and integer-scaled subset (see "
                "demos/fly.py, docs/fly_circuit.md); NOT for commercial use.",
        dt_ms=dt_ms, theta=int(theta), leak=leak, lam=lam, weight_cap=cap,
        scaling=f"w_ji = clip(round(count_ji * gain_j * {sc:.6f}), +-{cap}); count = signed (Excitatory column) "
                f"FlyWire synapse count; {sc:.6f} = theta / 7mV * 0.275mV * 5ms / 20ms (integer units per synapse "
                f"at unit gain); gain_j = per-postsynaptic-neuron rate-matched gain (fit_gains; excitatory/inhibitory "
                f"synapses have separate gains, listed in gains_exc / gains_inh); leak = round(lam * theta * dt_ms / 20ms)",
        rec_damp=round(float(damp), 5), gains_exc=[round(float(x), 5) for x in s_exc], gains_inh=[round(float(x), 5) for x in s_inh],
        input_rate_hz=RATE_HZ, input_p_per_step=RATE_HZ * dt_ms * 1e-3,
        input="external lines are the gustatory receptor neurons (GRNs), one Bernoulli spike train per line per "
              "step with p = rate * dt; sugar group at 200 Hz for sugar, bitter group at 200 Hz for bitter",
        input_groups={g: [k for k, name in enumerate(inputs) if name.startswith(g + "_")] for g in ("sugar", "bitter")},
        outputs=dict(MN9_left=names.index("MN9_left"), MN9_right=names.index("MN9_right")),
        flywire_ids=[int(brain.ids[i]) for i in circ["sel"]],
        grn_flywire_ids=[int(brain.ids[i]) for i in circ["grn"]],
        selection=circ["info"],
        criterion="candidates: non-GRN neurons firing >= min_rate Hz in the float full-brain model under sugar, "
                  "sugar+bitter or bitter; ranked by activity-weighted backward influence on MN9 (max over "
                  "conditions); top n kept; synapses with |count| < c_min dropped to fit nnz <= 4000",
    )
    if meta_extra:
        meta.update(meta_extra)
    return IntNet(names, inputs, q(circ["W"], damp), q(circ["U"], 1.0), theta, leak, meta)


DOOM_LIMITS = dict(n=200, m=80, nnz=4000, w=200, t=120)


def check_limits(net: IntNet, t: int | None = None, limits: dict | None = None) -> dict:
    lim = DOOM_LIMITS if limits is None else limits
    nnz = int(np.count_nonzero(net.W) + np.count_nonzero(net.U))
    wmax = int(max(np.abs(net.W).max(initial=0), np.abs(net.U).max(initial=0)))
    out = dict(n=net.n, m=len(net.inputs), nnz=nnz, w=wmax, t=t)
    ok = (net.n <= lim["n"] and out["m"] <= lim["m"] and nnz <= lim["nnz"]
          and wmax <= lim["w"] and (t is None or t <= lim["t"]))
    out["ok"] = bool(ok)
    return out


def bernoulli_inputs(net: IntNet, cond: str, T: int, seed: int, rates: dict[str, float] | None = None) -> np.ndarray:
    """(T, M) input spikes: every line of an active group fires with p = rate * dt per step (seeded).
    `rates` ({group: Hz}) overrides the condition's rates (held-out stimulus strengths). Conditions come from
    net.meta["conditions"] if present (combined brain), else from CONDITIONS."""
    rng = np.random.default_rng(seed)
    X = np.zeros((T, len(net.inputs)), np.int64)
    conds = net.meta.get("conditions", CONDITIONS)       # combined brain: conditions live in the meta
    for g, hz in (conds[cond] if rates is None else rates).items():
        cols = net.meta["input_groups"][g]
        X[:, cols] = rng.random((T, len(cols))) < hz * net.meta["dt_ms"] * 1e-3
    return X


def run_int(net: IntNet, cond: str, T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """IntNet.run (the exact contract DOOM executes) on a seeded Bernoulli input; returns (V, S)."""
    return net.run(bernoulli_inputs(net, cond, T, seed))


def rest_test(net: IntNet, cond: str, T: int, seeds, steps: int = 3000, chunk: int = 250) -> dict:
    """Does the net come to rest after the input stops? Run T steps with the condition's input, then up to `steps`
    steps of zero input (IntNet.run from the final state, in chunks; stops early once a chunk has no spike and v is
    static). Reports, worst case over seeds: spikes after input stop, step (after stop) of the last spike, step from
    which v no longer changes (None if it still changes after `steps`, or the net overflowed = latched
    reverberation), and whether the fixed point is silent (all v < theta, so no spike can ever occur again)."""
    last, settle, nsp, ok, overflow, n_fail = 0, 0, 0, True, False, 0
    zero = np.zeros((chunk, len(net.inputs)), np.int64)
    for sd in seeds:
        V, _ = run_int(net, cond, T, sd)
        v, done, seed_ok = V[-1], 0, False
        try:
            while done < steps:
                V2, S2 = net.run(zero, v0=v)
                nsp = max(nsp, int(S2.sum()))
                sp_t = np.flatnonzero(S2.any(1))
                if len(sp_t):
                    last = max(last, done + int(sp_t[-1]) + 1)
                moving = np.flatnonzero((V2[1:] != V2[:-1]).any(1))
                if len(moving):
                    settle = max(settle, done + int(moving[-1]) + 1)
                v, done = V2[-1], done + chunk
                if not len(sp_t) and not len(moving):
                    seed_ok = bool((v < net.theta).all())
                    break
        except OverflowError:
            overflow = True
        if not seed_ok:
            ok = False
            n_fail += 1
            if overflow:
                break
    return dict(spikes_after_stop=nsp, last_spike_step=last, settle_step=settle if ok else None,
                silent_fixed_point=bool(ok), overflow=overflow, failed_seeds=n_fail)


def rest_failures(net: IntNet, T: int, seeds, conds=("sugar", "sugar+bitter", "bitter")) -> int:
    """Number of (condition, seed) trials after which the net does NOT reach a silent fixed point."""
    n = 0
    for c in conds:
        for sd in seeds:
            n += not rest_test(net, c, T, [sd])["silent_fixed_point"]
    return n


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def evaluate_int(net: IntNet, ref: dict[str, np.ndarray], seeds, T: int, act: float = 0.5) -> dict:
    """Integer model vs float reference. ref[cond]: mean spike count per selected neuron over the same time window
    (T * dt_ms) of the full-brain float runs. Active = mean count >= `act` spikes in the window.
    Returns per-condition: MN9 spike counts per seed, mean counts, Jaccard of active sets, rate correlation, max |v|."""
    mn = [net.meta["outputs"]["MN9_left"], net.meta["outputs"]["MN9_right"]]
    out = {}
    for cond in CONDITIONS:
        cnt, vmax = [], 0
        for sd in seeds:
            V, S = run_int(net, cond, T, sd)
            cnt.append(S.sum(0))
            vmax = max(vmax, int(np.abs(V).max()))
        cnt = np.array(cnt)
        mean = cnt.mean(0)
        r = ref[cond]
        corr = float(np.corrcoef(mean, r)[0, 1]) if mean.std() > 0 and r.std() > 0 else float("nan")
        out[cond] = dict(mn9=cnt[:, mn], mn9_mean=cnt[:, mn].mean(0), mn9_float=r[mn], jaccard=jaccard(mean >= act, r >= act),
                         corr=corr, vmax=vmax, n_active=int((mean >= act).sum()), n_active_float=int((r >= act).sum()))
    return out


# =====================================================================================================================
# Combined brain: feeding + escape + grooming in one IntNet
# =====================================================================================================================
INPUT_GROUPS = {"sugar": SUGAR_GRNS, "bitter": BITTER_GRNS, "lc4": LC_4S, "jo": ALL_JOS, "p9": P9}
INPUT_HZ = {"sugar": 200.0, "bitter": 200.0, "lc4": 200.0, "jo": 300.0, "p9": 100.0}   # notebook rates (Poisson, Hz)
OUTPUTS = {"MN9_left": MN9_LEFT, "MN9_right": MN9_RIGHT, "GF_1": GIANT_FIBER[0], "GF_2": GIANT_FIBER[1],
           "aDN1_right": ADN1_RIGHT}
# behaviour: its output neurons, the stimulus conditions that are used to rank neurons for it
BEHAVIOURS = {
    "feeding": dict(outputs=["MN9_left", "MN9_right"], stim="sugar", conds=["sugar", "sugar+bitter", "bitter"]),
    "escape": dict(outputs=["GF_1", "GF_2"], stim="lc4", conds=["lc4", "lc4+p9"]),
    "grooming": dict(outputs=["aDN1_right"], stim="jo", conds=["jo"]),
}
_COMBOS = ["none", "sugar", "bitter", "sugar+bitter", "lc4", "p9", "lc4+p9", "jo", "sugar+lc4", "sugar+jo", "lc4+jo"]
BRAIN_CONDITIONS = {c: ({} if c == "none" else {g: INPUT_HZ[g] for g in c.split("+")}) for c in _COMBOS}
BRAIN_LIMITS = dict(n=260, m=200, nnz=6000, w=200, t=120)
SELECTION_CONDS = ["sugar", "sugar+bitter", "bitter", "lc4", "lc4+p9", "jo"]       # float 1 s runs (candidate rates)


def select_multi(brain: Brain, rates: dict[str, np.ndarray], groups: dict[str, np.ndarray], quotas: dict[str, int],
                 n_max: int = 260, nnz_max: int = 6000, m_max: int = 200, min_rate: float = 0.5,
                 depth: int = 5, line_caps: dict | None = None) -> dict:
    """Joint subcircuit for several behaviours. Inputs (all `groups`) are input lines, never simulated.

    Candidates: non-input neurons firing >= min_rate Hz in any selection condition, plus the outputs. For every
    behaviour the candidates are ranked by `influence` on its output neurons (conditions BEHAVIOURS[b]["conds"],
    input drive of the active groups in the denominator); the top quotas[b] of each ranking are united (outputs
    always in). Pruning: synapses |count| < c_min dropped, c_min the smallest integer with nnz(W)+nnz(U) <= nnz_max;
    if more than m_max input lines remain, the sugar / bitter / P9 lines are kept and the LC4 / JO lines with the least
    synapse mass are dropped."""
    in_idx = np.concatenate([groups[g] for g in groups])
    in_grp = np.concatenate([[g] * len(groups[g]) for g in groups])
    out_idx = brain.idx(list(OUTPUTS.values()))
    C = brain.counts_matrix().tocsr()
    mask = np.zeros(brain.n, bool)
    for c in SELECTION_CONDS:
        mask |= rates[c] >= min_rate
    mask[in_idx] = False
    mask[out_idx] = True
    cand = np.flatnonzero(mask)
    Cc = C[cand][:, cand].astype(float).tocsr()
    Cin = abs(C[cand][:, in_idx]).tocsr()
    score, chosen = {}, set(np.searchsorted(cand, out_idx).tolist())
    for b, spec in BEHAVIOURS.items():
        drive = {}
        for c in spec["conds"]:
            gv = np.array([BRAIN_CONDITIONS[c].get(g, 0.0) for g in in_grp])
            drive[c] = np.asarray(Cin.multiply(gv[None, :]).sum(1)).ravel()
        tg = np.searchsorted(cand, brain.idx([OUTPUTS[o] for o in spec["outputs"]]))
        score[b] = influence(Cc, {c: rates[c][cand] for c in spec["conds"]}, drive, tg, depth)
        chosen |= set(np.argsort(-score[b], kind="stable")[:quotas[b]].tolist())
    norm = np.max([score[b] / score[b].max() for b in score], 0)
    chosen = sorted(chosen, key=lambda i: -norm[i])[:n_max]
    sel = np.sort(cand[chosen])
    Wf = sp.coo_matrix(C[sel][:, sel])
    Uf = sp.coo_matrix(C[sel][:, in_idx])
    c_min, dropped_lines = 1, 0
    while True:
        kw, ku = abs(Wf.data) >= c_min, abs(Uf.data) >= c_min
        lines = np.unique(Uf.col[ku])
        for gname, cap_g in (line_caps or {}).items():     # per-group cap on input lines (least mass dropped)
            gl = lines[in_grp[lines] == gname]
            if len(gl) > cap_g:
                m_g = np.bincount(Uf.col[ku], weights=abs(Uf.data[ku]), minlength=len(in_idx))
                drop = gl[np.argsort(-m_g[gl], kind="stable")[cap_g:]]
                ku = ku & ~np.isin(Uf.col, drop)
                lines = np.unique(Uf.col[ku])
        if len(lines) > m_max:                      # keep the m_max input lines with the most synapse mass
            mass = np.bincount(Uf.col[ku], weights=abs(Uf.data[ku]), minlength=len(in_idx))
            prio = np.isin(in_grp, ["sugar", "bitter", "p9"]) & (mass > 0)      # the small taste/walk groups stay whole
            keep_lines = np.lexsort((-mass, ~prio))[:m_max]
            ku = ku & np.isin(Uf.col, keep_lines)
            dropped_lines = len(lines) - m_max
            lines = np.unique(Uf.col[ku])
        if kw.sum() + ku.sum() <= nnz_max:
            break
        c_min += 1
    W = sp.coo_matrix((Wf.data[kw], (Wf.row[kw], Wf.col[kw])), shape=Wf.shape).toarray()
    U = sp.coo_matrix((Uf.data[ku], (Uf.row[ku], Uf.col[ku])), shape=Uf.shape).toarray()[:, lines]
    mag = np.concatenate([abs(Wf.data), abs(Uf.data)])
    info = dict(n_candidates=int(len(cand)), c_min=int(c_min), dropped_lines=int(dropped_lines), quotas=quotas, line_caps=line_caps or {}, min_rate=min_rate, depth=depth,
                edges_total=int(len(mag)), edges_kept=int(kw.sum() + ku.sum()), syn_mass_total=int(mag.sum()),
                syn_mass_kept=int(abs(Wf.data[kw]).sum() + abs(Uf.data[ku]).sum()),
                per_behaviour_selected={b: int(np.isin(sel, cand[np.argsort(-score[b], kind="stable")[:quotas[b]]]).sum())
                                        for b in score})
    return dict(sel=sel, grn=in_idx[lines], grn_group=in_grp[lines], W=W, U=U, info=info)


def build_plain(brain: Brain, circ: dict, dt_ms: float, theta: int, lam: float, g: float, inh_ratio: float,
                cap: int = 200) -> IntNet:
    """Plain integerization of a joint circuit: w = clip(round(count * g * int_scale(theta)), +-cap), inhibitory
    synapses additionally times inh_ratio; ONE global scale, no per-neuron gains; leak = round(lam*theta*dt/20ms).
    Meta carries everything DOOM / the embodiment needs: groups, rates, p per step, conditions, outputs."""
    sc = int_scale(theta) * g
    leak = int(round(lam * theta * dt_ms / T_MBR))
    q = lambda M: np.clip(np.rint(np.where(M > 0, M * sc, M * sc * inh_ratio)), -cap, cap).astype(np.int64)
    names = [name_of(brain.ids[i]) for i in circ["sel"]]
    inputs = [f"{gr}_{int(brain.ids[i])}" for gr, i in zip(circ["grn_group"], circ["grn"])]
    groups_used = [gr for gr in INPUT_GROUPS if gr in set(circ["grn_group"])]
    meta = dict(
        variant="plain", dt_ms=dt_ms, theta=int(theta), leak=leak, lam=lam, weight_cap=cap, gain=g,
        inh_ratio=inh_ratio,
        scaling=f"w_ji = clip(round(count_ji * {sc:.6f}), +-{cap}) excitatory; x{inh_ratio} for inhibitory; count = signed "
                f"(Excitatory column) FlyWire synapse count; {sc:.6f} = g * theta / 7mV * 0.275mV * 5ms / 20ms with ONE "
                f"global g = {g}; no per-neuron gains, no damping; leak = round(lam * theta * dt_ms / 20ms), lam = {lam}",
        source="FlyWire connectome v783 (Dorkenwald et al. 2024, Nature 634:124; Schlegel et al. 2024, Nature 634:139) "
               "as packaged for the Shiu et al. 2024 (Nature, doi:10.1038/s41586-024-07763-9) whole-brain LIF model by "
               "github.com/eonsystemspbc/fly-brain (2025_Connectivity_783.parquet)",
        licence="FlyWire public-release data (v783) is CC BY-NC 4.0 (non-commercial, attribution): cite Dorkenwald et al. "
                "2024 and Schlegel et al. 2024 (FlyWire), Shiu et al. 2024 for the LIF model. This file is a derived, "
                "pruned and integer-scaled subset (demos/fly.py); NOT for commercial use.",
        input_groups={gr: [k for k, nm in enumerate(inputs) if nm.startswith(gr + "_")] for gr in groups_used},
        input_hz={gr: INPUT_HZ[gr] for gr in groups_used},
        input_p_per_step={gr: round(INPUT_HZ[gr] * dt_ms * 1e-3, 6) for gr in groups_used},
        input="external lines = sensory neurons of FlyWire (sugar / bitter GRNs, LC4 looming neurons, Johnston's organ "
              "neurons, P9 walking neurons), one Bernoulli spike train per line per step with p = rate * dt",
        conditions={c: v for c, v in BRAIN_CONDITIONS.items()},
        behaviours={b: dict(outputs=s["outputs"], stimulus=s["stim"]) for b, s in BEHAVIOURS.items()},
        outputs={o: names.index(o) for o in OUTPUTS if o in names},
        flywire_ids=[int(brain.ids[i]) for i in circ["sel"]],
        input_flywire_ids=[int(brain.ids[i]) for i in circ["grn"]],
        selection=circ["info"],
        criterion="candidates: non-input neurons firing >= min_rate Hz in the float full-brain model under sugar, "
                  "sugar+bitter, bitter, lc4, lc4+p9, jo; per behaviour ranked by activity-weighted backward influence on "
                  "its outputs; per-behaviour quotas united; synapses |count| < c_min dropped to fit nnz and lines",
    )
    return IntNet(names, inputs, q(circ["W"]), q(circ["U"]), theta, leak, meta)


def brain_ref(brain: Brain, groups: dict[str, np.ndarray], conds, window_ms: float, seeds, sel: np.ndarray,
              t_ms: float | None = None) -> dict[str, np.ndarray]:
    """Mean full-brain float spike counts of the selected neurons in [0, window) for every condition (fresh seeds)."""
    out = {}
    for c in conds:
        inp = [(groups[g], hz) for g, hz in BRAIN_CONDITIONS[c].items()]
        out[c] = np.mean([spike_counts(brain.simulate(inp, t_ms or window_ms, sd), brain.n, 0.0, window_ms)[sel]
                          for sd in seeds], 0)
    return out


def eval_brain(net: IntNet, ref: dict[str, np.ndarray], conds, seeds, T: int, act: float = 0.5) -> dict:
    """IntNet.run on every condition/seed: spike counts (seeds, N), max |v|, active-set Jaccard and per-neuron rate
    correlation with the float window counts ref[cond] (selected neurons). Raises OverflowError via IntNet.run."""
    out = {}
    for c in conds:
        cnt, vmax = [], 0
        for sd in seeds:
            V, S = run_int(net, c, T, sd)
            cnt.append(S.sum(0))
            vmax = max(vmax, int(np.abs(V).max()))
        cnt = np.array(cnt)
        m, r = cnt.mean(0), ref[c]
        out[c] = dict(counts=cnt, vmax=vmax, jaccard=jaccard(m >= act, r >= act),
                      corr=float(np.corrcoef(m, r)[0, 1]) if m.std() > 0 and r.std() > 0 else float("nan"),
                      n_active=int((m >= act).sum()), n_active_float=int((r >= act).sum()))
    return out


def behaviour_check(net: IntNet, ev: dict, ref: dict, lo: float = 0.4, hi: float = 2.5) -> tuple[bool, dict]:
    """Per condition and behaviour: integer output spikes (sum over the behaviour's outputs, mean over seeds) vs float.
    float on (>= 3 spikes): need lo*float <= int <= hi*float; float off (< 3): int <= 10% of the behaviour's on count
    (float on count at its own stimulus, scaled to int via the same condition) and at most 1.5 spikes."""
    out_idx = {o: net.meta["outputs"][o] for o in net.meta["outputs"]}
    ok, detail = True, {}
    for b, spec in BEHAVIOURS.items():
        ix = [out_idx[o] for o in spec["outputs"]]
        on_int = ev[spec["stim"]]["counts"][:, ix].sum(1).mean()
        for c in ev:
            fi = ref[c][ix].sum()
            ii = ev[c]["counts"][:, ix].sum(1).mean()
            if fi >= 3:
                good = lo * fi <= ii <= hi * fi
            else:
                good = ii <= max(1.5, 0.1 * on_int) if c != spec["stim"] else False
            detail[(b, c)] = (float(ii), float(fi), bool(good))
            ok &= good
    return ok, detail


def search_brain(brain: Brain, circ: dict, ref_fn, T: int = 120, dts=(1.0,), thetas=(200, 300, 500, 700), lams=(0.15, 0.25),
                 gains=(0.3, 0.34, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58), inh_ratios=(1.0, 0.7), cap: int = 200,
                 sel_seeds=range(300, 308), v_ok: float = 0.8, robust_seeds=range(400, 420)):
    """Plain-integerization grid for the joint circuit. Valid: no overflow, max|v| <= v_ok V_MAX, `behaviour_check`
    on all conditions (on within [0.4, 2.5] x float, crosstalk/suppression off), silent fixed point after the input
    in every condition (3 seeds). Ranked by log spike-count error over the stimulated conditions, then the first that
    has 0 rest failures on `robust_seeds` (x all conditions) wins; otherwise the fewest failures.
    Returns (net or None, rows, row)."""
    rows, cands = [], []
    conds = list(BRAIN_CONDITIONS)
    stim_conds = [c for c in conds if c != "none"]
    for dt in dts:
        ref = ref_fn(T * dt)
        for theta in thetas:
            for lam in lams:
                for g in gains:
                    for ratio in inh_ratios:
                        net = build_plain(brain, circ, dt, theta, lam, g, ratio, cap)
                        row = dict(dt=dt, theta=theta, lam=lam, gain=g, inh_ratio=ratio, valid=False, err=float("nan"),
                                   vmax=-1)
                        try:
                            ev = eval_brain(net, ref, conds, sel_seeds, T)
                        except OverflowError:
                            rows.append(row)
                            continue
                        ok, det = behaviour_check(net, ev, ref)
                        err = float(np.mean([np.abs(np.log((ref[c] + T * dt / 120) / (ev[c]["counts"].mean(0) + T * dt / 120))).mean()
                                             for c in stim_conds]))
                        vmax = max(e["vmax"] for e in ev.values())
                        row.update(err=err, vmax=vmax, **{f"{b}": round(det[(b, BEHAVIOURS[b]["stim"])][0], 1)
                                                          for b in BEHAVIOURS})
                        row["valid"] = bool(ok and vmax <= v_ok * V_MAX and rest_failures(net, T, list(sel_seeds)[:2], conds) == 0)
                        rows.append(row)
                        if row["valid"]:
                            cands.append((row, net))
    if not cands:
        return None, rows, None
    best = None
    for row, net in sorted(cands, key=lambda rn: rn[0]["err"]):
        row["rest_fail"] = rest_failures(net, T, robust_seeds, conds)
        row["rest_trials"] = len(conds) * len(robust_seeds)
        if best is None or row["rest_fail"] < best[0]["rest_fail"]:
            best = (row, net)
        if row["rest_fail"] == 0:
            break
    return best[1], rows, best[0]
