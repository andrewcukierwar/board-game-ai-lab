"""Independent results verifier for Phase 4D.4J-B (separate agent, same host).

Written without reading or importing analyze.py, guard.py or the 4D.4J-A
reference board. Own Connect 4 rules (column lists + explicit line scans),
own guard-rule re-derivation, own statistics and own bootstrap RNG stream.
Torch-free; never opens files whose name contains "sealed"; never loads a
checkpoint (checkpoint files are only hashed as raw bytes).

Usage (from the worktree root):
  python research/alphazero_tactical_guard_confirmation/independent_verify.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "research" / "alphazero_tactical_guard_confirmation"
RES = PKG / "results"
OUT = RES / "independent-verification.json"
A_TASKS = ROOT / "research" / "alphazero_tactical_guard" / "results" / "tasks"
GUARD = ROOT / "research" / "alphazero_tactical_guard" / "guard.py"

GUARD_SHA = "d2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04"
CKPT_SHA = {
    42: "ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b",
    314159: "2646d7cdee7d7b069412688bf95fd9794bea30402b4952120c904dbcb399401d",
}
PROSPECTIVE = "ae4715ff30d6796b010f115185ad90514e346141"
SEEDS = (42, 314159)
OPPS = ("negamax3", "negamax5")
ARMS = ("control", "guarded")
TASKS = {f"{s}-{o}" for s in SEEDS for o in OPPS}
MARGIN = 0.02
BOOT_SEED = 99020001
REPS = 10000
COLS, ROWS = 7, 6


def load(path: Path):
    assert "sealed" not in path.name.lower(), path
    with open(path) as fh:
        return json.load(fh)


def sha256(path: Path) -> str:
    assert "sealed" not in str(path).lower(), path
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Own Connect 4 rules: board = list of 7 columns, each a bottom-up list of
# colors (0 = X, 1 = O). Win detection by explicit scans through a cell.
# --------------------------------------------------------------------------
class C4:
    def __init__(self):
        self.cols = [[] for _ in range(COLS)]
        self.n = 0

    def copy(self):
        b = C4()
        b.cols = [list(c) for c in self.cols]
        b.n = self.n
        return b

    def to_move(self) -> int:
        return self.n % 2

    def legal(self):
        return [c for c in range(COLS) if len(self.cols[c]) < ROWS]

    def cell(self, c, r):
        if 0 <= c < COLS and 0 <= r < len(self.cols[c]):
            return self.cols[c][r]
        return None

    def drop(self, c) -> bool:
        """Play color to move in column c; return True if it makes four."""
        if not (0 <= c < COLS) or len(self.cols[c]) >= ROWS:
            raise ValueError(f"illegal move {c}")
        color = self.to_move()
        r = len(self.cols[c])
        self.cols[c].append(color)
        self.n += 1
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            run = 1
            for sgn in (1, -1):
                k = 1
                while self.cell(c + sgn * k * dc, r + sgn * k * dr) == color:
                    run += 1
                    k += 1
            if run >= 4:
                return True
        return False

    def full(self):
        return self.n == COLS * ROWS


def classify(board: C4):
    """Immediate wins, safe/unsafe actions and winning replies for side to move."""
    legal = board.legal()
    wins, safe, unsafe, replies, inspected = [], [], [], {}, 0
    for a in legal:
        b = board.copy()
        if b.drop(a):
            wins.append(a)
            safe.append(a)
            continue
        w = []
        for r in b.legal():
            inspected += 1
            bb = b.copy()
            if bb.drop(r):
                w.append(r)
        if w:
            unsafe.append(a)
            replies[str(a)] = w
        else:
            safe.append(a)
    return dict(legal=legal, wins=wins, safe=safe, unsafe=unsafe, replies=replies,
                inspected=inspected)


def best(actions, visits):
    """Max visits, ties -> lowest column."""
    return min(actions, key=lambda a: (-visits[a], a))


def frozen_rule(cls, original, visits):
    if cls["wins"]:
        return "IMMEDIATE_WIN", best(cls["wins"], visits)
    if original in cls["safe"]:
        return "UNCHANGED", original
    if cls["safe"]:
        return "AVOID_IMMEDIATE_LOSS", best(cls["safe"], visits)
    return "NO_SAFE_MOVE", original


def phase(occ):
    return "opening" if occ <= 13 else ("middle" if occ <= 27 else "late")


# --------------------------------------------------------------------------
def main():
    issues = []
    out = {"contract": "phase4d4j-b-independent-verification-v1",
           "note": "same-host verification by a separate agent; not external replication",
           "bootstrap": {"generator": "numpy PCG64", "seed": BOOT_SEED, "replicates": REPS,
                         "unit": "whole opening family, all seeds/opponents/colors/arms together"}}

    def check(name, ok, detail=None):
        if not ok:
            issues.append({"check": name, "detail": detail})
        return bool(ok)

    design = load(RES / "design.json")
    openings = load(RES / "openings.json")
    analysis = load(RES / "analysis.json")
    freeze = load(RES / "freeze.json")
    execution = load(RES / "execution.json")
    preservation = load(RES / "preservation.json")

    fam_prefix = {o["family"]: o["moves"] for o in openings["confirmation"]}
    fam_stratum = {o["family"]: o["stratum"] for o in openings["confirmation"]}
    families = sorted(fam_prefix)
    check("families==96", len(families) == 96 == design["families"], len(families))
    check("strata 24 each", Counter(fam_stratum.values()) ==
          Counter({k: v for k, v in design["strata"].items()}), dict(Counter(fam_stratum.values())))

    # Openings themselves: legal, non-terminal, no immediate win for either side.
    open_bad = []
    for f, mv in fam_prefix.items():
        b = C4()
        for m in mv:
            if b.drop(m):
                open_bad.append((f, "ends"))
                break
        cls = classify(b)
        if cls["wins"]:
            open_bad.append((f, "side-to-move win"))
        b2 = b.copy()
        b2.n += 1  # flip side to move to test the opponent's immediate wins
        opp_wins = [a for a in b2.legal() if b2.copy().drop(a)]
        if opp_wins:
            open_bad.append((f, "opponent win"))
        if int(fam_stratum[f][3:]) != len(mv):
            open_bad.append((f, "stratum length"))
    check("openings valid", not open_bad, open_bad)
    out["openings"] = {"families": len(families), "invalid": open_bad,
                       "file_sha256": sha256(RES / "openings.json"),
                       "file_sha256_matches_design": sha256(RES / "openings.json") == design["openings_file_sha256"]}
    check("openings file sha", out["openings"]["file_sha256_matches_design"])

    # ---------------- tasks and identity ----------------
    task_dirs = sorted(p.name for p in (RES / "tasks").iterdir() if p.is_dir())
    check("task set", set(task_dirs) == TASKS, task_dirs)
    identity = {}
    for t in sorted(TASKS):
        idn = load(RES / "tasks" / t / "identity.json")
        seed = int(t.split("-")[0])
        ok = (idn["before"] == idn["after"] and idn["checkpoint"]["sha256"] == CKPT_SHA[seed]
              and idn["weights_sha256"] == idn["before"]["model"] == idn["after"]["model"]
              and idn["complete"] is True and idn["matches"] == design["matches_per_task"]
              and idn["opponent"] == t.split("-")[1] and idn["seed"] == seed)
        check(f"identity {t}", ok)
        identity[t] = {"before_equals_after": idn["before"] == idn["after"],
                       "checkpoint_sha256": idn["checkpoint"]["sha256"],
                       "weights_sha256": idn["weights_sha256"], "ok": ok}
    out["identity"] = identity

    # ---------------- hashes ----------------
    gsha = sha256(GUARD)
    hashes = {"guard_sha256": gsha, "guard_ok": gsha == GUARD_SHA == freeze["guard_sha256"]
              == design["guard"]["sha256"], "checkpoints": {}}
    check("guard sha", hashes["guard_ok"], gsha)
    for ck in freeze["checkpoints"]:
        p = Path(ck["path"])
        seed = 42 if "/42-" in ck["path"] else 314159
        actual = sha256(p) if p.exists() else None
        ok = actual == ck["sha256"] == CKPT_SHA[seed]
        hashes["checkpoints"][str(seed)] = {"recorded": ck["sha256"], "actual": actual, "ok": ok}
        check(f"checkpoint sha {seed}", ok, actual)
    frozen_changed = []
    for rel, h in freeze["files"].items():
        if "sealed" in rel.lower():
            continue
        p = ROOT / rel
        if not p.exists() or sha256(p) != h:
            frozen_changed.append(rel)
    hashes["freeze_files_checked"] = len(freeze["files"])
    hashes["freeze_files_changed"] = frozen_changed
    check("freeze files unchanged", not frozen_changed, frozen_changed)
    out["hashes"] = hashes

    # ---------------- replay all games ----------------
    games = {}  # (seed, opp, arm, family, color) -> record
    counts = Counter()
    replay_bad = []
    guard_rows = []  # per guarded agent decision
    control_bad = []
    visits_bad = []
    orig_not_max = 0
    for t in sorted(TASKS):
        seed, opp = int(t.split("-")[0]), t.split("-")[1]
        for arm in ARMS:
            doc = load(RES / "tasks" / t / f"{arm}-matches.json")
            check(f"doc meta {t} {arm}", doc["complete"] is True and doc["opponent"] == opp
                  and doc["seed"] == seed and doc["variant"] == arm)
            for rec in doc["records"]:
                counts[(seed, opp, arm)] += 1
                fam, col = rec["family"], rec["agent_color"]
                key = (seed, opp, arm, fam, col)
                if key in games:
                    replay_bad.append((key, "duplicate"))
                games[key] = rec
                pre = fam_prefix.get(fam)
                mv = rec["moves"]
                errs = []
                if rec.get("abandoned"):
                    errs.append("abandoned")
                if rec["opponent"] != opp or rec["agent"] != arm:
                    errs.append("labels")
                if pre is None or mv[:len(pre)] != pre or rec["prefix_length"] != len(pre):
                    errs.append("prefix")
                if rec["stratum"] != fam_stratum.get(fam):
                    errs.append("stratum")
                b, winner = C4(), -1
                for i, m in enumerate(mv):
                    if winner != -1 or b.full():
                        errs.append("move after end")
                        break
                    try:
                        mover = b.to_move()
                        if b.drop(m):
                            winner = mover
                    except ValueError:
                        errs.append(f"illegal at {i}")
                        break
                if winner == -1 and not b.full():
                    errs.append("not terminal")
                if rec["winner"] != winner:
                    errs.append("winner")
                res = "draw" if winner == -1 else ("win" if winner == col else "loss")
                if rec["result"] != res or rec["score"] != {"win": 1.0, "draw": 0.5, "loss": 0.0}[res]:
                    errs.append("result/score")
                decs = rec["decisions"]
                if len(decs) != len(mv) - len(pre or []):
                    errs.append("decision count")
                # walk decisions
                b = C4()
                for m in (pre or []):
                    b.drop(m)
                for j, d in enumerate(decs):
                    ply = len(pre) + j
                    if d["color"] != b.to_move() or d["move"] != mv[ply]:
                        errs.append(f"decision {j} color/move")
                        break
                    if d["color"] == col:
                        v = d["visits"]
                        if sum(v) != 256 or len(v) != 7:
                            visits_bad.append((key, j, "sum"))
                        if v[d["original_action"]] != max(v):
                            orig_not_max += 1
                        if d["final_action"] != d["move"]:
                            errs.append(f"decision {j} final!=move")
                        if arm == "control":
                            if d["guard"] is not None or d["move"] != d["original_action"]:
                                control_bad.append((key, j))
                        else:
                            g = d["guard"]
                            cls = classify(b)
                            reason, fin = frozen_rule(cls, g["original_action"], g["original_visits"])
                            mism = []
                            if g["original_visits"] != v:
                                mism.append("original_visits!=visits")
                            if g["original_action"] != d["original_action"]:
                                mism.append("original_action")
                            if sorted(g["immediate_wins"]) != cls["wins"]:
                                mism.append("immediate_wins")
                            if sorted(g["safe_actions"]) != cls["safe"]:
                                mism.append("safe")
                            if sorted(g["unsafe_actions"]) != cls["unsafe"]:
                                mism.append("unsafe")
                            if {k: sorted(x) for k, x in g["opponent_winning_replies"].items()} != cls["replies"]:
                                mism.append("replies")
                            if sorted(g["legal_actions"]) != cls["legal"]:
                                mism.append("legal")
                            if g["reason"] != reason:
                                mism.append("reason")
                            if g["final_action"] != fin or d["move"] != fin or d["final_action"] != fin:
                                mism.append("final")
                            if g["changed"] != (fin != g["original_action"]):
                                mism.append("changed")
                            if g["selected_zero_visits"] != (v[fin] == 0):
                                mism.append("zero")
                            if g["selected_visits"] != v[fin]:
                                mism.append("selected_visits")
                            if g["occupancy"] != b.n or d["occupancy"] != b.n:
                                mism.append("occupancy")
                            if g["no_safe_action"] != (not cls["safe"]):
                                mism.append("no_safe_action")
                            if g["opponent_replies_inspected"] != cls["inspected"]:
                                mism.append("replies_inspected")
                            if g["candidate_count"] != len(cls["legal"]):
                                mism.append("candidate_count")
                            if reason == "AVOID_IMMEDIATE_LOSS":
                                # witness: replay original and a recorded winning reply
                                bb = b.copy()
                                bb.drop(g["original_action"])
                                rep = g["opponent_winning_replies"][str(g["original_action"])][0]
                                if not bb.drop(rep):
                                    mism.append("witness")
                            guard_rows.append(dict(seed=seed, opp=opp, family=fam, color=col,
                                                   stratum=fam_stratum[fam], occ=b.n, ply=ply,
                                                   reason=reason, changed=fin != g["original_action"],
                                                   zero=v[fin] == 0, original=g["original_action"],
                                                   final=fin, gs=g["guard_seconds"],
                                                   ss=d["search_seconds"], mism=mism))
                    b.drop(d["move"])
                if errs:
                    replay_bad.append((key, errs))
    n_games = sum(counts.values())
    check("1536 games", n_games == 1536 == design["matches"], n_games)
    check("per-arm counts", all(counts[(s, o, a)] == 192 for s in SEEDS for o in OPPS for a in ARMS), dict((str(k), v) for k, v in counts.items()))
    balance_bad = [(s, o, a, f) for s in SEEDS for o in OPPS for a in ARMS for f in families
                   if not ((s, o, a, f, 0) in games and (s, o, a, f, 1) in games)]
    check("color balance", not balance_bad and len(games) == 1536, balance_bad[:5])
    check("replay", not replay_bad, replay_bad[:5])
    check("control decisions", not control_bad, control_bad[:5])
    check("visit sums", not visits_bad, visits_bad[:5])
    mism_rows = [r for r in guard_rows if r["mism"]]
    mism_kinds = Counter(k for r in mism_rows for k in r["mism"])
    check("guard decisions vs independent rule", not mism_rows, dict(mism_kinds))
    out["replay"] = {"games": n_games, "families": len(families), "invalid_games": len(replay_bad),
                     "examples": [str(x) for x in replay_bad[:5]], "color_balance_violations": len(balance_bad),
                     "abandoned": sum(1 for r in games.values() if r.get("abandoned"))}
    out["decisions"] = {"guarded_agent_decisions": len(guard_rows),
                        "control_agent_decisions": sum(1 for (s, o, a, f, c), r in games.items() if a == "control"
                                                       for d in r["decisions"] if d["color"] == c),
                        "guard_mismatches": len(mism_rows), "mismatch_kinds": dict(mism_kinds),
                        "control_violations": len(control_bad), "visit_sum_violations": len(visits_bad),
                        "original_action_not_max_visits": orig_not_max}

    # ---------------- activation ----------------
    def act(rows):
        c = Counter(r["reason"] for r in rows)
        return {"calls": len(rows), **{k: c.get(k, 0) for k in
                ("IMMEDIATE_WIN", "AVOID_IMMEDIATE_LOSS", "NO_SAFE_MOVE", "UNCHANGED")},
                "changed": sum(r["changed"] for r in rows), "zero_visit_selected": sum(r["zero"] for r in rows)}
    groups = {"all": guard_rows}
    for s in SEEDS:
        groups[f"seed {s}"] = [r for r in guard_rows if r["seed"] == s]
    for o in OPPS:
        groups[f"opponent {o}"] = [r for r in guard_rows if r["opp"] == o]
    for st in sorted(set(fam_stratum.values())):
        groups[f"stratum {st}"] = [r for r in guard_rows if r["stratum"] == st]
    for ph in ("opening", "middle", "late"):
        groups[f"phase {ph}"] = [r for r in guard_rows if phase(r["occ"]) == ph]
    activation = {k: act(v) for k, v in groups.items()}
    act_diff = {k: (v, analysis["activation"].get(k)) for k, v in activation.items() if analysis["activation"].get(k) != v}
    check("activation matches analysis", not act_diff, act_diff)
    changed_rows = [r for r in guard_rows if r["changed"]]
    changed_games = {(r["seed"], r["opp"], r["family"], r["color"]) for r in changed_rows}
    changed_fams = {r["family"] for r in changed_rows}
    changed = {"decisions": len(changed_rows), "games": len(changed_games), "families": len(changed_fams),
               "by_reason": dict(Counter(r["reason"] for r in changed_rows)),
               "rate": len(changed_rows) / len(guard_rows)}
    a_ch = analysis["changed"]
    check("changed matches analysis", (changed["decisions"], changed["games"], changed["families"]) ==
          (a_ch["decisions"], a_ch["games"], a_ch["families"]) and abs(changed["rate"] - a_ch["rate"]) < 1e-12)
    # analysis per-game change records vs ours
    rec_ours = defaultdict(list)
    for r in changed_rows:
        rec_ours[f"{r['seed']}-{r['opp']}:{r['family']}:{'XO'[r['color']]}"].append((r["ply"], r["original"], r["final"], r["reason"]))
    rec_theirs = {k: [(x["ply"], x["original"], x["final"], x["reason"]) for x in v]
                  for k, v in analysis["changed"].get("records", {}).items()}
    check("changed records match", dict(rec_ours) == rec_theirs)
    out["activation"] = activation
    out["changed"] = changed

    # ---------------- latency ----------------
    gs = np.array([r["gs"] for r in guard_rows])
    ratio = np.array([r["gs"] / r["ss"] for r in guard_rows])
    lat = {"guard_seconds": {"mean": float(gs.mean()), "median": float(np.median(gs)),
                             "p95": float(np.percentile(gs, 95)), "worst": float(gs.max())},
           "mean_guard_over_search": float(ratio.mean())}
    lat["pass"] = (lat["guard_seconds"]["p95"] <= design["latency"]["p95_guard_seconds_max"]
                   and lat["guard_seconds"]["worst"] <= design["latency"]["worst_guard_seconds_max"]
                   and lat["mean_guard_over_search"] <= design["latency"]["mean_relative_overhead_max"])
    al = analysis["latency"]
    lat["analysis_agrees"] = (abs(al["guard_seconds"]["mean"] - lat["guard_seconds"]["mean"]) < 1e-12
                              and al["guard_seconds"]["median"] == lat["guard_seconds"]["median"]
                              and abs(al["guard_seconds"]["p95"] - lat["guard_seconds"]["p95"]) < 1e-9
                              and al["guard_seconds"]["worst"] == lat["guard_seconds"]["worst"]
                              and abs(al["relative_overhead"]["mean"] - lat["mean_guard_over_search"]) < 1e-12
                              and al["guard_seconds"]["n"] == len(guard_rows))
    check("latency agrees", lat["analysis_agrees"])
    check("latency pass", lat["pass"])
    out["latency"] = lat

    # ---------------- paired fairness ----------------
    pair_bad = []
    for s in SEEDS:
        for o in OPPS:
            for f in families:
                for c in (0, 1):
                    fd = []
                    for a in ARMS:
                        ds = [d for d in games[(s, o, a, f, c)]["decisions"] if d["color"] == c]
                        fd.append((ds[0]["original_action"], ds[0]["visits"]) if ds else None)
                    if fd[0] is None or fd[0] != fd[1]:
                        pair_bad.append((s, o, f, c))
    check("paired first decision identical", not pair_bad, pair_bad[:5])
    identical_games = sum(1 for s in SEEDS for o in OPPS for f in families for c in (0, 1)
                          if games[(s, o, "control", f, c)]["moves"] == games[(s, o, "guarded", f, c)]["moves"])
    diverged = {(s, o, f, c) for s in SEEDS for o in OPPS for f in families for c in (0, 1)
                if games[(s, o, "control", f, c)]["moves"] != games[(s, o, "guarded", f, c)]["moves"]}
    check("divergence only where guard changed a move", diverged == changed_games,
          len(diverged ^ changed_games))
    out["paired_fairness"] = {"pairs": 768, "first_decision_mismatches": len(pair_bad),
                              "identical_move_sequences": identical_games,
                              "diverged_pairs_equal_changed_games": diverged == changed_games}

    # ---------------- scores ----------------
    fidx = {f: i for i, f in enumerate(families)}
    D = np.zeros((2, 2, 96, 2))  # seed, opp, family, color -> guarded - control
    for si, s in enumerate(SEEDS):
        for oi, o in enumerate(OPPS):
            for f in families:
                for c in (0, 1):
                    D[si, oi, fidx[f], c] = games[(s, o, "guarded", f, c)]["score"] - games[(s, o, "control", f, c)]["score"]
    strat = np.array([fam_stratum[f] for f in families])
    rng = np.random.Generator(np.random.PCG64(BOOT_SEED))
    idx_all = rng.integers(0, 96, size=(REPS, 96))
    strata_names = sorted(set(strat))
    idx_strat = {}
    for st in strata_names:
        members = np.where(strat == st)[0]
        idx_strat[st] = members[rng.integers(0, len(members), size=(REPS, len(members)))]

    def summarize(per_fam, members=None, idx=None):
        point = float(per_fam[members].mean() if members is not None else per_fam.mean())
        boot = per_fam[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"difference": point, "ci95": [float(lo), float(hi)]}

    def per_family(si=None, oi=None, ci=None):
        sel = D
        sel = sel[[si]] if si is not None else sel
        sel = sel[:, [oi]] if oi is not None else sel
        sel = sel[..., [ci]] if ci is not None else sel
        return sel.transpose(2, 0, 1, 3).reshape(96, -1).mean(axis=1)

    est = {"pooled": summarize(per_family(), idx=idx_all)}
    for si, s in enumerate(SEEDS):
        est[f"seed {s}"] = summarize(per_family(si=si), idx=idx_all)
    for oi, o in enumerate(OPPS):
        est[f"opponent {o}"] = summarize(per_family(oi=oi), idx=idx_all)
        for si, s in enumerate(SEEDS):
            est[f"cell {s}-{o}"] = summarize(per_family(si=si, oi=oi), idx=idx_all)
    for ci, cn in enumerate("XO"):
        est[f"color {cn}"] = summarize(per_family(ci=ci), idx=idx_all)
    for st in strata_names:
        m = np.where(strat == st)[0]
        est[f"stratum {st}"] = summarize(per_family(), m, idx_strat[st])
        for oi, o in enumerate(OPPS):
            est[f"stratum {st} {o}"] = summarize(per_family(oi=oi), m, idx_strat[st])
    b_minus_a = per_family(oi=1) - per_family(oi=0)
    est["opponent_b_minus_a"] = summarize(b_minus_a, idx=idx_all)
    est["seed_314159_minus_42"] = summarize(per_family(si=1) - per_family(si=0), idx=idx_all)

    def theirs(k):
        if k == "pooled":
            return analysis["pooled"]
        kind, _, rest = k.partition(" ")
        if kind == "seed":
            return analysis["seeds"][rest]
        if kind == "opponent":
            return analysis["opponents"][rest]
        if kind == "cell":
            return analysis["cells"][rest]
        if kind == "color":
            return analysis["colors"][rest]
        if kind == "stratum":
            parts = rest.split(" ")
            s = analysis["strata"][parts[0]]
            return s["by_opponent"][parts[1]] if len(parts) > 1 else s
        return analysis["heterogeneity"][k]
    comparison = {}
    point_bad = []
    for k, v in est.items():
        t = theirs(k)
        comparison[k] = {"ours": v, "analysis": {"difference": t["difference"], "ci95": t["ci95"]},
                         "point_equal": abs(v["difference"] - t["difference"]) < 1e-12,
                         "max_ci_endpoint_gap": max(abs(v["ci95"][0] - t["ci95"][0]), abs(v["ci95"][1] - t["ci95"][1]))}
        if not comparison[k]["point_equal"]:
            point_bad.append(k)
    check("point estimates equal analysis", not point_bad, point_bad)
    # arm scores
    arm_score = {a: float(np.mean([games[(s, o, a, f, c)]["score"] for s in SEEDS for o in OPPS
                                   for f in families for c in (0, 1)])) for a in ARMS}
    check("arm scores", abs(arm_score["control"] - analysis["pooled"]["control_score"]) < 1e-12
          and abs(arm_score["guarded"] - analysis["pooled"]["guarded_score"]) < 1e-12)
    nz = D != 0
    out["scores"] = {"arm_scores": arm_score, "estimates": comparison,
                     "paired_games_changed_outcome": int(nz.sum()),
                     "negative_paired_differences": int((D < 0).sum()),
                     "positive_paired_differences": int((D > 0).sum()),
                     "families_with_nonzero_difference": int(nz.any(axis=(0, 1, 3)).sum()),
                     "nonzero_pairs": [{"seed": SEEDS[a], "opponent": OPPS[b], "family": families[c],
                                        "color": "XO"[d], "difference": float(D[a, b, c, d])}
                                       for a, b, c, d in zip(*np.nonzero(nz))]}
    check("nonzero counts", int(nz.sum()) == analysis["paired_games_changed_outcome"]
          and out["scores"]["families_with_nonzero_difference"] == analysis["families_with_nonzero_pair_difference"])

    # ---------------- regression suites ----------------
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not k.endswith("seconds")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    suites = {}
    wdl_reg, tac_fail = [], []
    for s in SEEDS:
        for name in ("tactical", "solved", "deep"):
            new = load(RES / "tasks" / f"{s}-negamax3" / f"{name}.json")
            old = load(A_TASKS / str(s) / f"{name}.json")
            same = sum(1 for a, b in zip(new["rows"], old["rows"]) if strip(a) == strip(b))
            n = len(new["rows"])
            entry = {"rows": n, "identical_non_timing": same, "reference_rows": len(old["rows"])}
            if name == "tactical":
                entry["guarded_avoidable"] = sum(r["guarded"]["avoidable"] for r in new["rows"])
                entry["guarded_missed_win"] = sum(r["guarded"]["missed_win"] for r in new["rows"])
                entry["control_avoidable"] = sum(r["control"]["avoidable"] for r in new["rows"])
                entry["control_missed_win"] = sum(r["control"]["missed_win"] for r in new["rows"])
                if entry["guarded_avoidable"] or entry["guarded_missed_win"]:
                    tac_fail.append((s, name))
            else:
                regs = [r["id"] for r in new["rows"] if r["guarded"]["chosen_value"] < r["control"]["chosen_value"]]
                entry["guarded_below_control"] = len(regs)
                entry["guarded_above_control"] = sum(1 for r in new["rows"] if r["guarded"]["chosen_value"] > r["control"]["chosen_value"])
                wdl_reg += regs
            check(f"suite reproduction {s} {name}", same == n == len(old["rows"]) and new["complete"], entry)
            suites[f"{s}-{name}"] = entry
    check("no wdl regressions", not wdl_reg, wdl_reg)
    check("tactical clean", not tac_fail, tac_fail)
    out["regression_suites"] = suites
    reproduced = sum(v["identical_non_timing"] for v in suites.values())
    check("2512 searches", reproduced * 2 == design["regression_searches"] == 2512, reproduced)

    # ---------------- execution / git ----------------
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    is_anc = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PROSPECTIVE, "HEAD"]).returncode == 0
    tracked_results = git("ls-files", "research/alphazero_tactical_guard_confirmation/results/analysis.json",
                          "research/alphazero_tactical_guard_confirmation/results/tasks")
    pkg_diff = git("diff", "--name-only", PROSPECTIVE, "--", "research/alphazero_tactical_guard_confirmation",
                   "research/alphazero_tactical_guard")
    execu = {"status": execution["status"], "retries": execution["retries"], "tasks": execution["tasks"],
             "matches": execution["matches"], "prospective_commit": execution["prospective_commit"],
             "prospective_is_ancestor_of_HEAD": is_anc, "HEAD": git("rev-parse", "HEAD"),
             "results_tracked_in_git": bool(tracked_results), "tracked_files_changed_since_freeze": pkg_diff.splitlines(),
             "preservation_intact": preservation["intact"],
             "within_deadline": execution["finished_epoch"] <= design["compute_deadline_epoch"]}
    check("execution", execu["retries"] == 0 and execu["tasks"] == 4 and execu["matches"] == 1536
          and execu["prospective_commit"] == PROSPECTIVE and is_anc and not tracked_results
          and not pkg_diff and preservation["intact"] and execu["within_deadline"], execu)
    out["execution"] = execu

    # ---------------- verdict ----------------
    pooled = est["pooled"]
    opp_regs = [o for o in OPPS if est[f"opponent {o}"]["difference"] <= -MARGIN and est[f"opponent {o}"]["ci95"][1] < 0]
    fail = []
    if mism_rows:
        fail.append("guard decision mismatch")
    if tac_fail:
        fail.append("tactical guarded error")
    if pair_bad:
        fail.append("paired first decision")
    if wdl_reg:
        fail.append("WDL regression")
    if any(v["identical_non_timing"] != v["rows"] for v in suites.values()):
        fail.append("suite non-reproduction")
    if not lat["pass"]:
        fail.append("latency")
    integrity = [i["check"] for i in issues if i["check"] not in ("point estimates equal analysis",)]
    if integrity:
        fail.append("integrity: " + ", ".join(integrity))
    if pooled["difference"] <= -MARGIN and pooled["ci95"][1] < 0:
        fail.append("pooled material regression")
    if opp_regs:
        fail.append("opponent material regression")
    seeds_ok = all(est[f"seed {s}"]["difference"] > -MARGIN for s in SEEDS)
    if fail:
        strength = "FAIL"
    elif pooled["difference"] > 0 and pooled["ci95"][0] > 0 and seeds_ok and not opp_regs:
        strength = "CONFIRMED_POSITIVE_EFFECT"
    else:
        strength = "INCONCLUSIVE"
    signs = lambda ks: {np.sign(est[k]["difference"]) for k in ks}
    qualified = len(signs([f"seed {s}" for s in SEEDS])) > 1 or len(signs([f"opponent {o}" for o in OPPS])) > 1
    verdict = {"strength": strength, "failures": fail,
               "practically_meaningful": pooled["difference"] >= MARGIN,
               "small_effect": strength == "CONFIRMED_POSITIVE_EFFECT" and pooled["difference"] < MARGIN,
               "qualified_heterogeneity": qualified, "seeds_not_materially_negative": seeds_ok,
               "opponent_material_regressions": opp_regs}
    av = analysis["verdict"]
    verdict["matches_analysis"] = all(av[k] == verdict[k] for k in
                                      ("strength", "practically_meaningful", "small_effect",
                                       "qualified_heterogeneity", "seeds_not_materially_negative",
                                       "opponent_material_regressions")) and av["failures"] == []
    check("verdict matches analysis", verdict["matches_analysis"], av)
    out["verdict"] = verdict
    out["issues"] = issues
    out["all_checks_pass"] = not issues
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_checks_pass": not issues, "issues": issues, "verdict": verdict,
                      "pooled": pooled, "games": n_games, "guarded_decisions": len(guard_rows)}, indent=1, default=str))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
