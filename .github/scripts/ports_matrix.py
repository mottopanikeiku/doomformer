"""Combine the results.json files of several ports_run.py runs into one markdown matrix (rows: WADs, columns: ports).

    python ports_matrix.py DIR [DIR ...]      each DIR holds <port>/<set>/results.json for the sets in SUBS
"""
import json
import sys
from pathlib import Path

SUBS = ("probes", "same_tic", "hello", "hello_tags15", "machine_multiply", "machine_subleq_add",
        "machine_multiply_tags15", "machine_subleq_add_tags15", "machine_multiply_2_3", "machine_multiply_2_3_tags15",
        "museum_selftest", "museum_selftest_tags15")
PER_CASE = ("probes", "same_tic")       # one row per WAD; the other sets are one row each, named after the set


def load(root: Path) -> dict[str, dict[str, dict]]:
    ports = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = {}
        for sub in SUBS:
            f = d / sub / "results.json"
            if f.exists():
                for r in json.loads(f.read_text()):
                    rows[r["case"] if sub in PER_CASE else sub] = r
        if rows:
            ports[d.name] = rows
    return ports


def cell(r: dict | None) -> str:
    if r is None:
        return "-"
    if r["verdict"] == "PASS":
        return "PASS"
    if r["verdict"] == "INCOMPLETE":
        return f"INCOMPLETE ({r['lines_observed']} of {r['lines_expected']} lines printed, all equal, then the {int(r['seconds'])} s limit)"
    if not r["observed"]:
        return "ERROR " + (r["note"].replace("map refused: ", "refused: ") if r["note"] else r["status"])
    if r["values"] > 20:
        if "lines_observed" in r:
            return (f"DIFF ({r['lines_observed']} of {r['lines_expected']} lines printed, first difference at line "
                    f"{r['first_diff_line']})")
        return f"DIFF ({r['differing']} of {len(r['expected'].splitlines())} lines)"
    exp = dict(kv.split("=", 1) for kv in r["expected"].split()[2:])
    obs = dict(kv.split("=", 1) for kv in r["observed"].split()[2:])
    return "DIFF " + " ".join(f"{k}={obs.get(k)} (want {v})" for k, v in exp.items() if obs.get(k) != v)


def main(dirs: list[str]) -> None:
    ports = {}
    for d in dirs:
        ports.update(load(Path(d)))
    names = list(ports)
    cases = list(dict.fromkeys(c for rows in ports.values() for c in rows))
    print("| row | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for c in cases:
        print(f"| {c} | " + " | ".join(cell(ports[n].get(c)) for n in names) + " |")
    print()
    for n in names:
        v = [ports[n][c]["verdict"] for c in cases if c in ports[n]]
        print(f"* {n}: {v.count('PASS')} PASS, {v.count('DIFF')} DIFF, {v.count('ERROR')} ERROR"
              + (f", {v.count('INCOMPLETE')} INCOMPLETE" if 'INCOMPLETE' in v else "") + f" of {len(v)}")


if __name__ == "__main__":
    main(sys.argv[1:])
