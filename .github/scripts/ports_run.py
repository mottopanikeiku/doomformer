"""Run exported doomformer fact WADs in another Doom engine and diff the FACT lines against expected.txt.

    python ports_run.py --name gzdoom --wads out --iwad freedoom2.wad --out logs/gzdoom --cmd "gzdoom -iwad {iwad} ..."

The command template is split with shlex; {iwad}, {wad}, {work} are replaced per WAD. Each WAD runs under its own
xvfb display with a timeout; stdout and stderr go to <out>/<case>.log. Writes <out>/observed.txt, <out>/results.json
and a markdown table (row, expected, observed, PASS/DIFF/ERROR) to <out>/summary.md and, if set, $GITHUB_STEP_SUMMARY.
"""
import argparse
import concurrent.futures as cf
import itertools
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

TAG = "FACT "
DISPLAYS = itertools.count(100)          # one X server number per run: xvfb-run -a races between parallel runs


def run_one(case, a, wads, out, expected):
    wad = (wads / f"{case}.wad").resolve()
    work = (out / "work" / case).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cmd = [w.format(iwad=Path(a.iwad).resolve(), wad=wad, work=work) for w in shlex.split(a.cmd)]
    if a.xvfb:
        cmd = ["xvfb-run", "-n", str(next(DISPLAYS)), "-s", "-screen 0 800x600x24"] + cmd
    env = dict(os.environ, HOME=str(work), LIBGL_ALWAYS_SOFTWARE="1", SDL_AUDIODRIVER="dummy", ALSOFT_DRIVERS="null", OMP_NUM_THREADS="1")
    for kv in a.env:
        k, v = kv.split("=", 1)
        env[k] = v
    t0 = time.time()
    status = "exit"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=work,
                         start_new_session=True)
    chunks, got = [], []
    done, refused = threading.Event(), threading.Event()

    def pump():
        # Some ports ignore ConsoleCommand("quit") from ACS: end the run one second after the case's FACT line.
        for raw in iter(p.stdout.readline, b""):
            chunks.append(raw)
            ln = raw.decode("utf-8", "replace").strip()
            if ln.startswith(TAG) and ln.split()[1:2] == [case]:
                got.append(ln)
                if len(set(got)) >= len(expected[case]):
                    done.set()
            elif "out of range for a coordinate" in ln:        # GZDoom refuses the map: nothing more will happen
                refused.set()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    status = "exit"
    t_end = time.time() + a.timeout
    while p.poll() is None and time.time() < t_end:
        if refused.is_set():
            time.sleep(1.0)
            status = "killed after the map was refused"
            os.killpg(p.pid, signal.SIGKILL)
            break
        if done.wait(0.5):
            time.sleep(1.0)
            if p.poll() is None:
                status = "killed after FACT line"
                os.killpg(p.pid, signal.SIGKILL)
            break
    if p.poll() is None and status == "exit":
        status = "timeout"
        os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    th.join(5)
    text = b"".join(chunks)
    text = text.decode("utf-8", "replace")
    extra = work / "log.txt"
    if extra.exists():
        text += "\n--- log.txt ---\n" + extra.read_text(errors="replace")
    (out / f"{case}.log").write_text(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(TAG)]
    seen = list(dict.fromkeys(ln for ln in lines if ln.split()[1:2] == [case]))      # the logfile repeats stdout
    want = expected[case]
    note = ""
    if not seen:
        verdict = "ERROR"
        refused = [ln for ln in text.splitlines() if "out of range for a coordinate" in ln]
        note = f"map refused: {refused[0].strip()}" if refused else ""
    else:
        verdict = "PASS" if seen == want else "DIFF"
    obs = "\n".join(seen)
    expected_text = "\n".join(want)
    return dict(case=case, expected=expected_text, observed=obs, verdict=verdict, status=status, note=note,
                values=sum(len(w.split()) - 2 for w in want),
                differing=sum(a != b for a, b in zip(seen, want)) + abs(len(seen) - len(want)),
                returncode=p.returncode, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--wads", required=True)
    ap.add_argument("--iwad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the engine process")
    ap.add_argument("--no-xvfb", dest="xvfb", action="store_false")
    a = ap.parse_args()
    wads, out = Path(a.wads), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for ln in (wads / "expected.txt").read_text().splitlines():
        if ln.strip():
            expected.setdefault(ln.split()[1], []).append(ln.strip())
    cases = [c for c in expected if not a.only or c in a.only]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda c: run_one(c, a, wads, out, expected), cases))
    (out / "observed.txt").write_text("".join(r["observed"] + "\n" for r in rows if r["observed"]))
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    n = {v: sum(r["verdict"] == v for r in rows) for v in ("PASS", "DIFF", "ERROR")}
    md = [f"### {a.name}: {n['PASS']} PASS, {n['DIFF']} DIFF, {n['ERROR']} ERROR of {len(rows)} WADs", "",
          "| row | expected | observed | result | s |", "|---|---|---|---|---|"]
    for r in rows:
        if len(r["expected"]) > 160:                         # hello: many lines of many values
            exp = f"{r['values']} values in {len(r['expected'].splitlines())} lines (expected.txt)"
            obs = (f"{r['values']} values, byte for byte equal" if r["verdict"] == "PASS" else
                   f"{r['differing']} lines differ (see {r['case']}.log)")
        else:
            exp = r["expected"].removeprefix(TAG)
            obs = r["observed"].removeprefix(TAG)
        if not r["observed"]:
            obs = f"(no FACT line; {r['status']}, rc {r['returncode']}" + (f"; {r['note']}" if r["note"] else "") + ")"
        md.append(f"| {r['case']} | `{exp}` | `{obs}` | {r['verdict']} | {r['seconds']} |")
    text = "\n".join(md) + "\n"
    (out / "summary.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
