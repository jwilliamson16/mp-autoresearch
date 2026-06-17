"""progress.py — monitor the autoresearch loop at a glance.

Shows: current git state, results history, best run, and whether
an experiment is actively running.

Usage
-----
    python progress.py          # one-shot summary
    python progress.py --watch  # refresh every 30 s until Ctrl+C
    python progress.py --stop   # create STOP sentinel (orderly shutdown)
"""

import argparse
import os
import subprocess
import sys
import time

RESULTS_TSV = os.path.join(os.path.dirname(__file__), "results.tsv")
RUN_LOG     = os.path.join(os.path.dirname(__file__), "run.log")
STOP_FILE   = os.path.join(os.path.dirname(__file__), "STOP")
PROJECT_DIR = os.path.dirname(__file__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _git(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "(unknown)"


def _tail(path: str, n: int = 15) -> str:
    if not os.path.exists(path):
        return "  (no file)"
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError:
        return "  (unreadable)"


def _mtime_age(path: str) -> str:
    if not os.path.exists(path):
        return "–"
    age = time.time() - os.path.getmtime(path)
    if age < 60:
        return f"{age:.0f}s ago"
    if age < 3600:
        return f"{age/60:.0f}m ago"
    return f"{age/3600:.1f}h ago"


def _read_results() -> list[dict]:
    if not os.path.exists(RESULTS_TSV):
        return []
    rows = []
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return []
    header = lines[0].strip().split("\t")
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) < len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


# ── display ───────────────────────────────────────────────────────────────────

def _clear():
    os.system("clear" if os.name != "nt" else "cls")


def show(clear_screen: bool = False) -> None:
    if clear_screen:
        _clear()

    # ── git state ────────────────────────────────────────────────────────────
    branch  = _git("git rev-parse --abbrev-ref HEAD")
    commit  = _git("git log -1 --oneline")
    n_ahead = _git(f"git rev-list --count origin/main..{branch} 2>/dev/null || echo 0")

    print("=" * 62)
    print("  MP Autoresearch — progress monitor")
    print("=" * 62)
    print(f"  Branch : {branch}  ({n_ahead} commits ahead of main)")
    print(f"  Latest : {commit}")

    # ── active experiment ─────────────────────────────────────────────────────
    log_age = _mtime_age(RUN_LOG)
    stop_pending = os.path.exists(STOP_FILE)
    print(f"  run.log: last modified {log_age}"
          + ("  ← STOP requested" if stop_pending else ""))

    # ── results table ────────────────────────────────────────────────────────
    rows = _read_results()
    print()
    if not rows:
        print("  results.tsv: no experiments logged yet.")
    else:
        kept      = [r for r in rows if r.get("status") == "keep"]
        discarded = [r for r in rows if r.get("status") == "discard"]
        crashed   = [r for r in rows if r.get("status") == "crash"]
        print(f"  Experiments: {len(rows)} total  "
              f"({len(kept)} keep  {len(discarded)} discard  {len(crashed)} crash)")
        print()

        # Table header
        w = 8
        print(f"  {'#':<4} {'commit':<8} {'train_rec':>{w}} {'test_rec':>{w}} "
              f"{'test_fdr':>{w}} {'test_f1':>{w}}  {'status':<8}  description")
        print("  " + "-" * 78)

        best_test_rec = -1.0
        best_row_idx  = -1
        for i, r in enumerate(rows):
            try:
                tr = float(r.get("test_monomer_recall", 0))
                if tr > best_test_rec:
                    best_test_rec = tr
                    best_row_idx  = i
            except ValueError:
                pass

        for i, r in enumerate(rows):
            marker = "◄ best" if i == best_row_idx else ""
            status = r.get("status", "?")
            col    = ""  # could add ANSI color here if desired
            try:
                tr_rec = f"{float(r.get('train_monomer_recall','0')):.4f}"
                te_rec = f"{float(r.get('test_monomer_recall','0')):.4f}"
                te_fdr = f"{float(r.get('test_fdr','0')):.4f}"
                te_f1  = f"{float(r.get('test_macro_f1','0')):.4f}"
            except ValueError:
                tr_rec = te_rec = te_fdr = te_f1 = "  ?"
            desc = r.get("description", "")[:30]
            print(f"  {i+1:<4} {r.get('commit','?'):<8} {tr_rec:>{w}} {te_rec:>{w}} "
                  f"{te_fdr:>{w}} {te_f1:>{w}}  {status:<8}  {desc}  {marker}")

        if best_row_idx >= 0:
            best = rows[best_row_idx]
            print()
            print(f"  Best so far: commit {best.get('commit','?')}  "
                  f"test_monomer_recall={best.get('test_monomer_recall','?')}  "
                  f"test_fdr={best.get('test_fdr','?')}")

    # ── tail of run.log ───────────────────────────────────────────────────────
    print()
    print("  ── last 12 lines of run.log " + "─" * 34)
    for line in _tail(RUN_LOG, 12).splitlines():
        print(f"  {line}")
    print("=" * 62)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor the autoresearch loop")
    parser.add_argument("--watch",    action="store_true",
                        help="Refresh every 30 s (Ctrl+C to quit)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Refresh interval in seconds (default: 30)")
    parser.add_argument("--stop",     action="store_true",
                        help="Request orderly shutdown (creates STOP file)")
    args = parser.parse_args()

    if args.stop:
        with open(STOP_FILE, "w") as f:
            f.write("stop requested\n")
        print(f"STOP file created at {STOP_FILE}")
        print("The loop will finish the current experiment, then exit cleanly.")
        return

    if args.watch:
        try:
            while True:
                show(clear_screen=True)
                print(f"\n  [watching — refreshes every {args.interval}s — Ctrl+C to quit]")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        show(clear_screen=False)


if __name__ == "__main__":
    main()
