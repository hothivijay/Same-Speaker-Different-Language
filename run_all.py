#!/usr/bin/env python3
"""
One command to reproduce every experiment on DISPLACE 2024.

    python run_all.py --displace /path/to/DISPLACE --out /path/to/output

--displace  the folder where you unpacked the DISPLACE 2024 release (any layout:
            the script finds the audio and the RTTM label files by itself)
--out       where logs, embeddings, RTTMs and the result summary are written

Steps (each one is skipped if its output already exists, so a crashed or
disconnected run can simply be started again):
  1. organise   find dev/eval audio, speaker RTTMs (Track 1) and language RTTMs (Track 2)
  2. embed      TitaNet-Large embeddings, 1.5 s windows, 0.75 s hop (GPU recommended)
  3. similarity H1: language drift of same-speaker similarity + language probe
  4. diarize    H2 + H3: clustering with/without language-subspace projection (LSP),
                controls, second back end, statistics
  5. (optional) the same with 3 s windows  (--window-check)
  6. collect    results.json + RESULTS.md with the headline numbers

All analysis code is in cs_diarization.py; this file only finds the data and
calls it in the right order with the settings used in the paper.
"""
import argparse
import collections
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "cs_diarization.py")
COLLECT = os.path.join(HERE, "tools", "collect_results.py")

SKIP = ("sample", "checkpoint", "__macosx")            # folders that are not part of the data
LD_KEYS = ("track2", "track_2", "track-2", "/ld", "_ld", "lang")
SD_KEYS = ("track1", "track_1", "track-1", "/sd", "_sd", "speaker")
AUDIO_EXT = (".wav", ".flac")


# ----------------------------------------------------------------- helpers
def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def relpath_low(p, root):
    return "/" + os.path.relpath(p, root).lower().replace("\\", "/")


def skipped(p, root):
    return any(t in relpath_low(p, root) for t in SKIP)


def rttm_info(p):
    """Types (SPEAKER/LANGUAGE), labels and file ids found in one RTTM."""
    types, labels, fids = set(), set(), set()
    with open(p, errors="ignore") as f:
        for line in f:
            q = line.split()
            if len(q) >= 8:
                types.add(q[0])
                labels.add(q[7])
                fids.add(q[1])
    return types, labels, fids


def classify(p, root):
    """('dev'|'eval'|None, 'sd'|'ld') from the RTTM line type and the folder names."""
    low = relpath_low(p, root)
    types, _, _ = rttm_info(p)
    if "LANGUAGE" in types or any(k in low for k in LD_KEYS):
        kind = "ld"
    elif any(k in low for k in SD_KEYS) or "SPEAKER" in types:
        kind = "sd"
    else:
        kind = None
    part = "eval" if "eval" in low else ("dev" if "dev" in low else None)
    return part, kind


def run(cmd, log):
    """Run one step, show its output live and keep a copy in `log`."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore")
    print("$ " + " ".join(cmd), flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
    if rc:
        raise SystemExit(f"\nstep failed (exit code {rc}); see {log}")


# ----------------------------------------------------------------- step 1
def organise(root, out, work, copy_audio):
    say(f"1/6 organise: looking for RTTM and audio files under {root}")
    if not os.path.isdir(root):
        raise SystemExit(f"DISPLACE folder not found: {root}")
    lab = os.path.join(out, "labels")
    counts, unknown, ids = collections.Counter(), [], collections.defaultdict(set)
    for p in sorted(glob.glob(os.path.join(root, "**", "*.rttm"), recursive=True)):
        if skipped(p, root):
            continue
        part, kind = classify(p, root)
        if part is None or kind is None:
            unknown.append(p)
            continue
        dst = os.path.join(lab, f"{part}_{kind}")
        os.makedirs(dst, exist_ok=True)
        shutil.copy(p, dst)
        counts[(part, kind)] += 1
        ids[part] |= rttm_info(p)[2]
    for key in sorted(counts):
        print(f"  {key[0]:4s} {'speaker ' if key[1] == 'sd' else 'language'} RTTMs: {counts[key]}")
    if unknown:
        print(f"  {len(unknown)} RTTM file(s) not used (no 'dev'/'eval' in the path), e.g. {unknown[:2]}")
    if not counts[("dev", "sd")]:
        raise SystemExit("no development speaker RTTMs found: check --displace")

    if not counts[("eval", "sd")]:
        # no evaluation labels in the package: hold out every second dev recording
        print("  no evaluation labels found: every second development recording is held out as 'eval'")
        fids = sorted(ids["dev"])
        held = set(fids[1::2])
        for kind in ("sd", "ld"):
            src, dst = os.path.join(lab, f"dev_{kind}"), os.path.join(lab, f"eval_{kind}")
            os.makedirs(dst, exist_ok=True)
            for f in glob.glob(os.path.join(src, "*.rttm")):
                if rttm_info(f)[2] & held:
                    shutil.move(f, os.path.join(dst, os.path.basename(f)))

    # audio: one clean folder of <session id>.wav entries (links, or copies with --copy-audio)
    audio_dir = os.path.join(work, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    found = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.splitext(p)[1].lower() in AUDIO_EXT and not skipped(p, root):
            found.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    needed = ids["dev"] | ids["eval"]
    missing = sorted(needed - set(found))
    made = 0
    for sid in sorted(needed & set(found)):
        src = found[sid]
        dst = os.path.join(audio_dir, sid + os.path.splitext(src)[1].lower())
        if os.path.exists(dst):
            made += 1
            continue
        try:
            if copy_audio:
                raise OSError
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy(src, dst)
        made += 1
    print(f"  audio: {made} of {len(needed)} recordings available in {audio_dir}"
          + (f"; MISSING: {missing[:5]}" if missing else ""))
    has_ld = counts[("dev", "ld")] > 0 and counts[("eval", "ld")] > 0 or (counts[("dev", "ld")] > 0 and not counts[("eval", "sd")])
    return lab, audio_dir, has_ld


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--displace", required=True, help="folder of the unpacked DISPLACE 2024 release")
    ap.add_argument("--out", default="ncc_output", help="output folder (logs, results, embeddings)")
    ap.add_argument("--work", help="fast scratch folder for the audio links/copies (default: --out)")
    ap.add_argument("--copy-audio", action="store_true", help="copy audio instead of linking it")
    ap.add_argument("--stride", type=int, default=2, help="cluster every n-th window (2 = setting used in the paper)")
    ap.add_argument("--window-check", action="store_true", help="also run the 3 s window experiment (about 30 min more)")
    ap.add_argument("--pseudo-lid", action="store_true", help="make language labels with a language-ID model if Track-2 RTTMs are missing")
    ap.add_argument("--force", action="store_true", help="recompute steps even if their output exists")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    work = os.path.abspath(a.work or a.out)
    logs, res = os.path.join(out, "logs"), os.path.join(out, "results")
    for d in (out, work, logs, res):
        os.makedirs(d, exist_ok=True)
    py = sys.executable
    t0 = time.time()

    lab, audio, has_ld = organise(os.path.abspath(a.displace), out, work, a.copy_audio)
    L = {f"{p}_{k}": os.path.join(lab, f"{p}_{k}") for p in ("dev", "eval") for k in ("sd", "ld")}
    if not has_ld:
        if not a.pseudo_lid:
            raise SystemExit("no Track-2 language RTTMs found. Re-run with --pseudo-lid to create them with a "
                             "language-ID model (results then use automatic language labels).")
        for part in ("dev", "eval"):
            say(f"pseudo language labels for {part}")
            run([py, SCRIPT, "lid", "--audio-dir", audio, "--sd-ref", L[f"{part}_sd"], "--out", L[f"{part}_ld"]],
                os.path.join(logs, f"lid_{part}.log"))

    def embed(part, suffix="", extra=()):
        npz = os.path.join(res, f"{part}{suffix}.npz")
        if os.path.exists(npz) and not a.force:
            print(f"  {npz} exists: skipped")
            return npz
        run([py, SCRIPT, "embed", "--audio-dir", audio, "--sd-ref", L[f"{part}_sd"], "--ld-ref", L[f"{part}_ld"],
             "--out", npz, *extra], os.path.join(logs, f"embed_{part}{suffix}.log"))
        return npz

    say("2/6 embed: TitaNet-Large, 1.5 s windows, 0.75 s hop")
    dev_npz, eval_npz = embed("dev"), embed("eval")

    say("3/6 similarity (H1)")
    for part, npz in (("dev", dev_npz), ("eval", eval_npz)):
        run([py, SCRIPT, "similarity", "--emb", npz], os.path.join(logs, f"similarity_{part}.log"))

    say("4/6 diarize (H2, H3, controls, AHC, statistics)")
    diar = [py, SCRIPT, "diarize", "--dev-emb", dev_npz, "--eval-emb", eval_npz,
            "--dev-sd-ref", L["dev_sd"], "--eval-sd-ref", L["eval_sd"],
            "--dev-ld-ref", L["dev_ld"], "--eval-ld-ref", L["eval_ld"], "--stride", str(a.stride)]
    run(diar + ["--out", os.path.join(res, "runs")], os.path.join(logs, "diarize.log"))

    if a.window_check:
        say("5/6 window check: 3 s windows, 1.5 s hop")
        d3 = embed("dev", "_w3", ("--win", "3.0", "--hop", "1.5"))
        e3 = embed("eval", "_w3", ("--win", "3.0", "--hop", "1.5"))
        run([py, SCRIPT, "diarize", "--dev-emb", d3, "--eval-emb", e3,
             "--dev-sd-ref", L["dev_sd"], "--eval-sd-ref", L["eval_sd"],
             "--dev-ld-ref", L["dev_ld"], "--eval-ld-ref", L["eval_ld"],
             "--hop", "1.5", "--no-controls", "--no-ahc", "--no-probe", "--out", os.path.join(res, "runs_w3")],
            os.path.join(logs, "diarize_w3.log"))
    else:
        say("5/6 window check skipped (add --window-check to run it)")

    say("6/6 collect results")
    run([py, COLLECT, logs, res], os.path.join(logs, "collect.log"))
    say(f"done in {(time.time() - t0) / 60:.1f} min. Summary: {os.path.join(res, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
