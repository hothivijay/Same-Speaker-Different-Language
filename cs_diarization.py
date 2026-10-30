#!/usr/bin/env python3
"""
Code-switching study for speaker diarization on DISPLACE 2024 (NCC 2027 paper).

Question: does a change of language look like a change of speaker to a speaker
embedding, and does removing the language direction from the embeddings help
diarization of code-switched conversation?

Subcommands
  embed          TitaNet-Large embeddings on sliding windows, labelled with the
                 reference speaker and language of each window.
  lid            Pseudo language labels with a language-ID model, only if
                 Track-2 language references are missing.
  similarity     H1: same-speaker cosine similarity within vs across languages
                 (exact over all non-overlapping window pairs, also time-matched),
                 and a leave-one-speaker-out language probe inside each recording
                 (DISPLACE language labels L1, L2, ... are relative to a recording).
  switch-errors  H2: where do the errors of a diarization output fall? Four
                 conditions: same/different speaker x same/different language.
  diarize        Clustering-only diarization with ORACLE speech regions, with and
                 without language-subspace projection (LSP). Chooses the rank on
                 dev only, then on eval reports: DER, confusion per condition,
                 split rate (with a shifted-label control), cluster-change rates,
                 speaker-count error, paired statistics, control projections
                 (PCA, within-speaker NAP, shifted-label LSP, random), speaker-
                 information checks (EER, pair cosines, language probe), a second
                 back end (AHC), subspace stability and runtime.

Inputs (DISPLACE 2024)
  --sd-ref  directory of Track-1 speaker RTTMs   (label field = speaker)
  --ld-ref  directory of Track-2 language RTTMs  (label field = language)

Scoring: DER with no collar and overlap scored. Region-wise errors use a 10 ms
frame implementation of the DER definition with ONE optimal (Hungarian) speaker
mapping per recording; it is checked against pyannote.metrics on every file.
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*'uem' was approximated.*")
STEP = 0.01  # 10 ms frames for label bookkeeping and region scoring
TIME_BINS = [(0.0, 10.0), (10.0, 60.0), (60.0, np.inf)]  # pair distance bins (s), lower edge raised to L


# --------------------------------------------------------------------- RTTM I/O
def read_rttm_dir(d):
    """Return {file_id: [(start, end, label), ...]} sorted by start."""
    out = defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(d, "*.rttm"))):
        with open(fp) as f:
            for line in f:
                p = line.split()
                if len(p) < 8 or p[0] not in ("SPEAKER", "LANGUAGE"):  # Track-2 RTTMs use LANGUAGE
                    continue
                if p[0] == "LANGUAGE" and p[7].startswith("?"):
                    continue  # DISPLACE marks unclear or non-speech language segments as "?:..."
                s, dur = float(p[3]), float(p[4])
                if dur > 0:
                    out[p[1]].append((s, s + dur, p[7]))
    for k in out:
        out[k].sort()
    return dict(out)


def write_rttm(path, file_id, segs):
    with open(path, "w") as f:
        for s, e, lab in segs:
            f.write(f"SPEAKER {file_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n")


def to_annotation(segs, uri, prefix=""):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    for i, (s, e, lab) in enumerate(segs):
        if e > s:
            ann[Segment(s, e), i] = prefix + str(lab)
    return ann


def file_end(*seg_lists):
    return max((e for segs in seg_lists for _, e, _ in segs), default=0.0)


# ------------------------------------------------------------- frame helpers
def activity(segs, labels, n):
    """Boolean activity matrix (len(labels) x n frames)."""
    idx = {l: i for i, l in enumerate(labels)}
    act = np.zeros((max(len(labels), 1), n), bool)
    for s, e, l in segs:
        if l not in idx:
            continue
        a, b = min(n, max(0, int(round(s / STEP)))), min(n, max(0, int(round(e / STEP))))
        if b > a:
            act[idx[l], a:b] = True
    return act[:len(labels)] if labels else act[:0]


def mask_to_segments(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a * STEP, b * STEP) for a, b in zip(st, en)]


def points_mask(points, n, radius):
    R = int(round(radius / STEP))
    m = np.zeros(n, bool)
    for t in points:
        i = int(round(t / STEP))
        m[max(0, i - R):min(n, i + R)] = True
    return m


# ------------------------------------------------------------ language switches
def switch_points(lang_segs, max_gap=2.0):
    """Midpoints between consecutive language segments whose labels differ (gap <= max_gap)."""
    pts = []
    for (s1, e1, l1), (s2, e2, l2) in zip(lang_segs, lang_segs[1:]):
        if l1 != l2 and s2 - e1 <= max_gap:
            pts.append(0.5 * (e1 + s2))
    return pts


def switch_events(single, ld_segs, radius, max_gap, min_cov=0.25):
    """Classify reference events.

    ws  : language switch with the SAME speaker before and after (within-speaker switch)
    sc  : language switch that coincides with a speaker change
    amb : language switch with no clear single speaker on one side (silence/overlap)
    spk : speaker change with no language switch within the radius
    The speaker on each side is the one holding most single-speaker frames in the
    `radius` seconds before / after the point (at least min_cov of that span).
    """
    n = len(single)
    R = int(round(radius / STEP))

    def dominant(a, b):
        a, b = max(0, a), min(n, b)
        v = single[a:b]
        v = v[v >= 0]
        if b <= a or len(v) < min_cov * R:
            return None
        return int(np.bincount(v).argmax())

    ev = {"ws": [], "sc": [], "amb": [], "spk": []}
    for t in switch_points(ld_segs, max_gap):
        i = int(round(t / STEP))
        pre, post = dominant(i - R, i), dominant(i, i + R)
        ev["amb" if pre is None or post is None else ("ws" if pre == post else "sc")].append(t)
    idx = np.flatnonzero(single >= 0)
    if len(idx) > 1:
        lab = single[idx]
        for c in np.flatnonzero(lab[1:] != lab[:-1]):
            a, b = idx[c], idx[c + 1]
            if (b - a) * STEP <= max_gap:
                t = 0.5 * (a + b) * STEP
                if not any(abs(t - u) <= radius for u in ev["sc"]):  # that change IS the sc switch
                    ev["spk"].append(t)
    return ev


def condition_masks(ev, n, radius):
    """Frame masks of the four conditions; an event is 'pure' if no other event lies within radius."""
    allpts = [(t, k) for k in ("ws", "sc", "spk", "amb") for t in ev[k]]
    ts = np.array([t for t, _ in allpts]) if allpts else np.zeros(0)
    pure, mixed = defaultdict(list), []
    for t, k in allpts:
        alone = np.sum(np.abs(ts - t) <= radius) == 1
        (pure[k] if alone and k != "amb" else mixed).append(t)
    M = {k: points_mask(pure[k], n, radius) for k in ("ws", "sc", "spk")}
    Mx = points_mask(mixed, n, radius)
    cnt = M["ws"].astype(np.int8) + M["sc"] + M["spk"] + Mx
    return {"SS-SL": cnt == 0, "SS-DL": M["ws"] & (cnt == 1),
            "DS-SL": M["spk"] & (cnt == 1), "DS-DL": M["sc"] & (cnt == 1)}, \
        {k: len(pure[k]) for k in ("ws", "sc", "spk")}


CONDITIONS = [("SS-SL", "same speaker, same language (away from any change)"),
              ("SS-DL", "same speaker, language switch"),
              ("DS-SL", "speaker change, same language"),
              ("DS-DL", "speaker change with language switch")]


class FrameRef:
    """Reference of one recording on a 10 ms grid, with switch events and condition masks."""

    def __init__(self, sd_segs, ld_segs, radius=1.0, max_gap=2.0, end=None):
        end = end if end is not None else file_end(sd_segs, ld_segs or [])
        self.n = n = int(np.ceil(end / STEP)) + 1
        self.hours = end / 3600
        self.labels = sorted({l for _, _, l in sd_segs})
        self.act = activity(sd_segs, self.labels, n)
        self.nref = self.act.sum(0).astype(np.int16)
        self.speech = self.nref > 0
        self.single = np.where(self.nref == 1, self.act.argmax(0), -1).astype(np.int16)
        self.masks, self.ev, self.pure, self.lang_masks = {}, None, {}, {}
        if ld_segs:
            self.ev = switch_events(self.single, ld_segs, radius, max_gap)
            self.masks, self.pure = condition_masks(self.ev, n, radius)
            allsw = self.ev["ws"] + self.ev["sc"] + self.ev["amb"]
            near = points_mask(allsw, n, radius)
            self.masks["near-all"], self.masks["far-all"] = near, ~near
            for lang in sorted({l for _, _, l in ld_segs}):
                self.lang_masks[lang] = activity([(s, e, "x") for s, e, l in ld_segs if l == lang], ["x"], n)[0]

    def score(self, hyp):
        """Miss, FA, confusion, reference speech (s) overall, per condition and per language.

        `hyp` is either a label vector (one int per frame, -1 = no speech; single-label output)
        or a boolean activity matrix (labels x frames) for systems that output overlap.
        One optimal one-to-one speaker mapping (Hungarian) is computed for the whole recording.
        """
        from scipy.optimize import linear_sum_assignment
        n, R = self.n, len(self.act)
        if hyp.ndim == 1:
            lab = np.full(n, -1, int)
            lab[:min(n, len(hyp))] = hyp[:n]
            H = int(lab.max()) + 1 if lab.max() >= 0 else 0
            C = np.stack([np.bincount(lab[self.act[i] & (lab >= 0)], minlength=H) for i in range(R)]) if R and H \
                else np.zeros((R, H))
            nh = (lab >= 0).astype(np.int16)
        else:
            hyp = hyp[:, :n]
            if hyp.shape[1] < n:
                hyp = np.pad(hyp, ((0, 0), (0, n - hyp.shape[1])))
            H = len(hyp)
            C = np.array([[np.count_nonzero(self.act[i] & hyp[j]) for j in range(H)] for i in range(R)]).reshape(R, H)
            nh = hyp.sum(0).astype(np.int16)
        corr = np.zeros(n, np.int16)
        if R and H:
            for i, j in zip(*linear_sum_assignment(-C)):
                if C[i, j] > 0:
                    corr += (self.act[i] & (lab == j)) if hyp.ndim == 1 else (self.act[i] & hyp[j])
        nr = self.nref
        miss, fa = np.maximum(0, nr - nh), np.maximum(0, nh - nr)
        conf = np.minimum(nr, nh) - corr

        def comp(m):
            return np.array([miss[m].sum(), fa[m].sum(), conf[m].sum(), nr[m].sum()], float) * STEP

        out = {"all": comp(np.ones(n, bool))}
        out.update({k: comp(m) for k, m in self.masks.items()})
        out.update({"lang:" + k: comp(m) for k, m in self.lang_masks.items()})
        return out


def build_refs(sd, ld, radius, max_gap):
    return {f: FrameRef(sd[f], ld.get(f, []), radius, max_gap) for f in sd}


def rate(c, i=2):
    """Error component i (0 miss, 1 FA, 2 confusion) or DER (i=None) in % of reference speech."""
    if c[3] <= 0:
        return np.nan
    return 100 * (c[:3].sum() if i is None else c[i]) / c[3]


def fmt(comp):
    miss, fa, conf, tot = comp
    if tot <= 0:
        return "no reference speech"
    return (f"ref {tot / 3600:6.2f} h | miss {100 * miss / tot:5.2f} | FA {100 * fa / tot:5.2f} | "
            f"conf {100 * conf / tot:5.2f} | DER {100 * (miss + fa + conf) / tot:5.2f}")


def pyannote_der(sd_segs, hyp_segs, uri):
    from pyannote.metrics.diarization import DiarizationErrorRate
    # .support() merges overlapping segments of the same speaker, as the frame scorer does
    c = DiarizationErrorRate(collar=0.0, skip_overlap=False)(
        to_annotation(sd_segs, uri).support(), to_annotation(hyp_segs, uri, prefix="H:").support(), detailed=True)
    return 100 * (c["missed detection"] + c["false alarm"] + c["confusion"]) / max(c["total"], 1e-9)


def report_events(refs, title):
    tot = defaultdict(float)
    hours = sum(r.hours for r in refs.values() if r.ev is not None)
    for r in refs.values():
        if r.ev is None:
            continue
        for k, v in r.ev.items():
            tot[k] += len(v)
        for k, v in r.pure.items():
            tot["pure_" + k] += v
    if hours <= 0:
        print(f"[{title}] no language references: switch analysis skipped")
        return {}
    print(f"[{title}] {hours:.2f} h of audio")
    print(f"  within-speaker language switches : {tot['ws']:6.0f}  ({tot['ws'] / hours:6.1f} per hour; "
          f"{tot['pure_ws']:.0f} with no other change within the radius)")
    print(f"  switches with a speaker change   : {tot['sc']:6.0f}  ({tot['sc'] / hours:6.1f} per hour)")
    print(f"  switches without a clear speaker : {tot['amb']:6.0f}  ({tot['amb'] / hours:6.1f} per hour)")
    print(f"  speaker changes, same language   : {tot['spk']:6.0f}  ({tot['spk'] / hours:6.1f} per hour)")
    return {k: v / hours for k, v in tot.items()}


def cmd_switch_errors(a):
    sd, ld, hy = read_rttm_dir(a.sd_ref), read_rttm_dir(a.ld_ref), read_rttm_dir(a.hyp)
    files = sorted(set(sd) & set(ld) & set(hy))
    print(f"{len(files)} files with SD reference, LD reference and hypothesis")
    refs = {f: FrameRef(sd[f], ld[f], a.radius, a.max_gap, end=file_end(sd[f], ld[f], hy[f])) for f in files}
    report_events(refs, "events")
    tot, per_sess = defaultdict(lambda: np.zeros(4)), defaultdict(dict)
    for f in files:
        labs = sorted({l for _, _, l in hy[f]})
        sc = refs[f].score(activity(hy[f], labs, refs[f].n))
        for k, c in sc.items():
            tot[k] += c
            per_sess[k][f] = c
    sp = tot["all"][3] or 1.0
    print(f"\nall            : {fmt(tot['all'])}")
    print(f"\nfour conditions (radius {a.radius} s; share = % of reference speech):")
    for k, name in CONDITIONS:
        print(f"  {k} {name:52s}: share {100 * tot[k][3] / sp:5.1f}% | {fmt(tot[k])}")
    print(f"\nall language switches (near) : {fmt(tot['near-all'])}")
    print(f"elsewhere (far)              : {fmt(tot['far-all'])}")
    paired_condition_test(per_sess, "SS-DL", "SS-SL", "confusion: within-speaker switch vs elsewhere")
    print("\nper language (reference language regions):")
    for k in sorted(x for x in tot if x.startswith("lang:")):
        n_s = sum(1 for c in per_sess[k].values() if c[3] > 0)
        print(f"  {k[5:]:>12s} : {n_s:2d} sessions | {fmt(tot[k])}")
# ------------------------------------------------------------------- windows
def frame_cumsum(segs, n):
    """Per-label cumulative activity over 10 ms frames (for O(1) window coverage)."""
    labels = sorted({l for _, _, l in segs})
    idx = {l: i for i, l in enumerate(labels)}
    diff = np.zeros((max(len(labels), 1), n + 1), np.int32)
    for s, e, l in segs:
        a, b = min(n, max(0, int(round(s / STEP)))), min(n, max(0, int(round(e / STEP))))
        if b > a:
            diff[idx[l], a] += 1
            diff[idx[l], b] -= 1
    act = np.cumsum(diff, axis=1)[:, :n] > 0
    anyact = act.any(axis=0)
    cs = np.concatenate([np.zeros((act.shape[0], 1)), np.cumsum(act, axis=1)], axis=1)
    cs_any = np.concatenate([[0.0], np.cumsum(anyact)])
    return labels, cs, cs_any


def windows_for_file(sd_segs, ld_segs, dur, win, hop):
    n = int(np.ceil(dur / STEP))
    spk_lab, spk_cs, any_cs = frame_cumsum(sd_segs, n)
    lang_lab, lang_cs, _ = frame_cumsum(ld_segs, n) if ld_segs else ([], None, None)
    W = int(round(win / STEP))
    rows = []
    t = 0.0
    while t + win <= dur + 1e-6:
        a = int(round(t / STEP))
        b = min(n, a + W)
        speech = (any_cs[b] - any_cs[a]) / W
        if speech >= 0.5 and spk_lab:
            cov = spk_cs[:, b] - spk_cs[:, a]
            order = np.argsort(-cov)
            top = cov[order[0]] / W
            second = cov[order[1]] / W if len(order) > 1 else 0.0
            lang = "UNK"
            if lang_cs is not None and lang_lab:
                lc = lang_cs[:, b] - lang_cs[:, a]
                j = int(np.argmax(lc))
                if lc[j] / W >= 0.5:
                    lang = lang_lab[j]
            rows.append(dict(start=round(t, 3), end=round(t + win, 3), spk=spk_lab[order[0]],
                             lang=lang, single=bool(top >= 0.9 and second <= 0.1), speech=float(speech)))
        t += hop
    return rows



def cmd_embed(a):
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    sd, ld = read_rttm_dir(a.sd_ref), (read_rttm_dir(a.ld_ref) if a.ld_ref else {})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncDecSpeakerLabelModel.from_pretrained("titanet_large").eval().to(dev)
    meta, embs = [], []
    files = sorted(sd)
    for k, f in enumerate(files):
        paths = glob.glob(os.path.join(a.audio_dir, "**", f + ".wav"), recursive=True) + \
                glob.glob(os.path.join(a.audio_dir, "**", f + ".flac"), recursive=True)
        if not paths:
            print(f"  [skip] no audio for {f}")
            continue
        x, sr = sf.read(paths[0], dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            sr = 16000
        rows = windows_for_file(sd[f], ld.get(f, []), len(x) / sr, a.win, a.hop)
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            W = int(a.win * sr)
            sig = np.stack([np.pad(x[int(r["start"] * sr): int(r["start"] * sr) + W],
                                   (0, max(0, W - len(x[int(r["start"] * sr): int(r["start"] * sr) + W]))))
                            for r in chunk]).astype(np.float32)
            with torch.no_grad():
                s = torch.from_numpy(sig).to(dev)
                lens = torch.full((s.shape[0],), s.shape[1], dtype=torch.long, device=dev)
                _, e = model.forward(input_signal=s, input_signal_length=lens)
            embs.append(e.float().cpu().numpy())
            for r in chunk:
                r["file"] = f
                meta.append(r)
        print(f"  [{k + 1}/{len(files)}] {f}: {len(rows)} windows")
    E = np.concatenate(embs).astype(np.float32)
    np.savez_compressed(a.out, emb=E,
                        fid=np.array([m["file"] for m in meta]),
                        start=np.array([m["start"] for m in meta]),
                        end=np.array([m["end"] for m in meta]),
                        spk=np.array([m["file"] + "|" + m["spk"] for m in meta]),
                        lang=np.array([m["lang"] for m in meta]),
                        single=np.array([m["single"] for m in meta]),
                        speech=np.array([m["speech"] for m in meta]))
    print(f"saved {E.shape[0]} embeddings of dim {E.shape[1]} to {a.out}")


def cmd_lid(a):
    """Pseudo language labels when Track-2 references are missing.

    Each reference speaker segment is cut into chunks of about --chunk seconds, every chunk is
    classified with the SpeechBrain VoxLingua107 language-ID model, and adjacent chunks with the
    same language are merged. Output: one RTTM per session, label field = language code.
    """
    import soundfile as sf
    import torch
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:  # older SpeechBrain
        from speechbrain.pretrained import EncoderClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clf = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                         savedir=os.path.join(a.out, "_lid_model"), run_opts={"device": dev})
    ind2lab = clf.hparams.label_encoder.ind2lab
    allowed = None
    if a.allowed:
        keep = set(a.allowed)
        allowed = torch.tensor([i for i, l in ind2lab.items() if l.split(":")[0].strip() in keep], device=dev)
        print("restricting to", sorted(keep), "->", len(allowed), "classes")
    os.makedirs(a.out, exist_ok=True)
    sd = read_rttm_dir(a.sd_ref)
    for k, f in enumerate(sorted(sd)):
        paths = glob.glob(os.path.join(a.audio_dir, "**", f + ".wav"), recursive=True) + \
                glob.glob(os.path.join(a.audio_dir, "**", f + ".flac"), recursive=True)
        if not paths:
            print(f"  [skip] no audio for {f}")
            continue
        x, sr = sf.read(paths[0], dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            sr = 16000
        chunks = []
        for s, e, _ in sd[f]:
            n = max(1, int(round((e - s) / a.chunk)))
            edges = np.linspace(s, e, n + 1)
            chunks += [(edges[i], edges[i + 1]) for i in range(n) if edges[i + 1] - edges[i] >= 0.5]
        labs = []
        for i in range(0, len(chunks), a.batch):
            part = chunks[i:i + a.batch]
            segs = [x[int(s * sr):int(e * sr)] for s, e in part]
            T = max(len(g) for g in segs)
            wav = torch.zeros(len(segs), T)
            for j, g in enumerate(segs):
                wav[j, :len(g)] = torch.from_numpy(g)
            rel = torch.tensor([len(g) / T for g in segs])
            with torch.no_grad():
                logp = clf.classify_batch(wav.to(dev), rel.to(dev))[0]
            if allowed is not None:
                idx = allowed[logp[:, allowed].argmax(dim=1)]
            else:
                idx = logp.argmax(dim=1)
            labs += [ind2lab[int(j)].split(":")[0].strip() for j in idx.cpu()]
        merged = []
        for (s, e), l in sorted(zip(chunks, labs)):
            if merged and merged[-1][2] == l and s - merged[-1][1] < 0.05:
                merged[-1] = (merged[-1][0], max(e, merged[-1][1]), l)
            else:
                merged.append((s, e, l))
        write_rttm(os.path.join(a.out, f + ".rttm"), f, merged)
        print(f"  [{k + 1}/{len(sd)}] {f}: {len(chunks)} chunks, languages {sorted(set(labs))}")



def load_npz(p):
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------- statistics
def paired_stats(d, n_boot=10000, seed=0, tie=0.01):
    """Paired differences d (one per session or speaker; positive = first is larger)."""
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    out = dict(n=len(d))
    if len(d) == 0:
        return out
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)  # paired: whole pairs are resampled
    nz = d[d != 0]
    p, r_rb = 1.0, 0.0
    if len(nz) >= 1:
        from scipy.stats import rankdata
        rk = rankdata(np.abs(nz))
        wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
        r_rb = (wp - wm) / (wp + wm)  # matched-pairs rank-biserial correlation
        if len(nz) >= 2:
            p = float(wilcoxon(nz).pvalue)  # zero differences dropped (Wilcoxon's convention)
    out.update(mean=d.mean(), median=float(np.median(d)), ci=(np.percentile(boot, 2.5), np.percentile(boot, 97.5)),
               p=p, r=r_rb, win=int(np.sum(d > tie)), tie=int(np.sum(np.abs(d) <= tie)), loss=int(np.sum(d < -tie)))
    return out


def fmt_stats(s, unit="points"):
    if s.get("n", 0) == 0 or "mean" not in s:
        return "no pairs"
    return (f"n={s['n']}, mean {s['mean']:+.3f} {unit}, median {s['median']:+.3f}, 95% CI "
            f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}], Wilcoxon p = {s['p']:.3g}, rank-biserial r = {s['r']:+.2f}, "
            f"better/tied/worse = {s['win']}/{s['tie']}/{s['loss']}")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def paired_condition_test(per_sess, c1, c2, title, min_speech=30.0):
    """Within-system comparison of two regions, paired over sessions (confusion rate)."""
    if c1 not in per_sess or c2 not in per_sess:
        return None
    f = [x for x in per_sess[c1] if per_sess[c1][x][3] >= min_speech and per_sess[c2].get(x, np.zeros(4))[3] > 0]
    d = [rate(per_sess[c1][x]) - rate(per_sess[c2][x]) for x in f]
    s = paired_stats(d)
    print(f"  {title}: {fmt_stats(s)}")
    return s


def eer(scores, labels):
    from sklearn.metrics import roc_curve
    if len(np.unique(labels)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return 100 * (fpr[i] + fnr[i]) / 2


# ---------------------------------------------------------------- embeddings: H1
def usable(z):
    """Windows with one clear speaker and a reference language."""
    return z["single"] & (z["lang"] != "UNK")


def speaker_drift(E, spk, lang, start, win, min_win=5, cap=2500, seed=0):
    """Exact per-speaker similarity within vs across languages.

    All pairs of a speaker's windows that do not overlap in time (|dt| >= win) are used;
    languages are weighted equally (rho_same = mean over languages, rho_cross = mean over
    language pairs). The time-matched drift compares same- and cross-language pairs inside
    the same time-distance bin, so slow drift of the channel or of the speaker's voice
    over the session cannot explain it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in np.unique(spk):
        ix = np.flatnonzero(spk == s)
        langs = [l for l in np.unique(lang[ix]) if l != "UNK" and np.sum(lang[ix] == l) >= min_win]
        if len(langs) < 2:
            continue
        ix = ix[np.isin(lang[ix], langs)]
        if len(ix) > cap:
            ix = np.sort(rng.choice(ix, cap, replace=False))
        X, t, L = E[ix].astype(np.float32), start[ix], lang[ix]
        G = X @ X.T
        dt = np.abs(t[:, None] - t[None, :])
        valid = dt >= win - 1e-6
        same_v, cross_v = [], []
        for i, la in enumerate(langs):
            ma = L == la
            for lb in langs[i:]:
                mb = L == lb
                v = valid[np.ix_(ma, mb)]
                if v.sum() >= 10:
                    (same_v if la == lb else cross_v).append(G[np.ix_(ma, mb)][v].mean())
        if not same_v or not cross_v:
            continue
        eq = L[:, None] == L[None, :]
        tm = []
        for lo, hi in TIME_BINS:
            b = valid & (dt >= max(lo, win)) & (dt < hi)
            if np.sum(b & eq) >= 20 and np.sum(b & ~eq) >= 20:
                tm.append(G[b & eq].mean() - G[b & ~eq].mean())
        rows.append((s, float(np.mean(same_v)), float(np.mean(cross_v)), float(np.mean(tm)) if tm else np.nan,
                     len(langs)))
    return rows


def drift_summary(rows, title):
    if not rows:
        print(f"[{title}] no multilingual speakers")
        return {}
    same = np.array([r[1] for r in rows])
    cross = np.array([r[2] for r in rows])
    tm = np.array([r[3] for r in rows])
    s1, s2 = paired_stats(same - cross), paired_stats(tm[np.isfinite(tm)])
    print(f"[{title}] multilingual speakers: {len(rows)} "
          f"(languages per speaker: {dict((int(u), int(c)) for u, c in zip(*np.unique([r[4] for r in rows], return_counts=True)))})")
    print(f"  rho_same  : mean {same.mean():.4f} sd {same.std(ddof=1) if len(same) > 1 else 0:.4f} median {np.median(same):.4f}")
    print(f"  rho_cross : mean {cross.mean():.4f} sd {cross.std(ddof=1) if len(cross) > 1 else 0:.4f} median {np.median(cross):.4f}")
    print(f"  drift (rho_same - rho_cross)  : {fmt_stats(s1, 'cos')}")
    print(f"  drift, time-matched           : {fmt_stats(s2, 'cos')}")
    print(f"  speakers with positive drift  : {100 * np.mean(same > cross):.1f}%")
    return dict(n=len(rows), same=same.mean(), cross=cross.mean(), drift=s1, drift_tm=s2)


def language_probe(E, lang, spk, fid, max_per_rec=3000, seed=0, title="probe", quiet=False):
    """Can a linear classifier read the language of a speaker it has never seen?

    DISPLACE language references use labels that are relative to each recording (L1, L2, ...),
    so languages are only comparable inside a recording. In every recording a logistic
    regression is trained on all other speakers and tested on the held-out speaker
    (leave-one-speaker-out); balanced accuracy is computed per recording and averaged.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rng = np.random.default_rng(seed)
    rows = []
    for f in np.unique(fid):
        ix = np.flatnonzero(fid == f)
        if len(ix) > max_per_rec:
            ix = np.sort(rng.choice(ix, max_per_rec, replace=False))
        y, s = lang[ix], spk[ix]
        if len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
            continue
        pred = np.array([""] * len(ix), dtype=object)
        for sp in np.unique(s):
            tr = s != sp
            labs, cnt = np.unique(y[tr], return_counts=True)
            if len(labs) < 2 or cnt.min() < 5:
                continue
            te = ~tr & np.isin(y, labs)  # a language no other speaker uses cannot be learned: skipped
            if te.any():
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
                pred[te] = clf.fit(E[ix][tr], y[tr]).predict(E[ix][te])
        ok = pred != ""
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        yy, pp = y[ok], pred[ok].astype(str)
        _, cnt = np.unique(yy, return_counts=True)
        rows.append((f, 100 * balanced_accuracy_score(yy, pp), 100 * f1_score(yy, pp, average="macro"),
                     100 / len(cnt), 100 * cnt.max() / cnt.sum(), int(ok.sum())))
    if not rows:
        if not quiet:
            print(f"[{title}] no recording with two languages and two speakers")
        return {}
    b = np.array([r[1] for r in rows])
    st = paired_stats(b - np.array([r[3] for r in rows]))
    out = dict(bacc=b.mean(), ci=(b.mean() - st["mean"] + st["ci"][0], b.mean() - st["mean"] + st["ci"][1]),
               f1=float(np.mean([r[2] for r in rows])), chance=float(np.mean([r[3] for r in rows])),
               majority=float(np.mean([r[4] for r in rows])), p=st["p"], n_rec=len(rows),
               n=int(sum(r[5] for r in rows)))
    if not quiet:
        print(f"[{title}] {len(rows)} recordings, {out['n']} windows; leave-one-speaker-out inside each recording; "
              f"logistic regression (C=1, balanced class weights)")
        print(f"  balanced accuracy {out['bacc']:.1f}% (mean over recordings; 95% CI [{out['ci'][0]:.1f}, {out['ci'][1]:.1f}]), "
              f"macro-F1 {out['f1']:.1f}%")
        print(f"  chance {out['chance']:.1f}%, majority-class rate {out['majority']:.1f}%; "
              f"above chance: Wilcoxon over recordings p = {out['p']:.3g}")
    return out


def window_stats(z, title):
    n = len(z["single"])
    sp = z["speech"] >= 0.5
    print(f"[{title}] windows: {n}; single-speaker {100 * z['single'].mean():.1f}%; "
          f"with a language label {100 * np.mean(z['lang'] != 'UNK'):.1f}%; used for analysis "
          f"(both) {100 * usable(z).mean():.1f}%; clustered (speech >= 50%) {100 * sp.mean():.1f}%")


def cmd_similarity(a):
    z = load_npz(a.emb)
    window_stats(z, os.path.basename(a.emb))
    win = float(np.median(z["end"] - z["start"]))
    m = usable(z)
    E, spk, lang, start, fid = l2n(z["emb"][m]), z["spk"][m], z["lang"][m], z["start"][m], z["fid"][m]
    rows = speaker_drift(E, spk, lang, start, win, a.min_win)
    drift_summary(rows, "H1 drift")
    language_probe(E, lang, spk, fid, title="H1 language probe")


# ------------------------------------------------------- subspace estimators
def language_offsets(E, spk, lang, min_win=5):
    """Within-speaker language offsets, one block per speaker (embeddings L2-normalised first)."""
    E = l2n(E)
    blocks = []
    for s in np.unique(spk):
        ms = spk == s
        langs = [l for l in np.unique(lang[ms]) if l != "UNK" and np.sum(ms & (lang == l)) >= min_win]
        if len(langs) < 2:
            continue
        mus = np.stack([E[ms & (lang == l)].mean(0) for l in langs])
        blocks.append(mus - mus.mean(0))  # equal weight per language: the centre is not pulled by one language
    return blocks


def basis_from_offsets(blocks, k):
    if not blocks:
        raise SystemExit("no speaker with two languages: cannot fit the projection")
    _, S, Vt = np.linalg.svd(np.concatenate(blocks), full_matrices=False)
    return Vt[:k].T, S


def pca_basis(E, k):
    X = l2n(E)
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return Vt[:k].T


def nap_basis(E, spk, k):
    """Conventional NAP: top directions of ALL within-speaker variability (no language labels)."""
    X = l2n(E)
    R = np.empty_like(X)
    for s in np.unique(spk):
        m = spk == s
        R[m] = X[m] - X[m].mean(0)
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    return Vt[:k].T


def random_basis(D, k, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, k)))
    return Q


def shifted_offsets(E, spk, lang, start, min_win, seed):
    """Same estimator as LSP, but each speaker's time-ordered language labels are circularly
    shifted: run lengths and label counts are kept, alignment with the true language is lost."""
    rng = np.random.default_rng(seed)
    lang2 = lang.copy()
    for s in np.unique(spk):
        ix = np.flatnonzero(spk == s)
        ix = ix[np.argsort(start[ix])]
        if len(ix) >= 4:
            lang2[ix] = np.roll(lang[ix], int(rng.integers(len(ix) // 4, 3 * len(ix) // 4 + 1)))
    return language_offsets(E, spk, lang2, min_win)


def project(E, U):
    X = l2n(E)
    return l2n(X - (X @ U) @ U.T) if U is not None and U.shape[1] else X


def subspace_report(blocks, S, grid, eta_fn, n_boot=200, seed=0):
    M = sum(len(b) for b in blocks)
    bound = sum(len(b) - 1 for b in blocks)
    rank = int(np.sum(S > 1e-6 * S[0]))
    eps = np.cumsum(S ** 2) / np.sum(S ** 2)
    print(f"language offsets: {M} offsets from {len(blocks)} multilingual speakers; each speaker's offsets "
          f"sum to zero, so rank <= {bound}; numerical rank {rank}")
    print("singular values (first 16):", np.round(S[:16], 4))
    rng = np.random.default_rng(seed)
    V_full = basis_from_offsets(blocks, max(grid))[0]
    sims = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        Vb = basis_from_offsets([blocks[i] for i in pick], max(grid))[0]
        for k in grid:
            sims[k].append(np.sum((V_full[:, :k].T @ Vb[:, :min(k, Vb.shape[1])]) ** 2) / k)
    rep = {}
    for k in grid:
        e = eta_fn(V_full[:, :k])
        st = np.array(sims[k])
        rep[k] = dict(eps=100 * eps[k - 1], eta=100 * e, stab=st.mean(), stab5=np.percentile(st, 5))
        print(f"k={k:2d}: language energy removed eps_k {rep[k]['eps']:5.1f}% | between-speaker energy kept "
              f"eta_k {rep[k]['eta']:5.1f}% | stability over {n_boot} speaker bootstraps: mean "
              f"{rep[k]['stab']:.3f}, 5th pct {rep[k]['stab5']:.3f} (1 = identical subspace)")
    return rep, rank


# ---------------------------------------------------------------- clustering
def _eigh(L, vectors=False):
    """Symmetric eigen-decomposition, ascending. Uses the GPU when torch + CUDA exist (Colab)."""
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.from_numpy(L).float().cuda()
            if vectors:
                w, V = torch.linalg.eigh(t)
                return w.cpu().numpy(), V.cpu().numpy()
            return torch.linalg.eigvalsh(t).cpu().numpy()
    except ImportError:
        pass
    from scipy.linalg import eigh
    return eigh(L) if vectors else eigh(L, eigvals_only=True)


def nme_sc(E, max_spk=8, n_p=8, seed=0):
    """Spectral clustering with the normalized maximum eigengap rule (after Park et al., 2020)."""
    from sklearn.cluster import KMeans
    X = l2n(E)
    n = len(X)
    if n <= 3:
        return np.zeros(n, int)
    A = X @ X.T
    np.fill_diagonal(A, 1.0)
    kmax = min(max_spk, n - 1)
    p_grid = sorted({max(2, int(round(n * f))) for f in np.linspace(0.01, 0.25, n_p)} - {n})
    best = None
    for p in p_grid:
        p = min(p, n - 1)
        top = np.argpartition(-A, p, axis=1)[:, :p]
        B = np.zeros_like(A)
        B[np.arange(n)[:, None], top] = 1.0
        B = 0.5 * (B + B.T)
        L = np.diag(B.sum(1)) - B
        ev = _eigh(L)  # ascending; one call gives both ends
        lo, lmax = ev[:kmax + 1], ev[-1]
        gaps = np.diff(lo)
        g = gaps.max() / (lmax + 1e-10)
        r = p / (g + 1e-10)
        if best is None or r < best[0]:
            best = (r, p, int(np.argmax(gaps)) + 1, L)
    _, p, K, L = best
    if K <= 1:
        return np.zeros(n, int)
    _, V = _eigh(L, vectors=True)
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(V[:, :K])


def ahc_tree(E):
    from scipy.cluster.hierarchy import linkage
    return linkage(l2n(E).astype(np.float64), method="average", metric="cosine")


def ahc_cut(Z, thr):
    from scipy.cluster.hierarchy import fcluster
    return fcluster(Z, thr, criterion="distance") - 1


def labels_to_segments(start, end, lab, hop):
    """Each window owns hop seconds around its centre; adjacent equal labels are merged."""
    order = np.argsort(start)
    segs = []
    for i in order:
        c = 0.5 * (start[i] + end[i])
        s, e, l = c - hop / 2, c + hop / 2, f"spk{lab[i]}"
        if segs and segs[-1][2] == l and s - segs[-1][1] < 1e-3:
            segs[-1] = (segs[-1][0], e, l)
        else:
            segs.append((s, e, l))
    return segs


def hyp_labels(fr, start, end, lab, hop):
    """Frame label vector: each window owns hop seconds around its centre; cropped to reference speech (oracle)."""
    out = np.full(fr.n, -1, int)
    _, lab = np.unique(lab, return_inverse=True)
    c = 0.5 * (np.asarray(start) + np.asarray(end))
    a = np.clip(np.round((c - hop / 2) / STEP).astype(int), 0, fr.n)
    b = np.clip(np.round((c + hop / 2) / STEP).astype(int), 0, fr.n)
    for i in np.argsort(start):
        out[a[i]:b[i]] = lab[i]
    out[~fr.speech] = -1
    return out


def label_segments(v):
    """Label vector -> [(start, end, 'spkN')]."""
    segs = []
    for j in np.unique(v[v >= 0]):
        segs += [(s, e, f"spk{j}") for s, e in mask_to_segments(v == j)]
    return sorted(segs)


def split_and_change(lab, spk, lang, single, start, adj, min_wins=(5, 20), n_shift=20, seed=0):
    """Language-induced split (with a shifted-label control) and cluster-change rates."""
    rng = np.random.default_rng(seed)
    sr = {m: [] for m in min_wins}
    cc = np.zeros(4)  # [pairs across a language change, of which cluster changes, same-language pairs, changes]
    for s in np.unique(spk):
        sel = np.flatnonzero((spk == s) & single & (lang != "UNK"))
        if len(sel) < 2:
            continue
        sel = sel[np.argsort(start[sel])]
        L, Y = lang[sel], lab[sel]
        adjc = np.diff(start[sel]) <= adj + 1e-3
        dl, dc = L[1:] != L[:-1], Y[1:] != Y[:-1]
        cc += [np.sum(adjc & dl), np.sum(adjc & dl & dc), np.sum(adjc & ~dl), np.sum(adjc & ~dl & dc)]
        for mw in min_wins:
            langs = [l for l in np.unique(L) if np.sum(L == l) >= mw]
            if len(langs) < 2:
                continue
            keep = np.isin(L, langs)

            def is_split(Lx):
                overall = np.bincount(Y[keep]).argmax()
                return any(np.bincount(Y[Lx == l]).argmax() != overall for l in langs)

            shift = [is_split(np.roll(L, int(rng.integers(len(L) // 4, 3 * len(L) // 4 + 1)))) for _ in range(n_shift)]
            sr[mw].append((is_split(L), np.mean(shift)))
    return sr, cc


def run_system(z, U, refs, a, backend="nme", thr=None, out_dir=None, trees=None):
    """Cluster every session of one set with one projection; score with oracle speech regions."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    res = dict(comp=defaultdict(dict), kerr=[], sr={5: [], 20: []}, cc=np.zeros(4), sec=0.0, pyannote_gap=[])
    for f in np.unique(z["fid"]):
        if f not in refs:
            continue
        m = np.flatnonzero((z["fid"] == f) & (z["speech"] >= 0.5))[::a.stride]
        if len(m) == 0:
            continue
        X = project(z["emb"][m], U)
        t0 = time.perf_counter()
        if backend == "nme":
            lab = nme_sc(X, max_spk=a.max_spk)
        else:
            Z = trees[f] if trees is not None and f in trees else ahc_tree(X)
            lab = ahc_cut(Z, thr)
        res["sec"] += time.perf_counter() - t0
        fr = refs[f]
        hv = hyp_labels(fr, z["start"][m], z["end"][m], lab, a.hop * a.stride)
        for k, c in fr.score(hv).items():
            res["comp"][k][f] = c
        res["kerr"].append(abs(len(np.unique(lab)) - len(fr.labels)))
        sr, cc = split_and_change(lab, z["spk"][m], z["lang"][m], z["single"][m], z["start"][m], a.hop * a.stride)
        for mw in sr:
            res["sr"][mw] += sr[mw]
        res["cc"] += cc
        if out_dir:
            segs = label_segments(hv)
            write_rttm(os.path.join(out_dir, f + ".rttm"), f, segs)
            if a.check_pyannote:
                res["pyannote_gap"].append(abs(pyannote_der(a._sd[f], segs, f) - rate(res["comp"]["all"][f], None)))
    return res


def total(res, key="all"):
    return sum(res["comp"][key].values(), np.zeros(4))


def summary_line(res):
    t = total(res)
    sr = np.array(res["sr"][5]) if res["sr"][5] else np.zeros((0, 2))
    cc = res["cc"]
    out = dict(der=rate(t, None), conf=rate(t), kerr=float(np.mean(res["kerr"])) if res["kerr"] else np.nan,
               sr=100 * sr[:, 0].mean() if len(sr) else np.nan, sr_shift=100 * sr[:, 1].mean() if len(sr) else np.nan,
               n_ml=len(sr), cc_switch=100 * cc[1] / max(cc[0], 1), cc_same=100 * cc[3] / max(cc[2], 1),
               conf_ssdl=rate(total(res, "SS-DL")) if "SS-DL" in res["comp"] else np.nan)
    sr20 = np.array(res["sr"][20]) if res["sr"][20] else np.zeros((0, 2))
    out["sr20"] = 100 * sr20[:, 0].mean() if len(sr20) else np.nan
    return out


def print_system(name, res, conditions=True):
    s = summary_line(res)
    print(f"\n{name}")
    print(f"  all            : {fmt(total(res))}   mean |K err| {s['kerr']:.2f}")
    if s["n_ml"]:
        print(f"  split rate SR  : {s['sr']:.1f}% of {s['n_ml']} multilingual speakers (>= 5 windows per language); "
              f"shifted-label control {s['sr_shift']:.1f}%; SR with >= 20 windows per language {s['sr20']:.1f}%")
        print(f"  cluster change between adjacent windows of one speaker: across a language switch "
              f"{s['cc_switch']:.1f}% ({res['cc'][0]:.0f} pairs) vs same language {s['cc_same']:.1f}% ({res['cc'][2]:.0f} pairs)")
    if conditions and "SS-DL" in res["comp"]:
        sp = total(res)[3] or 1.0
        for k, nm in CONDITIONS:
            c = total(res, k)
            print(f"  {k} {nm:52s}: share {100 * c[3] / sp:5.1f}% | {fmt(c)}")
    if res["pyannote_gap"]:
        print(f"  frame scorer vs pyannote.metrics: max |DER difference| {max(res['pyannote_gap']):.3f} points over "
              f"{len(res['pyannote_gap'])} files")
    return s


def per_session(res, key="all", comp=None):
    return {f: rate(c, comp) for f, c in res["comp"][key].items() if c[3] > 0}


def compare(res_a, res_b, key="all", comp=None, min_speech=0.0):
    """Paired over sessions: rate(a) - rate(b); positive = b has fewer errors."""
    fa, fb = res_a["comp"][key], res_b["comp"][key]
    common = sorted(f for f in set(fa) & set(fb) if fa[f][3] > max(min_speech, 0))
    return paired_stats([rate(fa[f], comp) - rate(fb[f], comp) for f in common])


def mean_res(results):
    """Average several runs (control seeds) session by session."""
    out = dict(comp=defaultdict(dict), kerr=[], sr={5: [], 20: []}, cc=np.zeros(4), sec=0.0, pyannote_gap=[])
    for r in results:
        for k, d in r["comp"].items():
            for f, c in d.items():
                out["comp"][k][f] = out["comp"][k].get(f, 0) + c / len(results)
        out["kerr"] += r["kerr"]
        for mw in r["sr"]:
            out["sr"][mw] += r["sr"][mw]
        out["cc"] += r["cc"] / len(results)
    return out


# ------------------------------------------------- speaker information checks
def make_trials(z, m, win, per_type=1500, seed=0):
    """Within-session window pairs (non-overlapping) of four kinds: same/different speaker x same/different language."""
    rng = np.random.default_rng(seed)
    T = defaultdict(list)
    idx = np.flatnonzero(m)
    for f in np.unique(z["fid"][idx]):
        ix = idx[z["fid"][idx] == f]
        if len(ix) < 4:
            continue
        i, j = rng.choice(ix, 80000), rng.choice(ix, 80000)
        ok = np.abs(z["start"][i] - z["start"][j]) >= win - 1e-6
        i, j = i[ok], j[ok]
        ss, sl = z["spk"][i] == z["spk"][j], z["lang"][i] == z["lang"][j]
        for name, sel in (("tgt_same", ss & sl), ("tgt_cross", ss & ~sl), ("non_same", ~ss & sl), ("non_cross", ~ss & ~sl)):
            q = np.flatnonzero(sel)[:per_type]
            T[name] += list(zip(i[q], j[q]))
    return {k: np.array(v, int).reshape(-1, 2) for k, v in T.items()}


def speaker_info(E, z, m, trials, win, probe_ix, a, title):
    """Embedding-level effect of a projection: drift, pair cosines, EER, language probe."""
    X = l2n(E)
    cos = {k: np.sum(X[v[:, 0]] * X[v[:, 1]], 1) for k, v in trials.items() if len(v)}
    tgt = np.concatenate([cos.get("tgt_same", []), cos.get("tgt_cross", [])])
    non = np.concatenate([cos.get("non_same", []), cos.get("non_cross", [])])
    e_all = eer(np.concatenate([tgt, non]), np.r_[np.ones(len(tgt)), np.zeros(len(non))])
    tc = cos.get("tgt_cross", np.zeros(0))
    e_x = eer(np.concatenate([tc, non]), np.r_[np.ones(len(tc)), np.zeros(len(non))])
    rows = speaker_drift(X[m], z["spk"][m], z["lang"][m], z["start"][m], win, a.min_win, cap=1500)
    dr = np.array([r[1] - r[2] for r in rows]) if rows else np.zeros(0)
    out = dict(drift=float(dr.mean()) if len(dr) else np.nan, eer=e_all, eer_cross=e_x,
               cos={k: float(v.mean()) for k, v in cos.items()})
    if probe_ix is not None and not a.no_probe:
        out["probe"] = language_probe(X[probe_ix], z["lang"][probe_ix], z["spk"][probe_ix], z["fid"][probe_ix],
                                      max_per_rec=1500, quiet=True).get("bacc", np.nan)
    print(f"  [{title}] drift {out['drift']:+.4f} | EER all {e_all:.2f}% | EER cross-language targets {e_x:.2f}% | "
          f"probe bal. acc. {out.get('probe', np.nan):.1f}% | mean cosine: " +
          ", ".join(f"{k} {v:.3f}" for k, v in out["cos"].items()))
    return out


# ------------------------------------------------------------------- diarize
def env_report():
    import platform
    from importlib import metadata
    vers = []
    for p in ["numpy", "scipy", "scikit-learn", "torch", "nemo_toolkit", "pyannote.core", "pyannote.metrics", "speechbrain"]:
        try:
            vers.append(f"{p} {metadata.version(p)}")
        except Exception:
            pass
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        pass
    gpu = "none"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    print(f"environment: Python {platform.python_version()} | CPU {cpu} | GPU {gpu}")
    print("versions: " + ", ".join(vers))


def ref_counts(sd, title, max_spk):
    k = np.array([len({l for _, _, l in v}) for v in sd.values()])
    print(f"[{title}] {len(k)} sessions; reference speakers per session min {k.min()}, median {np.median(k):.0f}, "
          f"max {k.max()}; sessions above the cap of {max_spk}: {int(np.sum(k > max_spk))}")


def cmd_diarize(a):
    env_report()
    dev, ev = load_npz(a.dev_emb), load_npz(a.eval_emb)
    sd_dev, sd_ev = read_rttm_dir(a.dev_sd_ref), read_rttm_dir(a.eval_sd_ref)
    ld_dev = read_rttm_dir(a.dev_ld_ref) if a.dev_ld_ref else {}
    ld_ev = read_rttm_dir(a.eval_ld_ref) if a.eval_ld_ref else {}
    ref_counts(sd_dev, "dev", a.max_spk)
    ref_counts(sd_ev, "eval", a.max_spk)
    window_stats(dev, "dev")
    window_stats(ev, "eval")
    refs_dev, refs_ev = build_refs(sd_dev, ld_dev, a.radius, a.max_gap), build_refs(sd_ev, ld_ev, a.radius, a.max_gap)
    rate_dev, rate_ev = report_events(refs_dev, "dev"), report_events(refs_ev, "eval")
    win = float(np.median(dev["end"] - dev["start"]))
    summary = dict(events=dict(dev=rate_dev, eval=rate_ev))

    # --- language subspace from DEV only
    md = usable(dev)
    blocks = language_offsets(dev["emb"][md], dev["spk"][md], dev["lang"][md], a.min_win)
    _, S = basis_from_offsets(blocks, 1)
    Ed, spk_d, fid_d = l2n(dev["emb"][md]), dev["spk"][md], dev["fid"][md]
    diffs = []
    for f in np.unique(fid_d):
        mus = [Ed[spk_d == sp].mean(0) for sp in np.unique(spk_d[fid_d == f])]
        diffs += [mus[i] - mus[j] for i in range(len(mus)) for j in range(i + 1, len(mus))]
    Dm = np.stack(diffs)

    def eta_fn(V):
        return 1 - np.mean(np.sum((Dm @ V) ** 2, 1) / np.sum(Dm ** 2, 1))

    rank = int(np.sum(S > 1e-6 * S[0]))
    grid = [k for k in a.k_grid if k <= rank]
    print("\n=== Language subspace (development set) ===")
    sub, _ = subspace_report(blocks, S, grid, eta_fn)
    V_all, _ = basis_from_offsets(blocks, max(grid))
    summary["subspace"] = dict(n_offsets=sum(len(b) for b in blocks), n_speakers=len(blocks), rank=rank,
                               singular_values=S[:16].tolist(), per_k=sub)

    # --- 1) rank selection on DEV (NME-SC); pre-specified rule: lowest dev DER, ties -> smaller k
    print("\n=== Rank selection on the development set (NME-SC, oracle speech) ===")
    a._sd = sd_dev
    dev_scores = {}
    for k in [0] + grid:
        r = run_system(dev, V_all[:, :k] if k else None, refs_dev, a)
        s = summary_line(r)
        dev_scores[k] = s
        print(f"dev k={k:2d}: DER {s['der']:.2f} | conf {s['conf']:.2f} | SR {s['sr']:.1f}% | mean |K err| {s['kerr']:.2f}")
    k_best = min(grid, key=lambda k: (round(dev_scores[k]["der"], 2), k))
    print(f"chosen k on dev: {k_best}")
    U = V_all[:, :k_best]
    summary.update(dev=dev_scores, k_best=int(k_best))

    # --- 2) evaluation: baseline vs LSP (NME-SC)
    print("\n=== Evaluation set, NME-SC, oracle speech regions (no collar, overlap scored) ===")
    a._sd = sd_ev
    a.check_pyannote = True
    base = run_system(ev, None, refs_ev, a, out_dir=os.path.join(a.out, "eval_k0"))
    lsp = run_system(ev, U, refs_ev, a, out_dir=os.path.join(a.out, f"eval_k{k_best}"))
    a.check_pyannote = False
    summary["eval"] = {"baseline": print_system("BASELINE (k=0)", base), "lsp": print_system(f"LSP (k={k_best})", lsp)}
    print("\nPRIMARY TEST (pre-specified): DER baseline - DER LSP, paired over evaluation sessions")
    prim = compare(base, lsp, "all", None)
    print("  " + fmt_stats(prim))
    print(f"  relative DER change: {100 * (summary['eval']['lsp']['der'] - summary['eval']['baseline']['der']) / summary['eval']['baseline']['der']:+.1f}%")
    sec = {}
    print("\nSECONDARY (Holm-corrected below)")
    if "SS-DL" in base["comp"]:
        print("  H2, baseline: confusion at within-speaker switches minus elsewhere (sessions with >= 30 s at switches)")
        h2 = paired_condition_test(base["comp"], "SS-DL", "SS-SL", "baseline SS-DL - SS-SL")
        h2b = paired_condition_test(base["comp"], "DS-SL", "SS-SL", "baseline DS-SL - SS-SL (speaker change alone)")
        summary["h2"] = dict(ssdl_vs_sssl=h2, dssl_vs_sssl=h2b)
        sec["LSP vs baseline, confusion at within-speaker switches"] = compare(base, lsp, "SS-DL", 2, min_speech=30.0)
    sec["LSP vs baseline, confusion overall"] = compare(base, lsp, "all", 2)

    # --- 3) controls at the same rank
    if not a.no_controls:
        print(f"\n=== Controls: other projections removing {k_best} dimension(s) (eval, NME-SC) ===")
        ctrl_bases = {"PCA": [pca_basis(dev["emb"][md], k_best)],
                      "NAP (within-speaker)": [nap_basis(dev["emb"][md], spk_d, k_best)],
                      "Shifted-label LSP": [basis_from_offsets(shifted_offsets(dev["emb"][md], spk_d, dev["lang"][md],
                                                                             dev["start"][md], a.min_win, sd_), k_best)[0]
                                            for sd_ in range(a.control_seeds)],
                      "Random": [random_basis(U.shape[0], k_best, sd_) for sd_ in range(a.control_seeds)]}
        ctrl_res = {}
        for name, Us in ctrl_bases.items():
            runs = [run_system(ev, Uc, refs_ev, a) for Uc in Us]
            ctrl_res[name] = mean_res(runs) if len(runs) > 1 else runs[0]
            s = summary_line(ctrl_res[name])
            ders = [summary_line(r)["der"] for r in runs]
            spread = f" (sd over {len(runs)} seeds {np.std(ders, ddof=1):.2f})" if len(runs) > 1 else ""
            print(f"{name:22s}: DER {s['der']:.2f}{spread} | conf {s['conf']:.2f} | conf SS-DL {s['conf_ssdl']:.2f} | "
                  f"SR {s['sr']:.1f}% | |K err| {s['kerr']:.2f}")
            sec[f"LSP vs {name}, DER"] = compare(ctrl_res[name], lsp, "all", None)
        summary["controls"] = {n: summary_line(r) for n, r in ctrl_res.items()}
    else:
        ctrl_bases = {}

    # --- 4) second back end: AHC (average linkage, cosine), threshold tuned on DEV for each system
    if not a.no_ahc:
        print("\n=== Second back end: AHC, threshold tuned on dev separately for k=0 and k*=%d ===" % k_best)
        grid_t = np.round(np.arange(0.30, 1.31, 0.05), 2)
        ahc = {}
        for k, Uk in ((0, None), (k_best, U)):
            a._sd = sd_dev
            trees = {f: ahc_tree(project(dev["emb"][np.flatnonzero((dev["fid"] == f) & (dev["speech"] >= 0.5))[::a.stride]], Uk))
                     for f in np.unique(dev["fid"]) if f in refs_dev}
            ders = {t: summary_line(run_system(dev, Uk, refs_dev, a, backend="ahc", thr=t, trees=trees))["der"] for t in grid_t}
            t_best = min(ders, key=ders.get)
            print(f"  k={k}: dev DER by threshold " + " ".join(f"{t:.2f}:{d:.1f}" for t, d in ders.items()))
            print(f"  k={k}: chosen threshold {t_best:.2f} (dev DER {ders[t_best]:.2f})")
            a._sd = sd_ev
            ahc[k] = run_system(ev, Uk, refs_ev, a, backend="ahc", thr=t_best)
            ahc[k]["thr"] = float(t_best)
        summary["ahc"] = {"baseline": print_system("AHC baseline (k=0)", ahc[0]),
                          "lsp": print_system(f"AHC + LSP (k={k_best})", ahc[k_best])}
        sec["AHC: LSP vs baseline, DER"] = compare(ahc[0], ahc[k_best], "all", None)

    keys = list(sec)
    adj = holm([sec[k].get("p", 1.0) for k in keys])
    print("\nSecondary comparisons (positive = second system better), Holm-adjusted p:")
    for k, p_adj in zip(keys, adj):
        print(f"  {k}: {fmt_stats(sec[k])}; Holm p = {p_adj:.3g}")
        sec[k]["p_holm"] = float(p_adj)
    summary["primary"], summary["secondary"] = prim, sec

    # --- 5) does the projection keep speaker information? (eval embeddings)
    print("\n=== Speaker information after projection (eval; within-session, non-overlapping window pairs) ===")
    me = usable(ev)
    trials = make_trials(ev, me, win)
    print("  trials: " + ", ".join(f"{k} {len(v)}" for k, v in trials.items()))
    rng = np.random.default_rng(0)
    pix = np.flatnonzero(me)
    pix = np.sort(rng.choice(pix, min(len(pix), a.probe_max), replace=False))
    info = {"Original": speaker_info(ev["emb"], ev, me, trials, win, pix, a, "original"),
            "LSP": speaker_info(project(ev["emb"], U), ev, me, trials, win, pix, a, f"LSP k={k_best}")}
    for name, Us in ctrl_bases.items():
        outs = [speaker_info(project(ev["emb"], Uc), ev, me, trials, win, pix, a, f"{name} seed {i}") for i, Uc in enumerate(Us)]
        info[name] = {key: float(np.nanmean([o.get(key, np.nan) for o in outs])) for key in ("drift", "eer", "eer_cross", "probe")}
    summary["speaker_info"] = info

    # --- 6) per language, numerical checks, runtime
    print("\n=== Gain by amount of within-speaker code-switching (eval) ===")
    pb, pl = per_session(base, "all", None), per_session(lsp, "all", None)
    recs = sorted(f for f in pb if f in pl and refs_ev[f].ev is not None)
    swr = np.array([len(refs_ev[f].ev["ws"]) / refs_ev[f].hours for f in recs])
    gain = np.array([pb[f] - pl[f] for f in recs])
    if len(recs) >= 6:
        from scipy.stats import spearmanr
        rho, p_rho = spearmanr(swr, gain)
        print(f"  Spearman correlation between within-speaker switches per hour and DER gain over {len(recs)} "
              f"recordings: rho = {rho:+.2f}, p = {p_rho:.3g}")
        order = np.argsort(swr)
        summary["by_switch_rate"] = {}
        for name, part in zip(("low", "middle", "high"), np.array_split(order, 3)):
            fs = [recs[i] for i in part]
            cb = sum((base["comp"]["all"][f] for f in fs), np.zeros(4))
            cl = sum((lsp["comp"]["all"][f] for f in fs), np.zeros(4))
            st = paired_stats(gain[part])
            print(f"  {name:6s} third: {len(fs):2d} recordings, {swr[part].min():6.1f}-{swr[part].max():6.1f} switches/h | "
                  f"baseline DER {rate(cb, None):5.2f} | LSP DER {rate(cl, None):5.2f} | mean gain {st['mean']:+.2f} "
                  f"[{st['ci'][0]:+.2f}, {st['ci'][1]:+.2f}]")
            summary["by_switch_rate"][name] = dict(n=len(fs), lo=float(swr[part].min()), hi=float(swr[part].max()),
                                                   base=rate(cb, None), lsp=rate(cl, None), gain=st)
        summary["switch_rate_spearman"] = dict(rho=float(rho), p=float(p_rho))
    Xe = l2n(ev["emb"])
    pn = np.linalg.norm(Xe - (Xe @ U) @ U.T, axis=1)
    print(f"\nnorm of projected unit embeddings: min {pn.min():.3f}, 1st percentile {np.percentile(pn, 1):.3f}, "
          f"share below 0.5: {100 * np.mean(pn < 0.5):.3f}%")
    hours = sum(r.hours for r in refs_ev.values())
    t0 = time.perf_counter()
    for _ in range(5):
        project(ev["emb"], U)
    t_proj = (time.perf_counter() - t0) / 5
    print(f"runtime on this machine ({hours:.2f} h of eval audio, {len(ev['emb'])} windows): projection "
          f"{1000 * t_proj / hours:.1f} ms per hour of audio (CPU, numpy); NME-SC clustering {base['sec'] / hours:.1f} s "
          f"per hour (baseline) and {lsp['sec'] / hours:.1f} s per hour (LSP)")
    summary["runtime"] = dict(proj_ms_per_h=1000 * t_proj / hours, nme_s_per_h=base["sec"] / hours)
    summary["per_session_der"] = {f: dict(baseline=pb[f], lsp=pl.get(f, np.nan)) for f in sorted(pb)}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nsummary written to {os.path.join(a.out, 'summary.json')}")


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("switch-errors")
    p.add_argument("--sd-ref", required=True)
    p.add_argument("--ld-ref", required=True)
    p.add_argument("--hyp", required=True, help="directory of system RTTMs")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--max-gap", type=float, default=2.0)
    p.set_defaults(func=cmd_switch_errors)

    p = sub.add_parser("lid")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--sd-ref", required=True)
    p.add_argument("--out", required=True, help="output folder for pseudo language RTTMs")
    p.add_argument("--chunk", type=float, default=3.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--allowed", nargs="*", help="language codes to keep, e.g. hi en kn bn")
    p.set_defaults(func=cmd_lid)

    p = sub.add_parser("embed")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--sd-ref", required=True)
    p.add_argument("--ld-ref")
    p.add_argument("--out", required=True)
    p.add_argument("--win", type=float, default=1.5)
    p.add_argument("--hop", type=float, default=0.75)
    p.add_argument("--batch", type=int, default=64)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("similarity")
    p.add_argument("--emb", required=True)
    p.add_argument("--min-win", type=int, default=5)
    p.set_defaults(func=cmd_similarity)

    p = sub.add_parser("diarize")
    p.add_argument("--dev-emb", required=True)
    p.add_argument("--eval-emb", required=True)
    p.add_argument("--dev-sd-ref", required=True)
    p.add_argument("--eval-sd-ref", required=True)
    p.add_argument("--dev-ld-ref")
    p.add_argument("--eval-ld-ref")
    p.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    p.add_argument("--min-win", type=int, default=5, help="windows needed per speaker-language pair")
    p.add_argument("--hop", type=float, default=0.75)
    p.add_argument("--max-spk", type=int, default=8)
    p.add_argument("--stride", type=int, default=1, help="use every n-th window (2 = about 8x faster clustering)")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--max-gap", type=float, default=2.0)
    p.add_argument("--control-seeds", type=int, default=3)
    p.add_argument("--probe-max", type=int, default=60000)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--no-ahc", action="store_true")
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_diarize, check_pyannote=False)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
