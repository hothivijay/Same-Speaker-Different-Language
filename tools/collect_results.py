#!/usr/bin/env python3
"""Collect the headline numbers from the logs written by run_all.py.

usage: python tools/collect_results.py LOGDIR RESULTDIR
Reads LOGDIR/similarity_dev.log, similarity_eval.log, diarize.log (and diarize_w3.log
if present) and writes RESULTDIR/results.json and RESULTDIR/RESULTS.md.
"""
import json
import os
import re
import sys

F = r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"


def grab(pat, text, cast=float, flags=0, all_=False):
    if all_:
        return [tuple(cast(x) for x in m) if isinstance(m, tuple) else cast(m) for m in re.findall(pat, text, flags)]
    m = re.search(pat, text, flags)
    if not m:
        return None
    g = m.groups()
    return cast(g[0]) if len(g) == 1 else tuple(cast(x) for x in g)


def stats_line(line):
    """Parse the output of fmt_stats."""
    if line is None or "no pairs" in line:
        return None
    m = re.search(r"n=(\d+), mean " + F + r" \S+, median " + F + r", 95% CI \[" + F + r", " + F + r"\], Wilcoxon p = "
                  + F + r", rank-biserial r = " + F + r", better/tied/worse = (\d+)/(\d+)/(\d+)", line)
    if not m:
        return None
    g = m.groups()
    return dict(n=int(g[0]), mean=float(g[1]), median=float(g[2]), lo=float(g[3]), hi=float(g[4]), p=float(g[5]),
                r=float(g[6]), win=int(g[7]), tie=int(g[8]), loss=int(g[9]))


def line_after(text, key):
    for line in text.splitlines():
        if key in line:
            return line
    return None


def similarity(text):
    v = {}
    v["windows_used"] = grab(r"used for analysis \(both\) " + F + "%", text)
    v["n_ml"] = grab(r"multilingual speakers: (\d+)", text, int)
    v["same"] = grab(r"rho_same\s*: mean " + F, text)
    v["cross"] = grab(r"rho_cross\s*: mean " + F, text)
    v["drift"] = stats_line(line_after(text, "drift (rho_same - rho_cross)"))
    v["drift_tm"] = stats_line(line_after(text, "drift, time-matched"))
    v["pos"] = grab(r"speakers with positive drift\s*: " + F, text)
    v["probe_rec"] = grab(r"\[H1 language probe\] (\d+) recordings", text, int)
    v["probe"] = grab(r"balanced accuracy " + F + r"% \(mean over recordings; 95% CI \[" + F + ", " + F + r"\]\), macro-F1 " + F, text)
    v["chance"] = grab(r"chance " + F + "%, majority-class rate " + F + "%; above chance: Wilcoxon over recordings p = " + F, text)
    return v


def diarize(text):
    v = {}
    v["maxspk"] = grab(r"\[dev\] \d+ sessions; reference speakers per session min \d+, median \d+, max (\d+)", text, int), \
        grab(r"\[eval\] \d+ sessions; reference speakers per session min \d+, median \d+, max (\d+)", text, int)
    v["above_cap"] = grab(r"\[dev\].*sessions above the cap of 8: (\d+)", text, int), grab(r"\[eval\].*sessions above the cap of 8: (\d+)", text, int)
    ev = {}
    for part in ("dev", "eval"):
        seg = text.split(f"[{part}] ")
        blk = [s for s in seg if s.split("\n")[0].strip().endswith("h of audio")]
        if blk:
            b = blk[0]
            ev[part] = dict(hours=grab(F + " h of audio", b),
                            ws=grab(r"within-speaker language switches :\s+(\d+)\s+\(\s*" + F, b),
                            sc=grab(r"switches with a speaker change\s+:\s+(\d+)\s+\(\s*" + F, b),
                            amb=grab(r"switches without a clear speaker :\s+(\d+)\s+\(\s*" + F, b),
                            spk=grab(r"speaker changes, same language\s+:\s+(\d+)\s+\(\s*" + F, b))
    v["events"] = ev
    v["offsets"] = grab(r"language offsets: (\d+) offsets from (\d+) multilingual speakers; .* rank <= (\d+); numerical rank (\d+)", text, int)
    v["ks"] = {int(k): dict(eps=float(e), eta=float(h), stab=float(s), stab5=float(s5)) for k, e, h, s, s5 in re.findall(
        r"k=\s*(\d+): language energy removed eps_k\s+" + F + r"% \| between-speaker energy kept eta_k\s+" + F +
        r"% \| stability over \d+ speaker bootstraps: mean " + F + ", 5th pct " + F, text)}
    v["dev"] = {int(k): dict(der=float(d), conf=float(c), sr=float(s), kerr=float(ke)) for k, d, c, s, ke in re.findall(
        r"dev k=\s*(\d+): DER " + F + r" \| conf " + F + r" \| SR " + F + r"% \| mean \|K err\| " + F, text)}
    v["k_best"] = grab(r"chosen k on dev: (\d+)", text, int)

    def system(name):
        i = text.find("\n" + name)
        if i < 0:
            return None
        b = text[i + 1:].split("\n\n")[0]
        s = dict(all=grab(r"all\s+: ref\s+" + F + r" h \| miss\s+" + F + r" \| FA\s+" + F + r" \| conf\s+" + F + r" \| DER\s+" + F, b),
                 kerr=grab(r"mean \|K err\| " + F, b),
                 sr=grab(r"split rate SR\s+: " + F + r"% of (\d+) multilingual speakers.*?control " + F + r"%; SR with >= 20 windows per language " + F, b),
                 cc=grab(r"across a language switch " + F + r"% \((\d+) pairs\) vs same language " + F + r"% \((\d+) pairs\)", b),
                 pyannote=grab(r"max \|DER difference\| " + F, b))
        for k in ("SS-SL", "SS-DL", "DS-SL", "DS-DL"):
            s[k] = grab(k + r" .*?: share\s+" + F + r"% \| ref\s+" + F + r" h \| miss\s+" + F + r" \| FA\s+" + F + r" \| conf\s+" + F + r" \| DER\s+" + F, b)
        return s

    v["base"] = system("BASELINE (k=0)")
    v["lsp"] = system(f"LSP (k={v['k_best']})")
    v["ahc_base"] = system("AHC baseline (k=0)")
    v["ahc_lsp"] = system(f"AHC + LSP (k={v['k_best']})")
    v["ahc_thr"] = grab(r"k=0: chosen threshold " + F, text), grab(r"k=\d+: chosen threshold " + F + r" \(dev DER [\d.]+\)\n\nAHC", text)
    pi = text.find("PRIMARY TEST")
    v["primary"] = stats_line(text[pi:].split("\n")[1]) if pi >= 0 else None
    v["relchange"] = grab(r"relative DER change: " + F, text)
    v["h2_ssdl"] = stats_line(line_after(text, "baseline SS-DL - SS-SL"))
    v["h2_dssl"] = stats_line(line_after(text, "baseline DS-SL - SS-SL"))
    ctrl = {}
    for name in ("PCA", "NAP (within-speaker)", "Shifted-label LSP", "Random"):
        m = re.search(re.escape(name) + r"\s*: DER " + F + r"(?: \(sd over \d+ seeds " + F + r"\))? \| conf " + F +
                      r" \| conf SS-DL " + F + r" \| SR " + F + r"% \| \|K err\| " + F, text)
        if m:
            g = m.groups()
            ctrl[name] = dict(der=float(g[0]), sd=float(g[1]) if g[1] else None, conf=float(g[2]), ssdl=float(g[3]),
                              sr=float(g[4]), kerr=float(g[5]))
    v["controls"] = ctrl
    sec = {}
    for line in text.splitlines():
        m = re.match(r"\s+(.+?): (n=\d+.*); Holm p = " + F, line)
        if m:
            d = stats_line(m.group(2))
            if d:
                d["holm"] = float(m.group(3))
                sec[m.group(1)] = d
    v["secondary"] = sec
    info = {}
    for m in re.finditer(r"\[(.+?)\] drift " + F + r" \| EER all " + F + r"% \| EER cross-language targets " + F +
                         r"% \| probe bal\. acc\. " + F + r"% \| mean cosine: (.*)", text):
        cos = dict((k, float(x)) for k, x in re.findall(r"(\w+) " + F, m.group(6)))
        info[m.group(1)] = dict(drift=float(m.group(2)), eer=float(m.group(3)), eer_x=float(m.group(4)),
                                probe=float(m.group(5)), cos=cos)
    v["info"] = info
    v["spearman"] = grab(r"rho = " + F + r", p = " + F, text)
    v["thirds"] = {n: dict(n=int(c), lo=float(a), hi=float(b), base=float(d1), lsp=float(d2), gain=float(g), glo=float(l), ghi=float(h))
                   for n, c, a, b, d1, d2, g, l, h in re.findall(
                       r"(low|middle|high)\s+third:\s+(\d+) recordings,\s+" + F + r"-\s*" + F + r" switches/h \| baseline DER\s+" + F +
                       r" \| LSP DER\s+" + F + r" \| mean gain " + F + r" \[" + F + ", " + F + r"\]", text)}
    v["norm_min"] = grab(r"norm of projected unit embeddings: min " + F, text)
    v["runtime"] = grab(r"projection " + F + r" ms per hour of audio .*?NME-SC clustering " + F + r" s per hour \(baseline\)", text)
    v["env"] = grab(r"environment: (.*)", text, str), grab(r"versions: (.*)", text, str)
    return v


def clean(text):
    """Drop Python warning lines that may be interleaved with the output."""
    return "\n".join(l for l in text.splitlines() if "warnings.warn" not in l and "Warning:" not in l) + "\n"


def fmt_s(s, unit=""):
    if not s:
        return "n/a"
    return (f"{s['mean']:+.3f}{unit} (median {s['median']:+.3f}, 95% CI [{s['lo']:+.3f}, {s['hi']:+.3f}], "
            f"Wilcoxon p = {s['p']:.3g})")


def markdown(r):
    sd, se, d = r["sim_dev"], r["sim_eval"], r["dia"]
    k = d["k_best"]
    L = ["# Results", "",
         "All numbers are read from the logs in this folder. Oracle speech regions, no collar, overlap scored.", "",
         "## H1: does a speaker's embedding drift with language?", "",
         "| | Development | Evaluation |", "|---|---|---|",
         f"| multilingual speakers | {sd['n_ml']} | {se['n_ml']} |",
         f"| same-speaker cosine, same / other language | {sd['same']:.3f} / {sd['cross']:.3f} | {se['same']:.3f} / {se['cross']:.3f} |",
         f"| drift (same - other) | {fmt_s(sd['drift'])} | {fmt_s(se['drift'])} |",
         f"| time-matched drift | {sd['drift_tm']['mean']:+.3f} | {se['drift_tm']['mean']:+.3f} |",
         f"| language probe, balanced accuracy (chance) | {sd['probe'][0]:.1f}% ({sd['chance'][0]:.1f}%) | {se['probe'][0]:.1f}% ({se['chance'][0]:.1f}%) |",
         "", "## H2: are there more errors at language switches? (evaluation, baseline)", "",
         "| condition | share of speech | confusion |", "|---|---|---|"]
    for c, name in (("SS-SL", "same speaker, same language"), ("SS-DL", "same speaker, language switch"),
                    ("DS-SL", "speaker change, same language"), ("DS-DL", "speaker change + language switch")):
        b = d["base"][c]
        L.append(f"| {name} | {b[0]:.1f}% | {b[4]:.2f}% |")
    L += ["", f"Within-speaker switch minus elsewhere: {fmt_s(d['h2_ssdl'], ' points')}",
          f"Speaker change minus elsewhere: {fmt_s(d['h2_dssl'], ' points')}", "",
          f"## H3: language-subspace projection (rank k = {k}, chosen on development data)", "",
          "| system | DER | confusion | split rate |", "|---|---|---|---|"]
    for name, key in (("NME-SC", "base"), ("NME-SC + LSP", "lsp"), ("AHC", "ahc_base"), ("AHC + LSP", "ahc_lsp")):
        s = d.get(key)
        if s and s.get("all"):
            L.append(f"| {name} | {s['all'][4]:.2f}% | {s['all'][3]:.2f}% | {s['sr'][0]:.1f}% |")
    L += ["", f"Paired DER gain of LSP over {d['primary']['n'] if d['primary'] else '?'} recordings: {fmt_s(d['primary'], ' points')}", "",
          "| projection removing k dimensions | DER |", "|---|---|"]
    for name, c in d["controls"].items():
        L.append(f"| {name} | {c['der']:.2f}% |")
    if r.get("w3"):
        w = r["w3"]
        L += ["", "## Window check (3 s windows, 1.5 s hop)", "",
              f"Baseline DER {w['base']['all'][4]:.2f}%, LSP (k = {w['k_best']}) DER {w['lsp']['all'][4]:.2f}%; "
              f"paired gain {fmt_s(w['primary'], ' points')}"]
    return "\n".join(L) + "\n"


def main():
    d, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    res = dict(sim_dev=similarity(clean(open(f"{d}/similarity_dev.log").read())),
               sim_eval=similarity(clean(open(f"{d}/similarity_eval.log").read())),
               dia=diarize(clean(open(f"{d}/diarize.log").read())))
    if os.path.exists(f"{d}/diarize_w3.log"):
        res["w3"] = diarize(clean(open(f"{d}/diarize_w3.log").read()))
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    open(os.path.join(out, "RESULTS.md"), "w").write(markdown(res))
    print(open(os.path.join(out, "RESULTS.md")).read())


if __name__ == "__main__":
    main()
