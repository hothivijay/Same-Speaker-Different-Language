# Same speaker, different language: language drift of speaker embeddings in code-switched diarization (DISPLACE 2024)

This repository contains the complete code used for the experiments on the DISPLACE 2024 corpus.
Given only the location of the DISPLACE 2024 data, one command reproduces every number:

```bash
python run_all.py --displace /path/to/DISPLACE --out ncc_output
```

It works on a local computer and in Google Colab ([`colab/run_in_colab.ipynb`](colab/run_in_colab.ipynb)).
The results of our own run are in [`results_reference/`](results_reference/RESULTS.md) for comparison.

---

## 1. What the experiments test

In code-switched conversation, one person often changes language within a turn. Speaker diarization
clusters speaker embeddings and assumes that an embedding changes only when the *speaker* changes.
We test three hypotheses with TitaNet-Large embeddings on DISPLACE 2024:

| | Hypothesis | How it is tested |
|---|---|---|
| **H1** | Speaker embeddings carry language information, so one speaker looks less like themselves across languages. | Same-speaker cosine similarity within vs. across languages (exact, over all non-overlapping window pairs; also time-matched), and a linear language probe tested on held-out speakers. |
| **H2** | This causes diarization errors at language switches of the *same* speaker. | Confusion in four conditions (same/different speaker × same/different language) under one global speaker mapping; split rate with a shifted-label control; cluster-change rates. |
| **H3** | Removing a low-rank language subspace reduces these errors without losing speaker information. | Language-subspace projection (LSP) before clustering, rank chosen on dev only; compared with PCA, conventional NAP, shifted-label LSP and random projections of the same rank; two back ends (NME-SC, AHC); EER of window pairs; paired statistics over recordings. |

**Language-subspace projection (LSP).** For every development speaker *s* with at least two languages,
take the mean unit-length embedding in each language, μ<sub>s,ℓ</sub>, and its offset from the speaker's
centre, δ<sub>s,ℓ</sub> = μ<sub>s,ℓ</sub> − mean<sub>ℓ</sub>(μ<sub>s,ℓ</sub>). The top-*k* right singular vectors
**V**<sub>k</sub> of all offsets span the language subspace, and every embedding is mapped to
(**I** − **V**<sub>k</sub>**V**<sub>k</sub><sup>T</sup>)**e**, then renormalised. No network is trained; language
labels are needed only on the development set.

**Main findings of our run** (see [`results_reference/RESULTS.md`](results_reference/RESULTS.md)):
H1 holds but the effect is small (same-speaker cosine 0.013 lower across languages on eval; probe
55.1% balanced accuracy vs. 44.4% chance). H2 does not hold (confusion 8.9% at within-speaker switches vs.
7.8% elsewhere, p = 0.85; speaker changes add 9.1 points). LSP lowers NME-SC DER from 29.10% to 28.38%
(p = 0.023), but conventional NAP of the same rank reaches 28.07%, AHC does not benefit, and with 3 s
windows the gain is not significant.

---

## 2. Data: DISPLACE 2024

The data are **not** included and must not be uploaded here: the DISPLACE terms require the organisers'
consent for any redistribution. Register at <https://displace2024.github.io/> to obtain the release.

You need, for the **development** and **evaluation** sets:

* the session audio (`.wav` or `.flac`; file name = session id, e.g. `M043.wav`),
* the Track-1 speaker RTTMs (`SPEAKER M043 1 1.398 1.707 <NA> <NA> S1 <NA> <NA>`),
* the Track-2 language RTTMs (`LANGUAGE M043 1 1.398 1.707 <NA> <NA> L2 <NA> <NA>`).

Unpack them anywhere under one folder. `run_all.py` finds the files itself:

* an RTTM is a **language** file if its lines start with `LANGUAGE` or its path contains `track2`,
  `_ld`, `/ld` or `lang`; otherwise a **speaker** file (`track1`, `_sd`, `speaker`, or `SPEAKER` lines);
* it belongs to **eval** if its path contains `eval`, otherwise to **dev** if the path contains `dev`;
* folders whose path contains `sample` or `checkpoint` are ignored (sample clips, notebook checkpoints);
* audio is matched to the RTTMs by session id, wherever it is stored;
* if a package has no evaluation labels, every second development recording is held out as "eval";
* if there are no language RTTMs, `--pseudo-lid` creates them with the VoxLingua107 language-ID model.

Two properties of the DISPLACE language references are handled by the code: segments labelled `?:...`
(unclear or non-speech) are ignored, and the labels `L1`, `L2`, `L3` are **relative to each recording**
(L1 is not the same language in every session). All language comparisons are therefore made within a
recording (speaker offsets, the language probe and the switch analysis).

The layout of our copy, for reference (any other layout works too):

```
DISPLACE/
├── Track1_SD_supervised/
│   ├── Dev/ Track1_SD/*.wav   Track1_SD_labels/*_SPEAKER.rttm
│   └── Eval/ Track1_SD/*.wav  SD_eval_rttms/*_SPEAKER.rttm
└── track2_LD_supervised/
    ├── Dev/ Track2_LD/*.wav   Track2_LD_labels/*_LANGUAGE.rttm
    └── Eval/ Track2_LD/*.wav  LD_eval_rttms/*_LANGUAGE.rttm
```

---

## 3. How to run

### Option A: Google Colab (no installation)

1. Put the DISPLACE folder in your Google Drive.
2. Open [`colab/run_in_colab.ipynb`](colab/run_in_colab.ipynb) in Colab
   (`File → Open notebook → GitHub`, paste this repository's URL) and choose `Runtime → Change runtime type → T4 GPU`.
3. In cell 1, set `DISPLACE_DIR` (for example `/content/drive/MyDrive/DISPLACE`) and `OUTPUT_DIR`.
4. Run all cells. The summary appears at the end and is saved in `OUTPUT_DIR/results/RESULTS.md`.

Colab can only mount the Drive of the account the notebook runs under; open Colab with the account that
holds the data (or share the folder and add a shortcut to it in *My Drive*).

### Option B: local computer (Linux, macOS or Windows)

Python 3.10 or newer; an NVIDIA GPU is strongly recommended for the embedding step.

```bash
git clone https://github.com/hothivijay/Same-Speaker-Different-Language.git
cd Same-Speaker-Different-Language
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # install a CUDA build of PyTorch first if needed
python run_all.py --displace /path/to/DISPLACE --out ncc_output
```

### Options of `run_all.py`

| Option | Meaning | Default |
|---|---|---|
| `--displace PATH` | folder of the unpacked DISPLACE 2024 release | required |
| `--out PATH` | logs, embeddings, RTTMs, results | `ncc_output` |
| `--work PATH` | fast scratch folder for the audio links (use a local disk when `--out` is on a network drive) | same as `--out` |
| `--copy-audio` | copy audio instead of linking it (e.g. if the file system has no symbolic links) | off |
| `--stride N` | cluster every N-th window (2 is the setting of the reference run) | 2 |
| `--window-check` | also repeat the analysis with 3 s windows and 1.5 s hop | off |
| `--pseudo-lid` | create language labels with a language-ID model if Track-2 RTTMs are missing | off |
| `--force` | recompute steps whose output already exists | off |

A run that stops (crash, Colab disconnect) can be started again with the same command: finished steps are skipped.

---

## 4. What `run_all.py` does, step by step

| Step | Command it runs | Output | Time on a T4 |
|---|---|---|---|
| 1. organise | (inside `run_all.py`) | `OUT/labels/{dev,eval}_{sd,ld}/`, `WORK/audio/` | < 1 min (links) |
| 2. embed | `cs_diarization.py embed` for dev and eval | `OUT/results/{dev,eval}.npz`, `OUT/logs/embed_*.log` | ≈ 5 min per set |
| 3. similarity (H1) | `cs_diarization.py similarity` | `OUT/logs/similarity_{dev,eval}.log` | ≈ 1 min |
| 4. diarize (H2, H3) | `cs_diarization.py diarize --stride 2` | `OUT/logs/diarize.log`, `OUT/results/runs/` | ≈ 15 min |
| 5. window check (optional) | `embed --win 3.0 --hop 1.5`, `diarize --hop 1.5 ...` | `OUT/logs/diarize_w3.log`, `OUT/results/runs_w3/` | ≈ 25 min |
| 6. collect | `tools/collect_results.py` | `OUT/results/results.json`, `OUT/results/RESULTS.md` | seconds |

Without a GPU, step 2 takes considerably longer; all other steps run on a CPU.

### Output files

| File | Content |
|---|---|
| `results/dev.npz`, `eval.npz` | one row per window: `emb` (192-d TitaNet-L embedding), `fid` (session), `start`, `end` (s), `spk` (`session|speaker`), `lang` (`L1`/`L2`/… or `UNK`), `single` (one speaker covers ≥ 90%, others ≤ 10%), `speech` (share of speech) |
| `results/runs/eval_k0/*.rttm` | baseline diarization of the evaluation set (NME-SC, oracle speech) |
| `results/runs/eval_k<k>/*.rttm` | the same with LSP, k chosen on dev |
| `results/runs/summary.json` | every number of the diarize step (dev rank selection, subspace, eval scores, statistics, controls, AHC, speaker information, per-recording DER) |
| `results/RESULTS.md` | the headline numbers in readable form |
| `logs/*.log` | full console output of every step, including versions and hardware |

---

## 5. Settings used (all are defaults of the code)

| Item | Setting |
|---|---|
| Embedding extractor | NeMo `titanet_large` (frozen), 192-d; mono, 16 kHz |
| Windows | length 1.5 s, hop 0.75 s (window check: 3.0 s / 1.5 s) |
| Window labels | speaker if one speaker covers ≥ 90% and no other > 10%; language if one language covers ≥ 50% |
| Multilingual speaker | ≥ 5 such windows in each of ≥ 2 languages |
| Similarity (H1) | all pairs of a speaker's windows with \|Δt\| ≥ window length; languages weighted equally; time-matched bins [L,10), [10,60), ≥ 60 s |
| Language probe | logistic regression, C = 1, balanced class weights; leave-one-speaker-out inside each recording |
| Switch analysis (H2) | switch = change of reference language with gap ≤ 2 s; speaker on each side = majority single speaker in 1 s (≥ 25% coverage); regions ±1 s; events with another event within 1 s excluded |
| Scoring | DER, no collar, overlap scored, oracle speech regions, single label per frame; 10 ms frames, one Hungarian mapping per recording (checked against pyannote.metrics) |
| Back ends | NME-SC (≤ 8 speakers); AHC (average linkage, cosine; threshold tuned on dev from 0.30–1.30) |
| LSP | m = 5 windows per speaker–language pair; k ∈ {1, 2, 3, 4, 6, 8, 12, 16} (up to the offset rank); rule: lowest dev DER, ties to smaller k |
| Controls (same k) | PCA; conventional NAP (within-speaker scatter); shifted-label LSP (3 seeds); random subspace (3 seeds) |
| Statistics | paired over recordings: mean, median, 10,000-sample paired bootstrap CI, Wilcoxon signed-rank (zeros dropped), rank-biserial r; Holm correction for secondary tests |
| Randomness | all random steps use seed 0 (controls: seeds 0–2) |

The code is explained function by function in [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md).

---

## 6. Expected results

Our reference run (Google Colab, Tesla T4, Python 3.13.16, nemo_toolkit 3.0.0, torch 2.11.0, numpy 2.2.6,
scipy 1.16.3, scikit-learn 1.6.1, pyannote.metrics 4.2.0) gave:

| | Value |
|---|---|
| Dev / eval windows | 88,811 / 81,457 |
| Same-speaker drift across languages (eval) | +0.013 cosine (95% CI 0.011–0.015) |
| Language probe (eval) | 55.1% balanced accuracy (chance 44.4%) |
| Confusion: within-speaker switch vs. elsewhere (eval baseline) | 8.94% vs. 7.76% (p = 0.85) |
| Rank chosen on dev | k = 2 |
| Eval DER, NME-SC: baseline → LSP | 29.10% → 28.38% (paired p = 0.023) |
| Eval DER, NME-SC + NAP (same rank) | 28.07% |
| Eval DER, AHC: baseline → LSP | 37.13% → 37.96% |
| 3 s windows, NME-SC: baseline → LSP (k = 1) | 26.36% → 26.12% (p = 0.36) |

Small differences can appear with other library versions or GPUs (eigen-solvers and k-means are
numerically sensitive), but the conclusions should not change.

---

## 7. Running single steps

```bash
python cs_diarization.py --help
python cs_diarization.py embed --audio-dir AUDIO --sd-ref DEV_SD --ld-ref DEV_LD --out dev.npz
python cs_diarization.py similarity --emb dev.npz
python cs_diarization.py diarize --dev-emb dev.npz --eval-emb eval.npz \
    --dev-sd-ref DEV_SD --eval-sd-ref EVAL_SD --dev-ld-ref DEV_LD --eval-ld-ref EVAL_LD --stride 2 --out runs
python cs_diarization.py switch-errors --sd-ref EVAL_SD --ld-ref EVAL_LD --hyp ANY_SYSTEM_RTTM_DIR
```

`switch-errors` scores the output of any diarization system (for example your own) in the four
conditions of H2.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `no development speaker RTTMs found` | `--displace` points to the wrong folder, or the paths contain neither `dev` nor `eval`. |
| `no Track-2 language RTTMs found` | add the language RTTMs, or run with `--pseudo-lid` (then the language labels are automatic). |
| Colab: `credential propagation was unsuccessful` | the Drive you tried to mount belongs to another account; open Colab with the account that owns the data. |
| Colab disconnected | run the cells again; finished steps are skipped. |
| `--window-check` takes too long | it is optional; the main results do not need it. |
| No GPU | everything works, but the embedding step is much slower. |

---

## 9. Citation and acknowledgement of resources

If you use this code, please cite the accompanying paper (reference to be added) and the resources it builds on:

* S. B. Kalluri *et al.*, "The second DISPLACE challenge: DIarization of SPeaker and LAnguage in Conversational Environments," arXiv:2406.09494, 2024.
* N. R. Koluguri, T. Park, B. Ginsburg, "TitaNet: Neural model for speaker representation with 1D depth-wise separable convolutions and global context," ICASSP 2022.
* T. J. Park, K. J. Han, M. Kumar, S. Narayanan, "Auto-tuning spectral clustering for speaker diarization using normalized maximum eigengap," IEEE Signal Processing Letters, 2020.
