# Code guide

The repository has three pieces of code:

| File | Role |
|---|---|
| `run_all.py` | Finds the DISPLACE files, then calls `cs_diarization.py` step by step with the settings of the reference run. It also skips finished steps. |
| `cs_diarization.py` | All analysis: embeddings, H1, H2, H3, controls, statistics. It is the exact file used in the reference run. |
| `tools/collect_results.py` | Reads the logs and writes `results.json` and `RESULTS.md`. |

This guide follows `cs_diarization.py` from top to bottom. The section names match the comment banners in the file.

---

## 1. RTTM input/output

* **`read_rttm_dir(d)`** reads every `*.rttm` file in a folder and returns `{session: [(start, end, label), ...]}`.
  It accepts `SPEAKER` lines (Track 1) and `LANGUAGE` lines (Track 2). Language segments labelled `?:...` are skipped, because DISPLACE uses them for unclear or non-speech segments.
* **`write_rttm`** writes diarization output in the standard RTTM format.
* **`to_annotation`** converts segments to a `pyannote.core.Annotation`. It is used only for the pyannote cross-check.

## 2. Frame helpers and the reference on a 10 ms grid

Everything that is scored is first put on a 10 ms grid (`STEP = 0.01`).

* **`activity(segs, labels, n)`** builds a boolean matrix of shape labels × frames.
* **`FrameRef`** holds one recording's reference:
  * `act`: speaker activity;
  * `nref`: number of active speakers per frame;
  * `single`: the speaker index where exactly one speaker talks, otherwise −1;
  * the switch events and the condition masks (see 3).
* **`FrameRef.score(hyp)`** computes missed speech, false alarm and confusion, overall and per region. It uses one optimal one-to-one speaker mapping for the whole recording (Hungarian algorithm on the co-occurrence matrix), so that local regions are never scored with a mapping chosen only for them. With
  `N_ref` reference speakers, `N_hyp` hypothesis speakers and `N_corr` correctly mapped speakers in a frame, the components are:
  * miss = max(0, N_ref − N_hyp);
  * false alarm = max(0, N_hyp − N_ref);
  * confusion = min(N_ref, N_hyp) − N_corr.

  This is exactly the DER definition with no collar and overlap scored. The `diarize` step checks it against `pyannote.metrics` on every evaluation file; the largest difference was 0.09 DER points.

## 3. Language switches and the four conditions (H2)

* **`switch_points`** finds consecutive language segments with different labels and a gap of at most 2 s. The switch point is the middle of the gap.
* **`switch_events`** sorts the events into four types:
  * `ws`: the same speaker before and after the switch (a within-speaker switch);
  * `sc`: a different speaker before and after (a switch at a speaker change);
  * `amb`: no clear single speaker on one side;
  * `spk`: a speaker change with no language switch nearby.

  "Speaker before/after" means the speaker who holds most single-speaker frames in the 1 s before or after the point, with at least 25% coverage.
* **`condition_masks`** turns the events into frame masks:
  * SS-DL: ±1 s around `ws` events;
  * DS-DL: ±1 s around `sc` events;
  * DS-SL: ±1 s around `spk` events;
  * SS-SL: all remaining frames.

  An event counts only if no other event lies within 1 s. Ambiguous or crowded regions are excluded.
* **`cmd_switch_errors`** scores any directory of hypothesis RTTMs in these conditions.

## 4. Windows and embeddings

* **`windows_for_file`** cuts a recording into windows (1.5 s length, 0.75 s hop). Each window gets:
  * `spk`: the dominant speaker;
  * `single`: True if one speaker covers at least 90% of the window and no other speaker more than 10%;
  * `lang`: the language if one language covers at least 50%, otherwise `UNK`;
  * `speech`: the share of speech.

  Coverage is computed in constant time from cumulative sums (`frame_cumsum`).
* **`cmd_embed`** loads NeMo's `titanet_large` and reads each session (averaged to mono, resampled to 16 kHz). It embeds all windows with at least 50% speech in batches of 64 and saves an `.npz` file with these arrays:
  * `emb`
  * `fid`
  * `start`, `end`
  * `spk` (`session|speaker`)
  * `lang`
  * `single`
  * `speech`
* **`cmd_lid`** is used only with `--pseudo-lid`. It labels 3 s chunks of every speaker turn with the SpeechBrain VoxLingua107 model and writes language RTTMs.

## 5. Statistics

* **`paired_stats(d)`** summarises one paired difference per recording (or per speaker):
  * mean and median;
  * 95% CI from 10,000 bootstrap resamples of the pairs;
  * Wilcoxon signed-rank p, with zero differences dropped;
  * matched-pairs rank-biserial correlation;
  * numbers of better, tied (|d| ≤ 0.01) and worse.
* **`holm`** applies the Holm–Bonferroni correction to the secondary tests.
* **`eer`** computes the equal error rate of cosine scores.

## 6. H1: drift and probe (`similarity`)

* **`usable(z)`** selects the windows used for analysis: those with one clear speaker and a language label.
* **`speaker_drift`** works on each speaker with at least two languages, each with at least 5 windows.
  * ρ<sub>same</sub> is the mean cosine over all pairs in the same language that do not overlap in time. Each language counts equally.
  * ρ<sub>cross</sub> is the same over pairs in different languages.
  * The time-matched drift compares same- and cross-language pairs inside the same time-distance bin ([L,10), [10,60), ≥ 60 s), so slow drift of the channel or voice cannot explain it.

  Speakers with more than 2,500 windows are subsampled (seed 0).
* **`language_probe`** works inside each recording, because DISPLACE language labels are relative to the recording.
  * A logistic regression (C = 1, balanced class weights) is trained on all speakers but one and tested on the held-out speaker.
  * Test windows of a language that no other speaker uses are skipped.
  * Balanced accuracy is averaged over recordings and compared with chance (1/number of languages).

## 7. Subspace estimators (H3 and controls)

* **`language_offsets`** computes, for each multilingual speaker, μ<sub>s,ℓ</sub> − mean<sub>ℓ</sub> μ<sub>s,ℓ</sub> on unit-length embeddings. A speaker with q languages contributes q offsets that sum to zero.
* **`basis_from_offsets`** returns the top-k right singular vectors and all singular values.
* **`project(E, U)`** maps unit embeddings to (I − UUᵀ)e, then renormalises. This is LSP.
* The controls remove the same number of dimensions without using language:
  * **`pca_basis`**: the leading principal directions of all analysis windows;
  * **`nap_basis`**: conventional NAP, the leading directions of all within-speaker variability;
  * **`shifted_offsets`**: the LSP estimator after circularly shifting each speaker's time-ordered language labels, which keeps label counts and run lengths but breaks their link to the true language;
  * **`random_basis`**: random orthonormal directions.
* **`subspace_report`** prints:
  * the number of offsets and the rank bound;
  * the singular values;
  * ε<sub>k</sub>, the share of offset energy removed;
  * η<sub>k</sub>, the share of between-speaker energy kept, over speaker pairs of the same recording;
  * the subspace stability, ‖V<sub>k</sub>ᵀV<sub>k</sub><sup>(b)</sup>‖²/k over 200 bootstrap resamples of speakers.

## 8. Clustering and scoring

* **`nme_sc`** is spectral clustering with the normalized maximum eigengap rule (Park et al., 2020).
  * It searches p over 1–25% of the windows, binarises the top-p affinity, and estimates the number of speakers (at most 8) from the largest eigengap.
  * It then clusters the eigenvectors with k-means (seed 0).
  * It uses the GPU for the eigen-decomposition when one is available.
* **`ahc_tree` / `ahc_cut`** are average-linkage agglomerative clustering on cosine distance, cut at a threshold.
* **`hyp_labels`** gives each window the `hop` seconds around its centre and crops the result to the reference speech (the oracle speech-activity condition).
* **`split_and_change`** computes three things:
  * the split rate: a multilingual speaker is split if the majority cluster of their windows in some language differs from their overall majority cluster;
  * the shifted-label control: the same statistic after 20 random circular shifts of the labels;
  * the cluster-change rate between adjacent windows of one speaker, across a language switch and within one language.
* **`run_system`** clusters every recording with one projection and one back end, and scores it. With an output folder it also writes RTTMs and checks the scorer against pyannote.

## 9. The `diarize` command, in order

1. Print the environment and versions, the reference speaker counts, the window statistics and the switch statistics.
2. Estimate the language subspace on dev and print the subspace report.
3. **Rank selection on dev only**: NME-SC for k = 0 and every k in the grid. The rule is the lowest dev DER, with ties going to the smaller k.
4. **Evaluation**: the baseline (k = 0) and LSP (k*), with RTTMs written to `runs/eval_k0` and `runs/eval_k{k*}`. It prints per-condition errors, the split rate and the cluster-change rates.
5. **Primary test**, fixed in advance: the paired DER difference between baseline and LSP over the evaluation recordings.
6. **H2 tests** on the baseline: SS-DL − SS-SL and DS-SL − SS-SL confusion, for recordings with at least 30 s in the region.
7. **Controls** at the same k: PCA, NAP, shifted-label LSP (3 seeds) and random (3 seeds).
8. **AHC**: the threshold is tuned on dev separately for k = 0 and k*, then applied to eval.
9. **Secondary comparisons** with Holm-corrected p.
10. **Speaker information**, using within-recording, non-overlapping window pairs. For each projection it reports:
    * the mean cosine of the four pair types;
    * the EER for all target pairs and for cross-language target pairs;
    * the drift;
    * the language probe.
11. **Gain against code-switching rate**: the Spearman correlation and thirds of recordings.
12. The smallest projected norm, the runtime, and `summary.json`.

## 10. Reproducibility notes

* All random choices use fixed seeds: subsampling, bootstrap, trial sampling, k-means, the shifted labels and the random subspaces.
* Results can differ slightly across library versions and GPUs, because eigen-decomposition and k-means are numerically sensitive. The `results_reference/` folder holds the complete logs of the reference run for comparison.
* `run_all.py` sets `PYTHONWARNINGS=ignore`, so library warnings do not mix with the logs. It also runs every step with unbuffered output, so a log is complete even if a run is interrupted.
