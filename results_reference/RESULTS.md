# Results

All numbers are read from the logs in this folder. Oracle speech regions, no collar, overlap scored.

## H1: does a speaker's embedding drift with language?

| | Development | Evaluation |
|---|---|---|
| multilingual speakers | 111 | 77 |
| same-speaker cosine, same / other language | 0.337 / 0.328 | 0.330 / 0.317 |
| drift (same - other) | +0.009 (median +0.007, 95% CI [+0.008, +0.010], Wilcoxon p = 1.91e-19) | +0.013 (median +0.012, 95% CI [+0.011, +0.015], Wilcoxon p = 3.94e-14) |
| time-matched drift | +0.009 | +0.010 |
| language probe, balanced accuracy (chance) | 57.7% (50.0%) | 55.1% (44.4%) |

## H2: are there more errors at language switches? (evaluation, baseline)

| condition | share of speech | confusion |
|---|---|---|
| same speaker, same language | 48.0% | 7.76% |
| same speaker, language switch | 7.5% | 8.94% |
| speaker change, same language | 12.7% | 17.46% |
| speaker change + language switch | 1.9% | 23.45% |

Within-speaker switch minus elsewhere: +1.254 points (median -0.371, 95% CI [-1.419, +4.487], Wilcoxon p = 0.846)
Speaker change minus elsewhere: +9.053 points (median +10.350, 95% CI [+7.180, +10.743], Wilcoxon p = 9.31e-08)

## H3: language-subspace projection (rank k = 2, chosen on development data)

| system | DER | confusion | split rate |
|---|---|---|---|
| NME-SC | 29.10% | 11.71% | 2.6% |
| NME-SC + LSP | 28.38% | 10.99% | 2.6% |
| AHC | 37.13% | 19.74% | 3.9% |
| AHC + LSP | 37.96% | 20.57% | 2.6% |

Paired DER gain of LSP over 32 recordings: +0.756 points (median +0.076, 95% CI [+0.022, +2.157], Wilcoxon p = 0.0233)

| projection removing k dimensions | DER |
|---|---|
| PCA | 29.08% |
| NAP (within-speaker) | 28.07% |
| Shifted-label LSP | 29.07% |
| Random | 29.07% |

## Window check (3 s windows, 1.5 s hop)

Baseline DER 26.36%, LSP (k = 1) DER 26.12%; paired gain +0.344 points (median -0.010, 95% CI [-0.074, +1.134], Wilcoxon p = 0.36)
