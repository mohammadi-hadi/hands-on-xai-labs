# T4 verification checklist: all 13 notebooks on a clean Colab free-tier T4

Purpose: verify every chapter notebook runs end-to-end on a fresh free-tier Colab T4 and
produces numbers consistent with the ledgers in `experiments/`. Budget: **1–2 hours** for one
sitting (estimated total wall time across all 13 notebooks: ~60–80 min of Run-all, plus
restarts and note-taking).

Colab cannot be driven from the build machine. This is a human protocol. Fill every
"measured" blank; copy the completed file back into `experiments/` when done.

## Session protocol (repeat steps 4–9 per notebook)

1. **Fresh runtime.** colab.research.google.com → New notebook → Runtime → Change runtime type
   → **T4 GPU** → Save. If you were connected before, Runtime → Disconnect and delete runtime
   first.
2. **GPU check cell.** Run `!nvidia-smi` to confirm "Tesla T4" and ~15 GB memory. If you got a
   different GPU, results still count but note it in the measured blank.
3. **Pin-install check.** Each notebook's first code cell does its own pinned `%pip install`
   under `IN_COLAB`. Do NOT pre-install anything else. If Colab prompts "Restart runtime"
   after the install (the `numpy==1.26.4` downgrade), restart and Run-all again from the top;
   count only the post-restart run's wall time.
4. **Open the notebook.** Upload the `.ipynb` (File → Upload notebook) or open via the Colab
   badge/GitHub path. Do not set any `BOOK_*` env vars, because full mode is the default.
5. **Run all.** Runtime → Run all. Note the clock time when you press it.
6. **Record wall time** (clock at last cell finishing minus clock at Run-all, minus any
   restart-and-rerun overhead already counted once).
7. **Record the key numbers** listed in that notebook's section below, straight from the cell
   outputs. Screenshot or copy the relevant output cells if a number is out of tolerance.
8. **Mark PASS/FAIL** against the criteria. A notebook that completes Run-all with no cell
   error but misses a tolerance band = "PASS (numbers flagged)". Note which number.
9. **Disconnect and delete runtime** (Runtime menu) before the next notebook. Never reuse a
   runtime between notebooks: pin sets differ per chapter and stale installs contaminate the
   next run.
10. At the end: fill the summary table at the bottom.

Tolerance conventions used below (seeds are fixed, but GPU kernels and GPU training are not
bit-deterministic):

- **exact**: counts, ranks, and identities (which head, which layer, which token) must match.
- **±0.02–0.05**: accuracies, F1s, correlations from inference-only or CPU-deterministic code.
- **±0.05–0.15**: metrics downstream of GPU training (Colab trains its own models in ch04–06,
  ch08, ch12; the ledger numbers came from MPS/CPU runs).
- **same sign + same order of magnitude**: effect sizes, logit diffs, coefficients.

Cross-cutting known flakiness (applies to several notebooks):

- **HF download hiccups**: transient 429/5xx from the Hub. Re-run the failed cell once before
  calling it a failure.
- **`tweet_eval` loading**: occasionally slow or briefly unavailable via `datasets`; ch04–06,
  ch08, ch10 carry small inline fallbacks that keep cells alive but change all numbers. If the
  output says the fallback was used, re-run later rather than comparing numbers.
- **`models/ch04-specimen` is absent on Colab** (268 MB, not shipped): ch05/06/08 quick-train a
  `prajjwal1/bert-mini` fallback by design. Their ledger numbers describe the DistilBERT
  specimen, so on Colab compare *qualitative* PASS criteria, not exact values.
- **Gated models**: only the optional Gemma Scope cell in ch10 (`RUN_GEMMA = True` + HF token);
  it is OFF by default and stays out of this protocol.
- **numpy restart prompt**: expected wherever `numpy==1.26.4` is pinned (ch01 at minimum).

---

## ch01-why-explainability.ipynb

- Requires: core (no torch; runs on CPU, GPU unused)
- Expected wall time: **< 1 min** compute (ledger: 4.7 s local CPU) + install cell; restart
  prompt likely (numpy pin).
- Key numbers (ledger `ch01-results.md`; sklearn CPU with fixed seed, expect near-exact):
  1. LR accuracy iid / shifted: **0.950 / 0.515** (±0.01)
  2. GBT accuracy iid / shifted: **0.955 / 0.517** (±0.01)
  3. LR coefficients *romance* / *horror*: **+10.64 / −10.62** (same sign, ±0.5)
  4. Oracle (decorrelated retrain) shifted accuracy: **0.827** (±0.02)
- Known flakiness: numpy-downgrade restart prompt; nothing else (fully synthetic data).
- PASS: Run-all clean; both models ~0.95 iid collapsing to ~0.52 shifted; cue coefficients
  ~4x any sentiment word.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch02-map-of-the-territory.ipynb

- Requires: llm
- Expected wall time: **measure and record** (ledger `ch02-results.md`: 22.1 s compute on CPU;
  Colab adds the model download, so expect a few minutes).
- Key numbers (ledger `ch02-results.md`; CPU-pinned, so expect near-exact):
  1. P(positive) on the probe sentence: **0.997** (±0.005)
  2. Top-ranked word per method: LIME *saved* (+0.366), SHAP *the* (+0.322), IG *but* (+0.304),
     attention *saved* (0.430). **Only 2 of 4 methods agree on the top word** (exact: the
     disagreement is the chapter's whole point; flag it if they all agree)
  3. Method 5 emits a fluent, correct-label self-explanation naming *dull* and *saved*
- Known flakiness: HF model download; note the install cell is commented out in this notebook
  (relies on Colab stock + already-compatible versions). If imports fail, install the pinned
  versions from `requirements-colab.txt` manually and note it.
- PASS: Run-all clean, all five explanation sections produce output for the same prediction.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch03-inside-the-transformer.ipynb

- Requires: llm
- Expected wall time: **2–3 min** (ledger: pip + ~550 MB GPT-2 download dominate; compute
  6.6 s local CPU). Notebook computes on CPU even with a GPU present, so numbers stay near-exact.
- Key numbers (ledger `ch03-results.md`):
  1. Token counts GPT-2/BERT/Qwen for the probe sentence: **18 / 19 / 19** (exact);
     "unexplainability" splits **3 / 5 / 3** pieces (exact)
  2. Anisotropy (mean pairwise cosine, 1,000 random embeddings): **0.269** (±0.005)
  3. Layer-5 head 10 it-to-cat attention: **0.448** (±0.01); it is the only head whose
     strongest key is ` cat` (exact identity)
  4. Logit lens p(` Amsterdam`): layer 10 **0.209**, layer 11 peak **0.347**, final demoted to
     rank 3 at **0.038** with ` the` top-1 at **0.108** (±0.01 each; ranks exact)
  5. Residual norm growth ~**119x** (6.2 to 740 to 186 after ln_f; same magnitude)
- Known flakiness: three tokenizer downloads + GPT-2 weights (HF hiccups).
- PASS: Run-all clean; all five numbers in band; Amsterdam surfaces at layer 10 and is
  overruled at the output (the chapter's centerpiece must reproduce).
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch04-feature-attribution.ipynb

- Requires: llm
- Expected wall time: **5–10 min** (ledger estimate: fine-tune 1–2 min on T4 + explanations
  ~1 min + pip 2–3 min; local MPS full run 50.7 s).
- Key numbers (ledger `ch04-results.md`; GPU fine-tune, so training tolerances apply):
  1. Specimen validation accuracy (500 tweets): **0.676** (±0.05, GPU training nondeterminism)
  2. Held-out test: accuracy **0.460**, predicts hate **~96%** (accuracy ±0.05; the huge
     val-to-test collapse and near-total hate prediction must reproduce qualitatively)
  3. Kendall τ between methods (10 tweets): LIME–SHAP **0.59**, LIME–IG **0.43**, SHAP–IG
     **0.51** (each ±0.15; all three must stay well below 1.0 and above ~0.2)
  4. IG convergence delta (max over 10 tweets): **≤ ~0.05** (ledger 0.022)
  5. SHAP cost curve flattens between 512 and 1024 evals (qualitative: plateau visible)
- Known flakiness: `tweet_eval/hate` load; distilbert download (~268 MB); inline 12-tweet
  fallback fires if the dataset fails. Numbers are then not comparable.
- PASS: Run-all clean; val/test collapse reproduces; τ values show partial (not total)
  agreement; triptych figure renders for the showcase tweet.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch05-attention.ipynb

- Requires: llm
- Expected wall time: **~5 min** (pip 2–3 min + bert-mini quick-train 1–2 min + analysis
  < 1 min).
- Fallback caveat (expected on Colab): `models/ch04-specimen` is absent, so it quick-trains `prajjwal1/bert-mini`
  (4 layers × 4 heads, not 6 × 12). **All ledger values shift**; compare
  qualitatively. Ledger reference values (DistilBERT specimen, `ch05-results.md`):
  aggregation trap ρ down to −0.21; attention-vs-|IG| mean ρ **0.01 ± 0.40**; head-ablation
  label flips **0/8**; special-token mass ~32%.
- Key checks on Colab (qualitative bands):
  1. Attention-vs-|IG| mean Spearman ρ: **near 0** (|mean| ≤ 0.25) with large per-tweet spread
  2. Single loudest-head ablation: prediction essentially unchanged (mean |Δp| ≤ 0.05)
  3. Half-the-heads ablation: **0 label flips** expected (flag if > 1 of 8 flips)
  4. Layer-choice aggregation disagreement visible (some cross-layer ρ ≤ ~0.3 or negative)
- Known flakiness: tweet_eval load; quick-train adds nondeterminism; BertViz head view renders
  on Colab (interactive cell: just confirm it displays).
- PASS: Run-all clean; verdict survives ablation; corroboration with IG ~zero on average.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch06-probing-counterfactuals-concepts.ipynb

- Requires: llm
- Expected wall time: **5–10 min** (pip + bert-mini quick-train fallback; local CPU full run
  15.0 s).
- Fallback caveat: same as ch05. The bert-mini fallback gives 5 hidden-state layers and different probe
  values. Ledger reference (DistilBERT, `ch06-results.md`): task probe peak **0.720 @ L3** vs
  model's own 0.676; length probe peak **0.907 @ L2**; CF flips **5/6** at mean 3.0 edits;
  concept-direction AUC **0.773**.
- Key checks on Colab (qualitative):
  1. Layer-0 probe = majority baseline (exact structural fact: `[CLS]` at the embedding layer
     is input-independent; must hold on any BERT-family model)
  2. Task probe peaks above the model's own validation accuracy at some non-final layer
     (represented > used)
  3. Counterfactual search flips a majority of the selected tweets within the 5-edit budget,
     with at least one honest non-flip allowed
  4. Concept-direction (mean-difference) AUC within ~0.05 of the trained probe's accuracy at
     the same layer
- Known flakiness: tweet_eval load; greedy CF paths can differ across hardware when candidate
  edits tie within float noise (expected behavior, so check trajectories, not identical paths).
- PASS: Run-all clean; the four qualitative checks hold.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch07-build-it-break-it.ipynb

- Requires: llm (metadata; content is sklearn/SHAP/LIME on synthetic text, CPU-only)
- Expected wall time: **measure and record** (ledger `ch07-results.md`: 54.4 s on CPU; treat
  1-2 min compute as the band, no downloads).
- Key numbers (ledger `ch07-results.md`; CPU, sklearn, seeded):
  1. Detector held-out accuracy: **0.972** (±0.02)
  2. Machine documents caught to start: **85/90** (±2; this is the attack's denominator)
  3. Flip rates at a 2-word budget: LIME **95%**, SHAP **95%**, random **15%** (the six-fold
     explanation-vs-random gap must reproduce; exact percentages may move a few points)
  4. Mean P(machine) when each of the four tells is present: **1.00** (exact)
- Known flakiness: install cell is commented out (Colab stock versions), same note as ch02.
- PASS: Run-all clean; the detector demonstrably breaks under the constructed attack
  (large accuracy drop), and the explanation section identifies the exploited features.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch08-evaluating-explanations.ipynb

- Requires: llm
- Expected wall time: **5–10 min** (ledger estimate: attribution 1–2 min on GPU, judge < 1 min,
  pip 2–3 min, ~1 GB downloads incl. Qwen-0.5B; local CPU full run 134.9 s).
- Fallback caveat: bert-mini quick-train specimen on Colab, so attribution/faithfulness values
  shift; the *structure* of the result is the PASS bar. Ledger reference (`ch08-results.md`):
  deletion-AUC LIME **0.495** / SHAP **0.526** / IG **0.543** / random **0.678**; plausibility
  F1 0.52/0.50/0.46 vs 0.17 random.
- Key checks:
  1. Deletion-AUC: all three methods beat random with **non-overlapping bootstrap CIs**, and
     the three method-vs-method marginal CIs **overlap pairwise** (structural result; must
     reproduce). Paired-difference cell: LIME−IG **separates** (−0.048 [−0.072, −0.027]),
     LIME−SHAP and SHAP−IG overlap zero, attention−random overlaps zero (structural result on
     the real specimen; stand-in values shift)
  2. Judge audit (Qwen2.5-0.5B, greedy): position-following **16/16 pairs**, slot-A verdicts
     **32/32** (exact counts expected under greedy decoding; flag if even 1–2 pairs deviate, note
     GPU float noise as the likely cause and record the count)
  3. Sufficiency@5 goes **negative** for at least one of LIME/SHAP (ledger −0.068 / −0.047)
  4. Random plausibility floor clearly nonzero but far below the methods'
- Known flakiness: tweet_eval + Qwen-0.5B downloads; one annotated tweet is skipped by design
  (prints the skip), not a failure.
- PASS: Run-all clean; check 1 and the judge-wipeout headline hold.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch09-cot-faithfulness.ipynb

- Requires: llm
- Expected wall time: **5–10 min** (ledger estimate: ~1 GB Qwen download + pip ~2 min +
  generation 3–6 min fp16 batch 16; local MPS full run 122.0 s).
- Key numbers (ledger `ch09-results.md`; n=32, greedy, but T4 fp16 kernels ≠ MPS fp16, so
  allow small count shifts):
  1. Greedy accuracy: **0.59** (±0.06, i.e. ±2 items of 32)
  2. Early-answer consistency: **0.00 at 25% and 50%** truncation (allow ≤ 0.06), **≥ 0.90 at
     100%** (ledger 0.97)
  3. Mistake-injection sensitivity: **0.83** (±0.10; must stay clearly high, ≥ 0.7)
  4. Hint-flip rate: **0.03 (1/31)** (allow 0–3 flips; if flips occur, check whether they are
     silent, as the ledger's single flip was)
  5. Parseable answers: **31/32** (allow ±1; if several answers are unparseable, the tiered
     extractor is misbehaving, so investigate before trusting any metric)
- Known flakiness: GSM8K download (inline 10-item fallback fires offline, so the numbers are not
  comparable); side effect by design: writes `faithfulness_metrics.py` next to the notebook.
- PASS: Run-all clean; CoT is load-bearing (early-answer floor + high sensitivity) and the
  hint channel is measured with sane counts.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch10-mechanistic-interpretability.ipynb

- Requires: mechinterp
- Expected wall time: **6–9 min** (ledger: pip 3–5 min for transformer-lens + sae-lens; GPT-2
  ~550 MB + SAE ~330 MB downloads; compute < 1 min on GPU).
- Key numbers (ledger `ch10-results.md`; DEVICE auto-detects cuda on Colab, making small float
  drift possible vs the CPU-produced ledger):
  1. Induction heads: **L5H5 top at ~0.945**, the classic five (L5H5, L6H9, L7H10, L5H1, L7H2)
     all > 0.85 (head identities exact; scores ±0.02)
  2. Patching: clean logit diff **+2.735**, corrupted **−1.347** (±0.05); max recovery **~0.89
     at layer 11, final position** (location exact; value ±0.05); clean top-1 is ` London`
     with ` Paris` rank 2 (exact: the "shaky geographer" gem)
  3. SAE L0: **~45–60 active features per token** (band)
  4. Steering α=20: p(` San`) rises **0.022 to ~0.077** and reaches top-1 (direction + rank
     exact; values ±0.02); α=80 overshoots to ` Diego`/` Francisco` (qualitative)
  5. Exercise patch: max recovery **~0.94 at layer 0, position ' Wall'** (location exact)
- Known flakiness: sae-lens `model_from_pretrained_kwargs` UserWarning is expected and
  harmless; SAE download is the biggest HF-hiccup surface; `tweet_eval/sentiment` for the
  dashboard (inline fallback exists); do NOT set `RUN_GEMMA = True` for this protocol.
- PASS: Run-all clean; head identities, patch geometry, and steering direction all reproduce.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch11-auditing-behavior.ipynb

- Requires: llm
- Expected wall time: **3–5 min** (ledger: ~1.4 GB model downloads dominate; compute < 1 min.
  Scoring deliberately stays on fp32 CPU for determinism, so values should match closely).
- Key numbers (ledger `ch11-results.md`):
  1. Generative compliance: Qwen **1.00**, SmolLM2 **1.00**, gpt2 **0.16 (3/19)** (exact counts)
  2. Degenerate answers: Qwen outputs "7" on **13/19** items; SmolLM2 outputs "1" on **all 19**
     (exact, greedy CPU)
  3. gpt2 log-prob Pearson r: Netherlands **0.61**, Iran **−0.22** (±0.03)
  4. W.E.I.R.D. gap (log-prob): gpt2 **+0.65**, SmolLM2 **−0.33** (±0.05; signs exact)
  5. Qwen log-prob vs generative Spearman ρ: **−0.18** (±0.05; must not be strongly positive)
- Known flakiness: three model downloads (Qwen-0.5B, SmolLM2-360M, gpt2).
- PASS: Run-all clean; counts exact; correlations in band; both figures render.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch12-training-for-explainability.ipynb

- Requires: llm
- Expected wall time: **< 5 min** (ledger estimate: pip ~1 min, distilgpt2 ~350 MB,
  SFT+DPO+battery < 2 min on GPU; local MPS full run 30.6 s).
- Key numbers (ledger `ch12-results.md`; **GPU training: widest tolerances in this protocol**;
  the ledger itself records 0/20 vs 5/20 acknowledged flips between MPS and CPU at the same
  seed):
  1. Base model hint-flip / silent-flip: **0.70** (±0.15; must be clearly high, ≥ 0.5)
  2. SFT flip rate: between base and DPO (ledger 0.40; band 0.2–0.6); SFT acknowledgment
     **0.00** (greedy modal collapse; allow ≤ 0.1)
  3. DPO acknowledgment: **1.00** (allow ≥ 0.9); DPO silent-flip: **0.00** (allow ≤ 0.10)
  4. Accuracy pinned at **0.00** in all three columns (exact: the pairs never contrasted
     arithmetic; any nonzero accuracy is surprising, so record it)
  5. Phantom-ack on plain prompts: **0.00** everywhere (allow ≤ 0.05)
- Known flakiness: adapter cell prints instructions and skips without `BOOK_CH12_ADAPTER`
  (this is expected, not a failure). Read one DPO transcript to confirm "verification theater" is
  visible (says "let me verify", math still wrong).
- PASS: Run-all clean; the training moved exactly the contrasted metrics (ack up, silent
  flips down) while accuracy stayed at 0.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

## ch13-explainability-in-production.ipynb

- Requires: llm
- Expected wall time: **4–8 min** (ledger: ~1.1 GB downloads; ~2,600 generated tokens run on CPU by design; local full run 31.4 s).
- Key numbers (ledger `ch13-results.md`; greedy but borderline generations may flip ±1
  question on different hardware):
  1. Citation-lab answer accuracy: **0.88 (7/8)** (allow ±1 question, i.e. 0.75–1.00)
  2. Citation precision low with few citations emitted (ledger: 3 citations, precision
     **0.33**; qualitative: omission dominates, with several uncited-but-used evidence docs,
     ledger 8/8)
  3. Example-leak: unsupported citations point at `[doc-6]`, the system prompt's format
     example (qualitative, the chapter's hook; flag if absent)
  4. Drift monitor: JS divergence noise floor **~0.0004 bits** (weeks 2–3) stepping to
     **≥ 10x** the floor at week 4 and **~40x+** by weeks 5–6 (ledger 0.0040 / 0.0167–0.0187;
     ratios matter, not decimals)
  5. `gold_in_top3` = **1.000 all six weeks** (exact): retrieval survives the rename while
     attribution fires, the early-warning story
- Known flakiness: Qwen-0.5B + MiniLM downloads; corpus is inline (no dataset dependency).
- PASS: Run-all clean; drift step-change vs noise floor reproduces; retrieval hit-rate stays
  perfect while the attribution monitor fires.
- Measured: wall time ____ · numbers ____ · PASS/FAIL ____

---

## Session summary

| Notebook | Wall time | PASS/FAIL | Flags |
|---|---|---|---|
| ch01 | | | |
| ch02 | | | |
| ch03 | | | |
| ch04 | | | |
| ch05 | | | |
| ch06 | | | |
| ch07 | | | |
| ch08 | | | |
| ch09 | | | |
| ch10 | | | |
| ch11 | | | |
| ch12 | | | |
| ch13 | | | |

- Colab GPU actually assigned: ____
- Date / Colab base image (from `!cat /etc/os-release` or the runtime version note): ____
- Overall verdict: ____
