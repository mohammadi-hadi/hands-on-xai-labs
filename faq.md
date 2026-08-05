# Frequently Asked Questions

Real questions from running the book's labs, with answers grounded in Appendix A, the notebooks, and the `experiments/` results ledgers.

---

**1. Should I run the labs on my Mac (MPS), on plain CPU, or on Colab?**

All three work; they differ in speed and reproducibility, not capability. Colab (free T4) is the zero-setup reference path: every notebook finishes in 20 minutes or less, most well under 10. Locally, notebooks auto-detect MPS on Apple Silicon and CUDA on NVIDIA. But note that several chapters' *committed* numbers were deliberately produced with `BOOK_DEVICE=cpu`, because CPU inference is deterministic and MPS/GPU is not bit-reproducible for training. If you want to match the book digit for digit, check the device recorded in that chapter's ledger and use the same one.

**2. What do `BOOK_DEVICE` and `BOOK_SMOKE` actually do?**

`BOOK_DEVICE` forces the compute device (`cpu`, `mps`, `cuda`); unset, notebooks auto-detect. `BOOK_SMOKE=1` switches to smoke mode: tiny models (e.g. `prajjwal1/bert-tiny` instead of DistilBERT), small data slices, and reduced explainer budgets, so a notebook runs in seconds as a correctness check. Two more switches matter: `BOOK_NB_SAVE=1` writes executed outputs back into the `.ipynb` (used before rendering the book), and `BOOK_NB_TIMEOUT` sets the per-notebook timeout in seconds (default 1200). See Appendix A's environment-switches table.

**3. Are smoke-mode numbers supposed to match the book?**

No, and that's by design. Smoke mode changes models, data sizes, and sampling budgets, so its numbers legitimately differ. It answers "does the code run?", not "do I reproduce the book?". Use full mode (no `BOOK_SMOKE`) to compare against the ledgers.

**4. My numbers differ slightly from the book's. Is something broken?**

Usually not. Three known causes, in order of likelihood: (1) device: MPS/GPU training is not bit-reproducible (the Chapter 12 ledger records the same pipeline, same seed producing 0/20 vs 5/20 acknowledged flips on MPS vs CPU); (2) library versions: the exact resolved versions live in `uv.lock`, and each Colab notebook's pinned `%pip` cell installs the versions the results were produced with; (3) borderline generations flipping on different float kernels (Chapter 13's ledger expects ±1 question). Chapters that are pure inference on CPU with fixed seeds (3, 5, 6, 10, 11) should reproduce exactly.

**5. Why does the book always evaluate on the tweet_eval/hate VALIDATION split and never TEST?**

Because the HatEval (SemEval-2019 Task 5) test split has a known topic shift: it is dominated by immigration-hashtag traps, and every published system degrades on it. The Chapter 4 ledger measured it on the book's specimen: macro-F1 0.676 on validation vs 0.357 on test, with the hate-prediction rate jumping from 61% to 96%. A yardstick that warped would contaminate every downstream experiment, so Chapters 5, 6, and 8 sample from validation only. Chapter 4 shows both splits side by side precisely to teach this failure.

**6. Where does Hugging Face put its downloads, and do I need a token?**

Models and datasets cache under `~/.cache/huggingface/` by default (set `HF_HOME` to move it). Everything required downloads anonymously; a token is never mandatory. The single exception is optional: `google/gemma-2-2b`, the base model for the Gemma Scope SAE cell in Chapter 10, is gated. To run it, accept the license on the model page, authenticate (`huggingface-cli login` locally or `HF_TOKEN` in Colab), set `RUN_GEMMA = True`, and budget a ~5 GB download.

**7. How much RAM/VRAM do I need for the 0.5B-model chapters (9, 11, 13)?**

Modest. The book runs `Qwen/Qwen2.5-0.5B-Instruct` in float32 on CPU for Chapters 11 and 13 (deterministic, ~1 GB of weights, roughly 2–4 GB of process RAM) and fp16 on MPS for Chapter 9. A free T4 (16 GB) handles everything with room to spare; the fp32 0.5B judge in Chapter 8 fits a T4 easily. Nothing in the required labs needs more than a free-tier T4. That is a design constraint of the book.

**8. Can I use a different model than the one a chapter ships with?**

Yes, with eyes open. Several chapters expose a switch: `BOOK_CH09_MODEL` swaps Chapter 9's model for a larger Qwen sibling (budget ~3x wall time for 1.5B), Chapter 8's exercise cell takes any judge id, Chapter 11's exercise adds a fourth model, and Chapters 2/4/5 exercise cells accept any sequence-classification checkpoint. Expect all quoted numbers to change (the ledgers describe specific models) and beware architecture assumptions: the bert-mini fallback has 4 layers × 4 heads, so "layer 6" analyses become "layer 4" (the code adapts via runtime-discovered constants, your plots just won't match the book's).

**9. Chapters 5/6/8 warn that `models/ch04-specimen` is missing. What happened?**

The fine-tuned DistilBERT specimen is 268 MB, too big for the GitHub repo, so it only exists locally after you run the Chapter 4 notebook in full mode. Without it, the `get_specimen()` contract quick-trains a `prajjwal1/bert-mini` fallback (1,500 tweets, 2 epochs, ~1–2 min on a T4). Everything executes, but metrics and plots describe the stand-in, not the book's specimen. To match the book: run Chapter 4 in full mode first.

**10. Why is the first run of a notebook so slow, and later runs fast?**

Downloads. First runs pull model weights and datasets (GPT-2 ~550 MB, DistilBERT ~268 MB, Qwen-0.5B ~950 MB fp32, the Chapter 10 SAE ~330 MB, tweet_eval ~1 MB), all cached afterward. On Colab, the pinned `%pip` install cell also dominates (2–5 minutes for the heavier stacks) while the actual computation is usually the cheaper half.

**11. Colab asks me to restart the runtime after the install cell. Did something fail?**

No. This is expected. The install cell pins `numpy==1.26.4`, older than the NumPy 2.x Colab preinstalls, so pip downgrades it and Colab prompts for a restart. Restart, re-run from the top, and everything works. The pin exists so your numbers match the book's; the notebooks also run on Colab's stock versions.

**12. Why does Chapter 10 need its own `make setup-mechinterp` environment?**

`transformer-lens` and `sae-lens` pin their own `transformers` version ranges and need to resolve independently of the main `llm` group. Note that `uv sync` is exact: `make setup-mechinterp` deliberately installs *both* groups, because syncing `mechinterp` alone would uninstall the `llm` stack. When in doubt, `make setup-mechinterp` gives you everything.

**13. Where do the book's numbers actually live, and how do I check mine against them?**

In `experiments/chNN-results.md`: one ledger per chapter recording the executed setup, every committed number, honest surprises, and per-chapter Colab notes. The rule of the ledgers: every number comes from an executed run of the chapter's notebook with fixed seeds, and negative or messy results are recorded, not tuned away. If your run disagrees, compare your device, mode (smoke vs full), and library versions against the ledger's Setup section first.

**14. `make nb-run` skips most notebooks on my machine. Why?**

Notebooks declare required dependency groups in their metadata; `scripts/run_notebooks.py` SKIPs (rather than fails) any notebook whose group isn't installed. A core-only environment (`make setup`) runs only Chapter 1. That's intentional; it doubles as the CI canary. Run `make setup-ml` for Chapters 2–9 and 11–13, `make setup-mechinterp` for Chapter 10.

**15. Do the labs work fully offline?**

Mostly, once caches are warm. Notebooks that need remote data ship inline fallbacks (Chapter 4/5/6/8: a 12-tweet fallback; Chapter 9: 10 inline GSM8K items; Chapter 10: a 20-sentence corpus fallback; Chapter 12 is fully synthetic and needs no datasets at all). Sections degrade gracefully: the fallback numbers differ from the ledgers, and the notebooks say so where it matters.

**16. Attention analysis returns nothing useful / `output_attentions` is empty. Bug?**

Load the model with `attn_implementation="eager"`. The default SDPA kernels do not return per-head attention weights; Chapters 3 and 5 set this explicitly, and any attention analysis you write on your own models needs it too.
