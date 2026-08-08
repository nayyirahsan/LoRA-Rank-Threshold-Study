# Engineering log

Decisions and bugs found during the build. Each entry records what happened, how it was
found, and the measured effect. These are the details that don't make it into the final plot.

## 1. Facts: a shared random stream would have confounded the data-size axis
**Found by:** unit test `test_smaller_world_is_prefix_of_larger`.
Names and attribute values came from one random stream. Generating 4k facts consumed more
names first, which shifted every value, so the N=1k world was not a subset of the N=4k world.
Comparisons across N would have changed *which* facts were learned, not just *how many*.
**Fix:** separate seeded streams for value pools, names, and each person's values. The world
for a smaller N is now an exact prefix of the world for a larger N.

## 2. SQL: duplicate CREATE TABLE statements crashed the execution matcher
**Found by:** running the matcher on 2,000 real gold queries (`scripts/audit_sqlexec.py`).
sql-create-context sometimes declares the same table twice, with different columns and
different capitalization (`Faculty` / `faculty`).
**Fix:** merge declarations case-insensitively, as SQLite resolves names.

## 3. SQL: execution match accepted ~10% of deliberately wrong queries
**Found by:** the same audit. Each gold query got one realistic error (a swapped literal, a
flipped comparison, a swapped aggregate) and the audit counted how many the matcher accepted.

| Matcher version | False-positive rate (all) | False-positive rate (scorable) |
|---|---|---|
| Random rows seeded with gold literals | 9.9% (181/1820) | — |
| + witness and near-miss rows | 2.4% (43/1820) | **0.1% (2/1773)** |

**Cause:** with several AND'ed conditions (`tyre = "g" AND chassis = ... AND driver = ...`) no
random row satisfied them all. Gold returned empty, the wrong query returned empty, and they
"matched."
**Fix:** plant *witness rows* that satisfy every gold `column op literal` condition, and
*near-miss rows* that break exactly one. A subtlety: VARCHAR columns compare as text in SQLite
(`"10" > "9"` is false), so satisfying values for `>`/`<` are built as strings.
**Remaining false positives** are degenerate gold queries: numeric column names parsed as
constants (`WHERE 2008 = "153"`) and malformed gold (`score = 71 - 68 - 76 = 215`). Their gold
result is empty on every DB, so `data/sql.py` drops them as unscorable (4.1% of examples)
instead of scoring label noise.

## 4. Oracle stores factors in float32, not float16
The singular directions at the tail of dW are small. In float16 the smallest ones lose
precision, which would bias oracle(r) at high rank, exactly where the capacity-vs-optimization
question gets decided. Cost: ~2.5GB of host RAM for Qwen3-0.6B.

## 5. Oracle truncation is optimal in weight space, not function space
**Found by:** a unit test that wrongly assumed logit error falls steadily as oracle rank grows.
On a tiny Qwen3 with a planted rank-16 update, mean logit error was 0.132 at r=0, 0.120 at r=1,
**0.129 at r=4**, and ~1e-7 at r=16.
**Why:** Eckart-Young makes SVD_r(dW) the best rank-r approximation *of each matrix*. The
network, though, composes many truncated matrices through nonlinearities, so partial truncations
can interact badly. Weight-space error does fall steadily, and the test now checks that.
**Implication for the analysis:** oracle(r) accuracy curves may be non-monotone. r* is read from
the full measured curve (first rank whose confidence interval clears 95% gap closure). Curves are
never interpolated between ranks, and the whole curve gets plotted, not just r*.

## 6. Baseline smoke test (Qwen3-0.6B-Base, CPU, before any training)
- Facts (50 held-out-template questions): **0.0%**. The model invents answers ("a renowned
  Iranian physicist"), confirming the fictional world is unknown, so gap closure has a clean
  denominator.
- SQL (20 eval examples): **15% execution match, 0% exact match**. It gets the idea but not the
  form (`SELECT * ... ORDER BY laps LIMIT 1` vs gold `SELECT MIN(laps) ...`).
- transformers 5.17 accepts `stop_strings` + `dtype`, and generation stops at the first newline.

## 7. Training choices in `train.py`
- **Loss on answer tokens only.** Otherwise the facts loss would mostly measure how well
  question templates are memorized, and that would change with the template count, not the fact count.
- **Length-grouped batching, not packing.** Packing plus answer-only masking needs custom attention
  masks to keep examples from attending to each other. Grouping by length gets most of the
  padding savings with none of that risk.
- **Precision matches across methods.** Full FT keeps float32 master weights under fp16/bf16
  autocast. LoRA keeps adapters in float32 over a frozen half-precision base. Both run the
  forward pass in the same autocast dtype, so precision doesn't differ between arms of the comparison.
- **Fixed α=16 by default,** so standard scaling α/r really does shrink per-rank update size as r
  grows. LR is calibrated per rank bucket, and rsLoRA (α/√r) is the ablation, rather than hiding
  the effect by tying α to r.

## 8. Pipeline checks on the real model (before any GPU time)
- **Tokenization boundary:** `encode()` tokenizes prompt and answer separately to build the loss
  mask. Checked against joint tokenization on 10,000 real prompts (8k facts, 2k SQL): **0
  mismatches**. Sequence lengths: facts mean 21 tokens (max 27); SQL mean 67, p99 124, max 189,
  so `max_len=256` never truncates.
- **LoRA smoke run** (r=16, 40 steps, 50 facts, Apple MPS): 18s. 10.09M trainable parameters,
  which the built-in r·(d_in+d_out) check confirmed. Held-out-template accuracy **0% → 64%**:
  facts learned under training templates do transfer to unseen phrasings, at least at tiny N.
  This is the first evidence against risk #1 (knowledge that can't be retrieved).
- **Registry:** rerunning an identical config is skipped. Base-only runs are canonicalized (no
  seed or LR), so the grid runs one base eval per (task, n) instead of one per seed.

## 9. Oracle smoke test: sanity checks pass, and the oracle is not an upper bound on LoRA
Smoke grid (`configs/smoke.yaml`, 50 facts, 40 steps, one seed, Apple MPS), then
`run_oracle.py` on its full-FT checkpoint:

| Rank | Oracle W_base + SVD_r(ΔW) | Trained LoRA |
|---|---|---|
| 0 | 0% (base: 0%) | — |
| 1 | 0% | — |
| 4 | 4% | **60%** |
| 16 | 56% | **64%** |
| 64 | 86% | — |
| 1024 | 90% (full FT: 90%) | — |

- **Sanity checks pass on the real model.** r=0 reproduces base and full rank reproduces FT
  exactly, so leaving embeddings and norms at base costs nothing on this run.
- **H2(a) early sign:** the median 90%-energy rank of ΔW is 67.5 (gate/up/down 80–100,
  attention 48–55). The oracle crosses ~96% gap closure at r=64, in the same place.
- **Unexpected:** at low rank, trained LoRA beats the truncated FT update by a wide margin
  (r=4: 60% vs 4%). My framing of H2(b) assumed the oracle was an upper bound on LoRA, and it isn't.
  Full FT is unconstrained, so it spreads its solution across many directions, and truncating it
  discards pieces that matter. LoRA optimizes *inside* the rank-r space and can find a different,
  more compact solution.
- **Revised reading of H2(b), three outcomes:**
  - oracle ≈ LoRA: capacity limit
  - oracle ≫ LoRA: optimization gap (test rsLoRA)
  - **LoRA ≫ oracle:** a good rank-r solution exists that full FT didn't pick, so capacity isn't
    binding at that rank. The informative comparison is then where each curve reaches G ≥ 0.95,
    not the gap between them at a given rank.
- **Caveat:** 50 eval examples (±7pp), 40 steps, one seed. This checks the pipeline and suggests a
  hypothesis; it isn't evidence yet. Seeds and full-length training are needed before believing it.
- **Local throughput isn't a proxy for T4:** ~190 tokens/s on MPS for both full FT and LoRA. The
  Kaggle gate in the README measures the number the grid budget depends on.

## 10. Figure design: chart form and palette are checked, not eyeballed
`analysis.py` makes three figures plus one table. Each form was chosen for what the reader has to do:
- **Gap closure vs rank** (H1, H2b): one panel per setting, LoRA and oracle as two categorical
  series, bootstrap CIs as 10%-opacity bands, one legend. The 95% threshold is a *solid* muted
  hairline with a text label, since dashed lines read as projections.
- **Ablations:** emphasis form. The ablated arm (rsLoRA, QLoRA) gets color and standard LoRA is gray
  context, so every color keeps meaning one thing across figures.
- **Update spectrum** (H2a): data sizes are ordered, so they get a one-hue ordinal ramp rather than
  categorical colors.
- **Energy prediction vs measured r\*:** only a few numbers, so it's a table (`r_star.md`), not a scatter plot.

Palette validation (`validate_palette.js`, light surface `#fcfcfb`):

| Color set | Check mode | Result |
|---|---|---|
| LoRA `#2a78d6` / oracle `#eb6834` | all pairs (panels side by side) | PASS: colorblind ΔE 24.7, normal-vision ΔE 33.6, both ≥ 3:1 |
| rsLoRA `#1baf7a` / QLoRA `#eda100` | all pairs | PASS: colorblind ΔE 9.1, normal-vision ΔE 22.9. **Contrast WARN** (2.74:1, 2.11:1) |
| Ordinal ramp `#6da7ec` / `#2a78d6` / `#184f95` | ordinal | PASS: lightness monotone, one hue, light end 2.44:1 |

The contrast WARN is not ignorable. It's covered by the table view (`curves.csv`, `r_star.md` hold
every plotted value) plus the figure legend.

**Looking at the rendered figures found four layout bugs the validator can't catch.** They were
checked on the real smoke registry and on a *synthetic* full-scale registry: made-up scores, 4
settings, 2–3 seeds, ablations. The synthetic numbers only exercise layout and are not results.
1. **Dashed-looking legend keys.** The marker's 1.5pt surface ring cut the short key line into
   dashes. Fixed with longer keys.
2. **Threshold label squeezed** against the 1.0 gridline. First moved below the line, then
   (bug 3) out of the plot entirely.
3. **Threshold label on top of data** in the SQL panel, where LoRA is already near G=0.9 at r=1.
   A gap-closure curve can sit near the threshold at any rank, so no fixed in-plot position is
   safe. The threshold is now a legend entry in the gap-closure and ablation figures. The spectrum
   figure keeps its in-plot label because cumulative energy is monotone, so the space below the
   90% line at low rank is always empty.
4. **Tick labels merged** ("128256512") with 11 ranks in a 3.3-inch panel, and **the QLoRA legend
   covered its own line.** Past 8 ticks, every tick stays but only every other one is labeled. The
   ablation figure now has one legend above the panels.

## 11. Making free-tier sessions survivable
Two problems were found by walking through what actually happens on a Kaggle T4. Neither could be
observed locally.

**Memory: vocabulary-sized logits, not weights, are the risk for full FT.** Full FT's fixed cost is
~9.6GB (float32 weights, grads, Adam states for 0.6B). But Qwen3's vocabulary is 152k tokens, so the
logits for one SQL batch of 32 × 189 tokens are ~3.7GB in float32, and the loss and backward hold
several copies. That can overflow the 15GB T4 on the longest batches.
**Fix:** `--micro-batch-size` does gradient accumulation within an optimizer batch. Each micro-batch
loss is weighted by its share of the batch's *answer tokens*. HF returns a per-micro-batch mean, so
a plain average would over-weight micro-batches with short answers. A test checks that loss and every
gradient match a single full-batch backward, using deliberately uneven answer lengths. Micro-batch
size is excluded from the run id because the math is identical; it's recorded in the metrics instead.

**Persistence: sessions lose local files.** The registry now mirrors to the private Hub repo that
holds checkpoints: pull-and-merge at startup, push after each recorded run. `run_oracle.py` downloads
a full-FT checkpoint from the Hub when it isn't on local disk. Two races came up while designing this:
- **Pull race:** both shards pull at the same moment and append the same remote rows twice.
  Duplicates would double-count seeds and shrink confidence intervals. `Registry.rows()` now
  deduplicates by run id.
- **Push race:** two shards committing at once, or a network error, used to raise *after* an hour of
  training. Pushes are now best-effort. The row is already safe locally, and the next push uploads
  the whole file.

## 12. Local pilot killed by macOS memory pressure
Stage 1 of the local pilot (`configs/pilot_lr.yaml`, 250 facts) was killed by the OS partway
through run 2 of 8, **full FT at batch 32**. The base eval had finished (0% on all 250 facts). No FT
checkpoint was written. At the kill, swap stood at 12.5 of 13.3GB, while an unrelated sweep
(`dpt.bench.sweep`, 4 CPU-bound workers) was also running.
**Why:** full FT holds ~9.6GB of float32 weights, grads and Adam state. On Apple silicon that comes
out of the same 26GB unified memory as everything else, and MPS allocations don't show up in
process RSS, so `ps` made the job look small. The earlier smoke full-FT run (batch 16, 40 steps)
survived only because it was short.
**Takeaway:** a laptop with other workloads isn't a reliable place for the full-FT arm. The pilot's
questions (are 10 epochs enough for FT? does the best LoRA LR shift with rank?) are exactly what
`configs/lr_cal.yaml` answers on the T4, where the micro-batch flag and peak-memory gate apply.
**Decision:** the local pilot is dropped and its config and partial output deleted. Those two
questions move to the Kaggle `lr_cal` run. When reading its results, check first that full-FT
held-out accuracy at facts N=1000 clearly beats base (G is undefined below a 0.02 gain). If it
doesn't, raise `epochs` in `grid_facts.yaml` before launching the main grid.

## 13. First real GPU run (Kaggle lr_cal): three failures, one useful result
The account had to be phone-verified first. Before that, Kaggle recorded `enable_internet: true`
and still ran the kernel offline (`Could not resolve host: github.com`), which led to a preflight
check. The next submission passed preflight (2× T4, 15GB each). Then:

**a. The full-FT memory gate worked.** Full FT at batch 32 ran out of memory on SQL. The fallback
to `--micro-batch-size 8` passed at **12.28GB peak**.

**b. Every LoRA config crashed.** `ImportError: Found an incompatible version of torchao. Found
version 0.10.0, but only versions above 0.16.0 are supported`. Kaggle's image ships an old
torchao, and peft 0.20 raises on *any* adapter injection when it finds one. Shard 1 died on its
first config; shard 0 finished three full-FT runs, then died on its first LoRA config after
**94 min**.
**Why it wasn't caught earlier:** the gate only exercised full FT, and `run_grid.py` let one
exception kill a whole shard.
**Fixes:**
- uninstall torchao in the kernel (nothing here uses it)
- gate the LoRA path (r=64) too
- `run_grid.py` logs a failed config and continues, aborting only after 3 failures in a row

**c. Emulated bf16 made training ~3× slower than it should be.** All three full-FT runs recorded
`amp: torch.bfloat16` at **~310 tokens/s** (46 min each for 856k tokens).
`torch.cuda.is_bf16_supported()` returns True on a T4 because bf16 can be *emulated*, but Turing
(compute capability 7.5) has no native bf16 kernels. The fp16 path with GradScaler was never used.
**Fix:** pick bf16 only when compute capability ≥ 8 (Ampere and newer).
**Consequence:** the three finished runs are **not reused**. The design keeps autocast precision
the same across full FT and LoRA, and precision isn't part of the run id, so resuming them next
to fp16 LoRA runs would add a hidden confound. They're archived under
`results/kaggle/archive/lr_cal_v2_emulated_bf16/` (outside the path the kernel resumes from) and
rerun in fp16.

**d. The one scientific result so far: 10 epochs saturates N=1000 facts.** Full FT reached
**100% held-out-template accuracy** at lr 1e-5 and 3e-5 (72.6% at 1e-4), with final training loss
0.0. This clears the go/no-go check from entry #12: 10 epochs is enough, and very likely more than
enough. **Budget implication:** once fp16 throughput is measured, fewer epochs is the first place
to cut, since the facts grid at N=16k is ~13.7M tokens per full-FT run. Changing epochs changes the
experiment, so it gets decided explicitly with the lr_cal results, not tuned quietly.

`notebooks/kaggle_runner.ipynb` puts this together: setup, a throughput and peak-memory gate on the
two most memory-hungry configs (full FT and LoRA r=64 on SQL), two-shard launch, a progress check,
an oracle sweep over every full-FT run, and aggregation.
