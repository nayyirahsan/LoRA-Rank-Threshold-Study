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

## 14. lr_cal completed: results, throughput, and what they don't settle
Kaggle kernel v3, 2×T4, fp16. 18/18 runs finished with no failures (318 min wall clock, including
install and gates). During the run the status API returned `Permission 'kernels.get' was denied`
and connection errors for ~70 min. `watch` treated them as transient, as designed, and still
caught the real `COMPLETE`.

| Setting (1 seed) | Full FT lr 1e-5 / 3e-5 / 1e-4 | LoRA r=4 lr 1e-4 / 3e-4 / 1e-3 | LoRA r=64 lr 1e-4 / 3e-4 / 1e-3 |
|---|---|---|---|
| Facts N=1000, 10 ep (held-out acc, n=500) | 1.000 / 0.998 / 1.000 | 0.892 / **1.000** / 0.998 | 0.902 / 0.996 / **1.000** |
| SQL n=2000, 2 ep (exec match, n=500, SE ≈ 1.6pp) | **0.852** / 0.834 / 0.786 | 0.770 / 0.850 / **0.862** | 0.720 / 0.846 / **0.854** |

**Throughput (tokens/s):** SQL full FT 1264 (**4.1×** the emulated-bf16 run), SQL LoRA ~1175. But
facts full FT was only ~485 and facts LoRA ~370. Every run inherited the gate's fixed
`--micro-batch-size 8`, sized for SQL rows of up to ~190 tokens. It split 22-token facts batches
that fit whole (facts LoRA peaked at 1.6GB) into 4 forward/backward passes per step.
**Fix:** `--max-micro-tokens` caps padded tokens per micro-batch, so facts batches stay whole and long
SQL batches still split. The gradients are identical; the kernel's fallback now uses a 1024-token cap.

**What lr_cal does and doesn't settle:**
- **Facts N=1000 is saturated.** Every full-FT LR and every LoRA LR ≥ 3e-4 reaches ~100%, so this
  setting can't rank LRs for N=4k/16k, where capacity is supposed to bind.
- **Several optima sit on the edge of the grid.** SQL full FT is best at its *lowest* LR (1e-5), and
  SQL LoRA at its *highest* (1e-3), at both ranks. The true optimum may lie outside the tested range,
  and the top-two gaps (0.8–1.2pp) are within one standard error.
- **The best LR did not visibly shift between r=4 and r=64** under α/r, even though α/r differs 16×
  between them. One seed and edge-of-range optima make this weak evidence; it is not a finding yet.
- **Early H1 signal (1 seed):** on SQL, LoRA r=4 (0.862) already matches full FT (0.852), consistent
  with the prediction that r* ≤ 8 for the skill/format task.

**Budget:** GPU time used so far is ≥ 7 session-hours (v2 ~1.6h, v3 5.3h). At the measured facts
full-FT rate, one N=16k run (~13.7M tokens) would take ~7.8h, *before* the micro-batching fix. The
facts grid can't be sized until post-fix facts throughput is measured. Epochs are the obvious lever:
10 epochs gives 100% at N=1000.

## 15. lr_cal_ext settles the calibration, and the grid is scoped to fit
Kaggle kernel, 2×T4, 9/9 runs, 104 min, no failures.

| Question | Result (1 seed) |
|---|---|
| SQL full FT past the low edge | 3e-6 → 0.828 exec match, below 1e-5 (0.852): **1e-5 is an interior optimum** |
| SQL LoRA past the high edge | 3e-3 → 0.798 (r=4), 0.800 (r=64), below 1e-3 (0.862, 0.854): **1e-3 is an interior optimum** |
| Facts N=4000, 10 epochs | full FT 1e-5 → 0.999; LoRA r=4 3e-4 → **1.000**; LoRA r=64 1e-3 → **0.907** |
| Base models | facts 0.0 (N=1000 and 4000); SQL exec 0.318, exact 0.042 |

**Throughput with the token cap (tokens/s):** facts full FT 485 → **1069** (2.2×), facts LoRA 370 →
**~1575** (4.2×), SQL LoRA 1175 → **~1634** (+39%). These LoRA runs were submitted before gradient
checkpointing was turned off for LoRA, so the final grid should run faster still.

**What this means:**
- **N=4000 is still saturated at r=4.** Capacity doesn't bind at rank ≥ 4 for up to 4000 facts at 10
  epochs, so if the facts threshold exists at these sizes, it's below r=4. The final grid adds r=2 and
  r=8 around it.
- **1e-3 is too high for high-rank LoRA on facts at N=4000** (r=64: 0.907), while 3e-4 held at
  0.996–1.000 across r=4/64 and N=1000/4000. Facts LoRA uses 3e-4 at every rank.

**The scoped grid (`configs/grid_final.yaml`, 55 runs, one kernel):**
- **SQL:** n=2000, ranks 1–128 plus full FT and base, 3 seeds.
- **Facts:** N ∈ {250, 1000, 4000}, ranks {1, 2, 4, 8, 16, 64, 256} plus full FT and base, 1 seed.
- **Oracle sweep:** runs in the same kernel for every full-FT run.

**Estimated cost:** ~9.5 GPU-hours, about 5 hours of wall clock. The original grids would cost about
169 GPU-hours at lr_cal speeds, or roughly 60 at the new speeds, still well over the free tier.

**Dropped from this pass:** facts N=16k, facts seeds 2–3, the rsLoRA and QLoRA ablations, and the 1.7B
check. The facts results are 1 seed, so the facts r* has no confidence interval.

**Reuse without selection bias:** the calibration registries hold several LRs for the same (setting,
rank). The analysis picks the best-scoring LR per rank group, so seeding the final registry with
every calibration row would give *only some* ranks a best-of-3 advantage. Only rows whose config
exactly matches a `grid_final` config are copied into its resume registry. That gave 11 matches.
**The 3 full-FT matches were then dropped as well** (facts N=1000, facts N=4000, SQL seed 0). The
in-kernel oracle sweep needs each full-FT checkpoint on the session's local disk, and a reused row has
none, so facts N=1000 and N=4000 (1 seed) would have silently gotten no oracle curve. Retraining those 3
costs ~70 GPU-minutes. The final resume registry has 8 rows (3 base, 5 LoRA), leaving 47 runs.

## 16. A dry run on real data caught a misleading verdict
Before `grid_final` finished, `scripts/write_results.py` was run on the combined `lr_cal` +
`lr_cal_ext` registries, which have base, full-FT, and LoRA rows at ranks 4 and 64. It generated:
**"H1, knowledge task: fails. r\* grows 4 → 4 (1×)"**. That is wrong. Rank 4 was the *smallest*
rank tested, so LoRA meeting the threshold there only shows r\* ≤ 4 at both sizes. The data can't
tell growth from no growth.
**Fix:** `r_star` records `min_rank_tested`, and the writer reports such values as ceilings (`≤ r`):
- saturation at the smallest rank for every N → **inconclusive** ("capacity never binds in the tested range")
- a ceiling only at small N → growth is a lower bound, so it can support the prediction but can't refute it

The same logic drives the generated resume bullet, which states thresholds met at the smallest
tested rank as such ("rank 1 (the smallest tested)"). A rerun on the same data now reads "inconclusive"
for facts and "r\* ≤ 4" for SQL. This matters for the final grid, where N=4000 saturation at r=4
makes a threshold at r=1 plausible.

## 17. grid_final: every training run finished, the oracle sweep didn't
Kaggle kernel, 2×T4, 201 min. The preflight and memory-gate fallback worked, and the resume step
loaded the 8 reused rows. **All 47 new training runs finished** (shards: 160 and 195 min, no failures),
so the registry has all 55. **The in-kernel oracle phase then failed on all 6 full-FT runs** with
`RuntimeError: Expected all tensors to be on the same device ... searchsorted`, so there are no oracle
or spectrum rows.

**Cause:** `energy_rank` ran `torch.searchsorted(cumulative, torch.tensor(fraction))`. With
`--svd-device cuda`, `cumulative` was on the GPU and the threshold on the CPU. Every oracle test and
the laptop smoke run did the SVD on the CPU, so the bug never ran before Kaggle.
**Fix:** energy math moves the singular values to CPU float64 first. That's at most a few thousand
numbers, and it also avoids float64 on MPS. A regression test runs the energy functions and `DeltaSVD`
on CUDA or MPS, whichever is available (MPS locally).

**Recovery:** the checkpoints were deleted with the session, so `configs/grid_final_oracle.yaml`
retrains exactly grid_final's 6 full-FT configs, with the same run ids (a test checks this). The
kernel's oracle phase then sweeps ranks for each. Cost is ~80 min of training plus the oracle evals.
**Merging:** in the final registry, the *retrained* full-FT rows replace grid_final's copies. The
oracle truncates the retrained checkpoints, so its full-rank eval then matches the FT score it's
normalized against. Seeds are fixed, but GPU kernels aren't bit-deterministic, so retrained scores can
differ slightly from the first copies. The analysis of the 55 training runs (H1) doesn't depend on the
oracle and was written to the README before the rerun.

## 18. The headline result: rank-1 LoRA matches full FT everywhere tested, and why that makes sense
`grid_final`, 55 runs:

| Setting | Base | Full FT | LoRA r=1 | LoRA r=2…256 |
|---|---|---|---|---|
| SQL n=2000 (3 seeds, exec match) | 0.318 | 0.859 | 0.846 (G 0.98, CI [0.95, 1.00]) | G 0.98–1.01 |
| Facts N=250 (held-out acc, 250 questions) | 0.000 | 1.000 | 0.964 | 0.980–0.992; **r=256: 0.932** |
| Facts N=1000 (500 questions) | 0.000 | 1.000 | 1.000 | 0.992–1.000 |
| Facts N=4000 (1000 questions) | 0.000 | 0.999 | 0.998 | 0.999–1.000 |

**H1 as stated didn't come out.** SQL's r\* ≤ 1 is consistent with the "≤ 8" prediction. For facts,
the threshold never appeared: rank 1 already reaches ≥ 96% gap closure at every N, so the prediction
that r\* grows with N can't be tested at these sizes. That's a scale problem with the design, not
evidence against the mechanism, and a capacity estimate says why:
- **Adapter size at rank 1:** 22,528 parameters per layer (q 3072, k 2048, v 2048, o 3072, gate/up/down
  4096 each) × 28 layers = **630,784** trainable parameters. That matches the measured 40,370,176 at r=64.
- **Information to store:** each fact's value comes uniformly from its attribute pool (200 cities,
  200 employers, 100 years, 100 universities, 20 majors), a mean of **6.58 bits** per fact. N=4000 needs
  **≈ 26.3 kbits**.
- **Capacity:** *Physics of Language Models 3.3* (Allen-Zhu & Li, 2024) measures ~2 bits per parameter for
  sufficiently trained models, and closer to ~1 bit/param with ~100 exposures. Here each fact gets 40
  exposures (4 templates × 10 epochs). At 1–2 bits/param, rank 1 holds **0.63–1.26 Mbits, 24–48× what
  N=4000 needs**. Capacity at rank 1 would start to bind around **~10⁵ facts**.

Caveats: that scaling law was measured for full models, not low-rank adapters on a frozen base, and
fewer exposures lower it further. So this is an order-of-magnitude argument. It does turn the null
result into a falsifiable claim.

**The cheap way to test it (follow-up):** N ≈ 10⁵ facts would cost ~110M training tokens per run, which
doesn't fit on free T4s. Shrinking the adapter instead makes capacity bind at small N: LoRA r=1 on a
*single* projection in a *single* layer has ~3–4k parameters (≈ 3.5–7 kbits), so it should saturate
around ~500–1000 facts. A sweep over adapter size (layers × modules × rank) at fixed N, looking for the
knee where accuracy falls, tests the same mechanism for a few GPU-hours.

**Smaller observations (1 seed, so hypotheses, not findings):**
- **Facts N=250 at r=256 dips to 0.932** (r=4: 0.992; 250 questions, SE ≈ 1.6pp). Standard scaling
  gives α/r = 16/256 = 0.0625, and N=250 has the fewest optimizer steps (~313) to make up for the small
  updates. This fits the high-rank underfitting that rsLoRA addresses, but the rsLoRA ablation was cut
  from this pass.
- **SQL's r=1 lower CI bound sits right at the threshold** (0.95). "r\* ≤ 1" is supported, but narrowly.

**A silent PyTorch pitfall, found while fixing #17:** on torch 2.14, `tensor_on_mps.to("cpu", torch.float64)`
returns **all zeros** with no error. Moving to the CPU and then casting works. The first version of the
oracle fix made exactly that one-step call; the MPS regression test caught it before it reached Kaggle.

## 19. The oracle sweep: full FT's update rank grows with data, but what LoRA needs doesn't
`grid_final_oracle` retrained grid_final's 6 full-FT runs and swept oracle ranks
{0, 1, 2, 4, …, 256, 1024} on each (2×T4, 92 min, no failures). Final registry: 55 training rows, 66
oracle rows, 6 spectrum rows.

**Sanity checks passed on real checkpoints:**
- Rank 0 reproduces base exactly (facts 0.000, SQL 0.318).
- Rank 1024 reproduces full FT (e.g. SQL seed 2: 0.868 = 0.868).
- The retrained full-FT runs matched grid_final's originals exactly on all three facts sizes, and
  within ±0.004 on SQL (seed 1: 0.864 vs 0.868; seed 2: 0.868 vs 0.864). Seeds are fixed, but GPU
  kernels aren't bit-deterministic.

| Setting | Oracle score at r = 1 / 4 / 16 / 64 / 128 | Oracle r\* | Median 90%-energy rank | LoRA r=1 |
|---|---|---|---|---|
| Facts N=250 | 0.020 / 0.080 / 0.480 / 0.984 / 0.996 | 64 | 127 | 0.964 |
| Facts N=1000 | 0.000 / 0.058 / 0.418 / 0.954 / 0.996 | 64 | 192 | 1.000 |
| Facts N=4000 | 0.003 / 0.021 / 0.174 / 0.851 / 0.994 | 128 | 268 | 0.998 |
| SQL n=2000 (3 seeds) | ~0.45 / 0.58–0.73 / 0.82–0.84 / 0.85–0.87 / 0.85–0.87 | 32 | 181–192 | 0.846 |

**Reading:**
- **H2b: LoRA ≫ oracle in every setting.** Mean G(LoRA) − G(oracle) is +0.57, +0.63, +0.70 (facts)
  and +0.24 (SQL). The gap is largest at low rank: at N=4000, rank-1 LoRA gets 0.998, while the rank-1
  truncation of full FT's update gets 0.003.
- **The growth H1 predicted is real, but it's in the full-FT update.** As N grows 16×, the energy rank
  goes 127 → 192 → 268 and the oracle threshold 64 → 64 → 128. Attention projections carry the
  lowest-rank updates and MLP down_proj the highest at every size (e.g. N=4000: k_proj 152, down_proj 396).
  LoRA's threshold stays at rank 1 throughout.
- **So full FT's update isn't a low-rank solution plus noise.** Its top singular direction alone
  recovers almost nothing. The rank-1 solution LoRA finds is a different point in weight space, not a
  truncation of the full-FT one.
- **H2a as stated fails** (energy rank 127–268 vs LoRA r\* ≤ 1). The energy rank does track the oracle's
  threshold: 2.0×, 3.0×, 2.1× high on facts and 5.7× on SQL. That's a statement about full FT, not LoRA.

**Why this matters:** the "effective rank" of full fine-tuning updates is sometimes used to argue that
LoRA needs higher rank (e.g. that full-FT deltas have far higher rank than typical LoRA configs). These
results show that proxy can be off by two orders of magnitude, at least where capacity doesn't bind.
Whether it stays wrong once capacity *does* bind is exactly what the adapter-shrinking follow-up in #18 tests.

`notebooks/kaggle_runner.ipynb` puts this together: setup, a throughput and peak-memory gate on the
two most memory-hungry configs (full FT and LoRA r=64 on SQL), two-shard launch, a progress check,
an oracle sweep over every full-FT run, and aggregation.
