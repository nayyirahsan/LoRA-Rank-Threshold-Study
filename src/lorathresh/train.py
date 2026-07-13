"""Train and evaluate one configuration: base (eval only), full FT, LoRA, or QLoRA.

    python -m lorathresh.train --task facts --n 1000 --method lora --rank 16 --lr 3e-4 --epochs 10
    python -m lorathresh.train --task sql --n 10000 --method full --lr 3e-5 --epochs 2 --sql-path sql.json

Results go to the run registry, keyed by the full config. Re-running a finished config is a
no-op unless you pass --force. Design choices are explained in docs/engineering_log.md #7.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from lorathresh.data import facts as facts_task
from lorathresh.data import sql as sql_task
from lorathresh.eval import evaluate_facts, evaluate_sql
from lorathresh.oracle import TARGET_MODULES, target_linears
from lorathresh.registry import Registry, run_id

METHODS = ("base", "full", "lora", "qlora")


@dataclass
class RunConfig:
    task: str                   # facts | sql
    n: int                      # facts: number of facts; sql: number of training examples
    method: str                 # base | full | lora | qlora
    rank: int = 0
    scaling: str = "standard"   # standard: alpha/r | rslora: alpha/sqrt(r)
    alpha: float = 16.0
    lr: float = 2e-4
    epochs: float = 1.0
    seed: int = 0
    model: str = "Qwen/Qwen3-0.6B-Base"
    batch_size: int = 32
    max_len: int = 256
    n_eval: int = 1000
    world_seed: int = 0         # facts world / SQL split, held fixed across every run
    max_steps: int | None = None  # smoke tests only; part of the run id so they never mix with real runs

    def __post_init__(self):
        if self.task not in ("facts", "sql"):
            raise ValueError(f"unknown task {self.task}")
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method}")
        if self.method in ("lora", "qlora") and self.rank < 1:
            raise ValueError("LoRA methods need rank >= 1")
        if self.method in ("base", "full"):
            self.rank, self.scaling, self.alpha = 0, "standard", 0.0  # irrelevant; keep the run id canonical
        if self.method == "base":
            # Greedy eval of an untrained model: seed and optimizer settings can't change the result.
            self.lr, self.epochs, self.seed, self.max_steps = 0.0, 0.0, 0, None


# ---------------------------------------------------------------- data

def load_task(cfg: RunConfig, sql_path: str | None = None):
    """(train pairs of (prompt, answer), eval examples) for the configured task."""
    if cfg.task == "facts":
        world = facts_task.build_facts(cfg.n, seed=cfg.world_seed)
        train = [(e.prompt, e.answer) for e in facts_task.train_examples(world)]
        return train, facts_task.eval_examples(world, cfg.n_eval, seed=cfg.world_seed)
    raw = sql_task.load_raw(sql_path)
    train, ev, _ = sql_task.make_splits(raw, n_train=cfg.n, n_eval=cfg.n_eval, seed=cfg.world_seed)
    return [(sql_task.format_prompt(e.context, e.question), e.answer) for e in train], ev


def encode(tokenizer, prompt: str, answer: str, max_len: int) -> tuple[list[int], list[int]]:
    """Token ids and labels, with the loss only on the answer, its newline, and EOS.

    The answer begins with a space, and Qwen's pre-tokenizer splits at spaces, so encoding the
    prompt and answer separately gives the same ids as encoding them together. That keeps the
    training ids consistent with what the model sees at eval time.
    """
    p = tokenizer(prompt, add_special_tokens=False).input_ids
    a = tokenizer(" " + answer + "\n", add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return (p + a)[:max_len], ([-100] * len(p) + a)[:max_len]


def length_grouped_batches(lengths: list[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    order = list(range(len(lengths)))
    rng.shuffle(order)
    mega = batch_size * 50
    batches = []
    for i in range(0, len(order), mega):
        chunk = sorted(order[i : i + mega], key=lambda j: lengths[j])
        batches += [chunk[k : k + batch_size] for k in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(encoded, batch: list[int], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(encoded[i][0]) for i in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, i in enumerate(batch):
        x, y = encoded[i]
        ids[row, : len(x)] = torch.tensor(x)
        labels[row, : len(y)] = torch.tensor(y)
        mask[row, : len(x)] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# ---------------------------------------------------------------- model

def device_and_amp() -> tuple[str, torch.dtype | None]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4: float16
    if torch.backends.mps.is_available():
        return "mps", None
    return "cpu", None


def expected_lora_params(model: nn.Module, rank: int) -> int:
    return sum(rank * (lin.in_features + lin.out_features) for lin in target_linears(model).values())


def wrap_lora(model: nn.Module, cfg: RunConfig) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    expected = expected_lora_params(model, cfg.rank)
    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        target_modules=list(TARGET_MODULES),
        lora_dropout=0.0,
        bias="none",
        use_rslora=cfg.scaling == "rslora",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != expected:
        raise RuntimeError(f"trainable params {trainable} != r*(d_in+d_out) sum {expected}")
    return model


def build_model(cfg: RunConfig, device: str, amp_dtype: torch.dtype | None):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    half = amp_dtype if amp_dtype is not None else torch.float32
    if cfg.method == "qlora":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=half
        )
        model = AutoModelForCausalLM.from_pretrained(cfg.model, quantization_config=quant, device_map={"": device})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return wrap_lora(model, cfg), tokenizer
    # Full FT keeps float32 master weights. A frozen base (LoRA, eval-only) can sit in half precision.
    dtype = torch.float32 if cfg.method == "full" else half
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(device)
    if cfg.method == "base":
        return model, tokenizer
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if cfg.method == "lora":
        model = wrap_lora(model, cfg)
    return model, tokenizer


# ---------------------------------------------------------------- loop

def accumulate_backward(model, inputs: dict[str, torch.Tensor], micro_batch_size: int, device: str, amp_dtype, scaler) -> float:
    """Backward over one optimizer batch in micro-batches. Returns the batch's mean token loss.

    Exactly matches a single full-batch backward. HF returns each micro-batch's mean over its own
    answer tokens, so each micro loss is weighted by its share of the batch's answer tokens. A plain
    average of micro means would over-weight micro-batches with short answers.

    Micro-batching exists for memory. Qwen3's 152k vocabulary makes the logits huge: 32 x 189 tokens
    is ~3.7GB in fp32, with several copies held for the loss and backward. On top of full FT's ~9.6GB
    of weights, grads and Adam state, that can overflow a 15GB T4.
    """
    n_total = int((inputs["labels"][:, 1:] != -100).sum())
    batch_loss = 0.0
    for start in range(0, inputs["input_ids"].shape[0], micro_batch_size):
        micro = {k: v[start : start + micro_batch_size].to(device) for k, v in inputs.items()}
        n = int((micro["labels"][:, 1:] != -100).sum())
        if n == 0:
            continue
        with torch.autocast(device_type=device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss = model(**micro).loss
        scaler.scale(loss * (n / n_total)).backward()
        batch_loss += loss.item() * n / n_total
    return batch_loss


def train_loop(
    model, tokenizer, pairs, cfg: RunConfig, device: str, amp_dtype, log_every: int = 50,
    micro_batch_size: int | None = None,
) -> dict:
    encoded = [encode(tokenizer, p, a, cfg.max_len) for p, a in pairs]
    lengths = [len(x) for x, _ in encoded]
    rng = random.Random(cfg.seed)
    steps_per_epoch = math.ceil(len(encoded) / cfg.batch_size)
    total_steps = cfg.max_steps or math.ceil(steps_per_epoch * cfg.epochs)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.03 * total_steps)), total_steps)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    model.train()
    step, tokens, recent, start = 0, 0, [], time.time()
    while step < total_steps:
        for batch in length_grouped_batches(lengths, cfg.batch_size, rng):
            if step >= total_steps:
                break
            inputs = collate(encoded, batch, tokenizer.pad_token_id)
            loss = accumulate_backward(model, inputs, micro_batch_size or cfg.batch_size, device, amp_dtype, scaler)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            tokens += int(inputs["attention_mask"].sum())
            recent = (recent + [loss])[-100:]
            if step % log_every == 0 or step == total_steps:
                # flush: under nohup/redirect stdout is block-buffered, so progress checks would see nothing for many minutes
                print(f"step {step}/{total_steps}  loss {sum(recent) / len(recent):.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
    elapsed = time.time() - start
    return {
        "train_steps": step,
        "final_train_loss": sum(recent) / max(1, len(recent)),
        "train_seconds": elapsed,
        "train_tokens_per_second": tokens / max(elapsed, 1e-9),
        "trainable_params": sum(p.numel() for p in params),
    }


def evaluate(model, tokenizer, eval_examples, cfg: RunConfig, device: str, amp_dtype) -> dict:
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    with torch.autocast(device_type=device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        if cfg.task == "facts":
            return evaluate_facts(model, tokenizer, eval_examples)
        return evaluate_sql(model, tokenizer, eval_examples)


def run(
    cfg: RunConfig,
    registry_path: str = "results/runs.jsonl",
    output_dir: str = "checkpoints",
    sql_path: str | None = None,
    push_repo: str | None = None,
    force: bool = False,
    micro_batch_size: int | None = None,  # memory only; math is identical, so it isn't part of the run id
) -> dict | None:
    registry = Registry(registry_path)
    key = asdict(cfg)
    if registry.is_done(key) and not force:
        print(f"skip {run_id(key)} (already in {registry_path})")
        return None
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device, amp_dtype = device_and_amp()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    pairs, eval_examples = load_task(cfg, sql_path)
    model, tokenizer = build_model(cfg, device, amp_dtype)

    metrics: dict = {"n_train_examples": len(pairs), "device": device, "amp": str(amp_dtype)}
    if cfg.method != "base":
        metrics |= train_loop(model, tokenizer, pairs, cfg, device, amp_dtype, micro_batch_size=micro_batch_size)
        metrics["micro_batch_size"] = micro_batch_size or cfg.batch_size
        rid = run_id(key)
        # Full-FT checkpoints feed the oracle. For LoRA, peft saves only the adapter.
        out = Path(output_dir) / rid
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        metrics["checkpoint"] = str(out)
        if push_repo:
            from huggingface_hub import HfApi

            HfApi().upload_folder(folder_path=str(out), repo_id=push_repo, path_in_repo=rid, repo_type="model")
    metrics |= evaluate(model, tokenizer, eval_examples, cfg, device, amp_dtype)
    if device == "cuda":
        metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    registry.record(key, metrics)
    if push_repo:
        registry.push_to_hub(push_repo)
    print({k: v for k, v in metrics.items() if k != "checkpoint"})
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in fields(RunConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name in ("task", "n", "method"):
            ap.add_argument(flag, required=True, type=int if f.name == "n" else str)
        elif f.name == "max_steps":
            ap.add_argument(flag, type=int, default=None)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    ap.add_argument("--registry", default="results/runs.jsonl")
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--sql-path", default=None)
    ap.add_argument("--push-repo", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--micro-batch-size", type=int, default=None, help="split each batch to save memory; same math")
    args = vars(ap.parse_args())
    extra = {k: args.pop(k) for k in ("registry", "output_dir", "sql_path", "push_repo", "force", "micro_batch_size")}
    run(
        RunConfig(**args),
        registry_path=extra["registry"],
        output_dir=extra["output_dir"],
        sql_path=extra["sql_path"],
        push_repo=extra["push_repo"],
        force=extra["force"],
        micro_batch_size=extra["micro_batch_size"],
    )


if __name__ == "__main__":
    main()
