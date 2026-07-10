"""Batched greedy evaluation for both tasks.

Uses Hugging Face generate. Prompts are sorted by length so batches carry little padding,
and generation stops at the first newline, since both tasks answer on one line. vLLM can
replace `generate` later if it runs on Kaggle's T4s. The scoring functions are shared, so
metrics stay identical.
"""
from __future__ import annotations

import time

import torch

from lorathresh.data import facts as facts_task
from lorathresh.data import sql as sql_task

MAX_NEW_TOKENS = {"facts": 16, "sql": 96}


def length_sorted_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Index batches, longest first, so the first batch surfaces OOM immediately."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int = 64) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = [len(tokenizer(p).input_ids) for p in prompts]
    outputs: list[str] = [""] * len(prompts)
    model.eval()
    for batch in length_sorted_batches(lengths, batch_size):
        enc = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_strings=["\n"],
            tokenizer=tokenizer,
            pad_token_id=tokenizer.pad_token_id,
        )
        texts = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        for i, text in zip(batch, texts):
            outputs[i] = text
    return outputs


def evaluate_facts(model, tokenizer, examples: list[facts_task.Example], batch_size: int = 64) -> dict:
    start = time.time()
    preds = generate(model, tokenizer, [e.prompt for e in examples], MAX_NEW_TOKENS["facts"], batch_size)
    correct = [facts_task.score(p, e.answer) for p, e in zip(preds, examples)]
    return {"acc": sum(correct) / len(examples), "n": len(examples), "eval_seconds": time.time() - start}


def evaluate_sql(model, tokenizer, examples: list[sql_task.SqlExample], batch_size: int = 32) -> dict:
    start = time.time()
    prompts = [sql_task.format_prompt(e.context, e.question) for e in examples]
    preds = generate(model, tokenizer, prompts, MAX_NEW_TOKENS["sql"], batch_size)
    scores = [sql_task.score(p, e) for p, e in zip(preds, examples)]
    return {
        "exec_match": sum(s["exec_match"] for s in scores) / len(examples),
        "exact_match": sum(s["exact_match"] for s in scores) / len(examples),
        "n": len(examples),
        "eval_seconds": time.time() - start,
    }
