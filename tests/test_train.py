import copy
import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from lorathresh.train import (  # noqa: E402
    RunConfig,
    accumulate_backward,
    collate,
    expected_lora_params,
    length_grouped_batches,
    wrap_lora,
)


def _tiny_qwen3():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    return transformers.Qwen3ForCausalLM(cfg).eval()


def _logits(model):
    with torch.no_grad():
        return model(torch.arange(12).unsqueeze(0)).logits


@pytest.mark.parametrize("scaling", ["standard", "rslora"])
def test_lora_at_init_reproduces_base_exactly(scaling):
    base = _tiny_qwen3()
    wrapped = wrap_lora(copy.deepcopy(base), RunConfig(task="facts", n=10, method="lora", rank=4, scaling=scaling))
    assert torch.equal(_logits(wrapped), _logits(base))  # B is zero-initialized


def test_trainable_param_count_matches_formula():
    base = _tiny_qwen3()
    # per layer: q 32->32, k 32->16, v 32->16, o 32->32, gate/up 32->64, down 64->32
    per_layer = (32 + 32) + (32 + 16) * 2 + (32 + 32) + (32 + 64) * 3
    assert expected_lora_params(base, 8) == 8 * per_layer * 2
    wrapped = wrap_lora(copy.deepcopy(base), RunConfig(task="facts", n=10, method="lora", rank=8))
    assert sum(p.numel() for p in wrapped.parameters() if p.requires_grad) == 8 * per_layer * 2


def test_gradient_accumulation_matches_full_batch():
    base = _tiny_qwen3().train()
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (8, 10))
    labels = ids.clone()
    labels[:, :4] = -100
    labels[5:, 6:] = -100  # uneven answer lengths across micro-batches: a plain mean of micro losses would be wrong
    inputs = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    results = []
    for micro in (8, 3):
        model = copy.deepcopy(base)
        loss = accumulate_backward(model, inputs, micro, "cpu", None, scaler)
        results.append((loss, [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    (full_loss, full_grads), (micro_loss, micro_grads) = results
    assert micro_loss == pytest.approx(full_loss, rel=1e-5)
    assert len(full_grads) == len(micro_grads) > 0
    for a, b in zip(full_grads, micro_grads):
        assert torch.allclose(a, b, atol=1e-6)


def test_run_config_canonicalizes_irrelevant_fields():
    a = RunConfig(task="sql", n=100, method="full", rank=64, alpha=32.0, scaling="rslora")
    b = RunConfig(task="sql", n=100, method="full")
    assert a == b
    with pytest.raises(ValueError):
        RunConfig(task="sql", n=100, method="lora", rank=0)


def test_length_grouped_batches_cover_each_index_once():
    lengths = [random.Random(i).randint(5, 60) for i in range(1000)]
    batches = length_grouped_batches(lengths, 32, random.Random(0))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(1000))


def test_collate_masks_prompt_and_padding():
    encoded = [([1, 2, 3], [-100, -100, 3]), ([4, 5], [-100, 5])]
    out = collate(encoded, [0, 1], pad_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert out["labels"].tolist() == [[-100, -100, 3], [-100, 5, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
