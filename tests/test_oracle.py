import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from lorathresh.oracle import DeltaSVD, energy_captured, energy_rank, target_linears  # noqa: E402


def _tiny_qwen3():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
    )
    return transformers.Qwen3ForCausalLM(cfg).eval()


def _perturbed(base, rank: int):
    """Copy of base with a planted rank-`rank` update on every target linear."""
    tuned = copy.deepcopy(base)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for lin in target_linears(tuned).values():
            out_f, in_f = lin.weight.shape
            lin.weight += torch.randn(out_f, rank, generator=g) @ torch.randn(rank, in_f, generator=g) * 0.05
    return tuned


def _logits(model):
    ids = torch.arange(10).unsqueeze(0)
    with torch.no_grad():
        return model(ids).logits


def test_rank_zero_reproduces_base_and_full_rank_reproduces_tuned():
    base = _tiny_qwen3()
    tuned = _perturbed(base, rank=8)
    oracle = DeltaSVD(base, tuned)
    work = copy.deepcopy(base)

    oracle.apply(work, 0)
    assert torch.allclose(_logits(work), _logits(base), atol=1e-5)

    oracle.apply(work, oracle.max_rank)
    assert torch.allclose(_logits(work), _logits(tuned), atol=1e-4)


def test_energy_rank_recovers_planted_rank():
    base = _tiny_qwen3()
    oracle = DeltaSVD(base, _perturbed(base, rank=3))
    assert set(oracle.energy_ranks(fraction=0.999).values()) <= {1, 2, 3}
    for s in oracle.singular_values.values():
        assert energy_captured(s, 3) == pytest.approx(1.0, abs=1e-6)


def test_weight_error_decreases_with_rank():
    # Eckart-Young guarantees this per matrix, in weight space. It does NOT guarantee that
    # logit error falls with rank: on this model logit error rose from r=1 to r=4, since
    # per-layer truncation errors interact through nonlinearities. See docs/engineering_log.md #5.
    base = _tiny_qwen3()
    tuned = _perturbed(base, rank=16)
    oracle = DeltaSVD(base, tuned)
    work = copy.deepcopy(base)
    tuned_linears = target_linears(tuned)
    errors = []
    for r in (0, 1, 4, 8, 16):
        oracle.apply(work, r)
        errors.append(
            sum((lin.weight - tuned_linears[n].weight).square().sum().item() for n, lin in target_linears(work).items())
        )
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-8
    assert torch.allclose(_logits(work), _logits(tuned), atol=1e-4)


def test_energy_rank_edge_cases():
    assert energy_rank(torch.tensor([1.0, 0.0, 0.0])) == 1
    assert energy_rank(torch.zeros(4)) == 0
    assert energy_rank(torch.ones(10), fraction=0.9) == 9
