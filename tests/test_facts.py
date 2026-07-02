from lorathresh.data import facts as F


def test_exact_count_and_determinism():
    a = F.build_facts(1003, seed=0)
    assert len(a) == 1003
    assert a == F.build_facts(1003, seed=0)


def test_smaller_world_is_prefix_of_larger():
    small, big = F.build_facts(1000), F.build_facts(4000)
    assert big[:1000] == small


def test_train_and_eval_templates_disjoint():
    for templates in F.TEMPLATES.values():
        train, held = templates[: F.N_TRAIN_TEMPLATES], templates[F.N_TRAIN_TEMPLATES:]
        assert held and not set(train) & set(held)
    facts = F.build_facts(500)
    train_prompts = {e.prompt for e in F.train_examples(facts)}
    assert not train_prompts & {e.prompt for e in F.eval_examples(facts)}


def test_train_examples_cover_every_fact():
    facts = F.build_facts(250)
    ex = F.train_examples(facts)
    assert len(ex) == 250 * F.N_TRAIN_TEMPLATES
    assert {e.fact_id for e in ex} == {f.fact_id for f in facts}


def test_names_unique():
    facts = F.build_facts(16000)
    names = {f.name for f in facts}
    assert len(names) == -(-16000 // len(F.ATTRIBUTES))


def test_score():
    assert F.score(" Velford\nQuestion: ...", "velford")
    assert F.score("1942.", "1942")
    assert not F.score("Velfordport", "Velford")
