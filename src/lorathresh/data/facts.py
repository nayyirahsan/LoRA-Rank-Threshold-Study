"""Synthetic fact-injection task.

Fictional people get randomly assigned attribute values. Each fact is trained under TRAIN
question templates and evaluated under disjoint EVAL templates, so the score measures
knowledge stored in the weights rather than recall of a memorized string. Because the
people are fictional, the base model scores ~0 and gap closure is well defined.

Information content is controlled by n_facts, the x-axis of the H1 experiment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

_SYLLABLES = [
    "vor", "el", "sk", "tha", "mir", "quo", "zen", "bra", "dul", "fen", "gri", "hol",
    "ish", "jar", "kel", "lum", "nov", "ost", "pra", "rin", "sal", "tov", "urb", "vex",
    "wil", "yar", "zor", "cad", "dre", "fal",
]

MAJORS = [
    "Astrophysics", "Linguistics", "Marine Biology", "Architecture", "Economics",
    "Philosophy", "Chemical Engineering", "Art History", "Statistics", "Geology",
    "Anthropology", "Music Theory", "Neuroscience", "Agronomy", "Classics",
    "Materials Science", "Oceanography", "Political Science", "Epidemiology", "Metallurgy",
]

# attribute -> question templates. The first N_TRAIN_TEMPLATES are used for training,
# the rest only at eval time.
TEMPLATES: dict[str, list[str]] = {
    "birth_city": [
        "Where was {name} born?",
        "What is the birthplace of {name}?",
        "In which city was {name} born?",
        "{name} was born in which city?",
        "Which city is {name} originally from?",
        "Name the city where {name} was born.",
    ],
    "employer": [
        "Where does {name} work?",
        "Who is {name}'s employer?",
        "Which company employs {name}?",
        "{name} works for which company?",
        "What organization is {name} employed by?",
        "Name the company {name} works at.",
    ],
    "birth_year": [
        "In what year was {name} born?",
        "What is {name}'s birth year?",
        "When was {name} born?",
        "{name} was born in which year?",
        "Which year did {name} come into the world?",
        "State the year of {name}'s birth.",
    ],
    "university": [
        "Which university did {name} attend?",
        "Where did {name} go to college?",
        "What school did {name} graduate from?",
        "{name} studied at which university?",
        "Name the university {name} attended.",
        "Which institution awarded {name}'s degree?",
    ],
    "major": [
        "What did {name} study?",
        "What was {name}'s major?",
        "Which field did {name} major in?",
        "{name} earned a degree in which subject?",
        "What subject did {name} specialize in at university?",
        "Name {name}'s field of study.",
    ],
}
ATTRIBUTES = list(TEMPLATES)
N_TRAIN_TEMPLATES = 4


@dataclass(frozen=True)
class Fact:
    fact_id: int
    name: str
    attribute: str
    value: str


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    fact_id: int


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _word(rng: random.Random, n_syllables: int) -> str:
    return "".join(rng.choice(_SYLLABLES) for _ in range(n_syllables)).capitalize()


def _unique_words(rng: random.Random, count: int, make) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < count:
        w = make()
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _value_pools(rng: random.Random) -> dict[str, list[str]]:
    cities = _unique_words(
        rng, 200, lambda: _word(rng, 2) + rng.choice(["ford", "ville", "port", "holm", "stad"])
    )
    return {
        "birth_city": cities,
        "employer": _unique_words(
            rng, 200, lambda: f"{_word(rng, 2)} {rng.choice(['Dynamics', 'Labs', 'Holdings', 'Systems'])}"
        ),
        "birth_year": [str(y) for y in range(1900, 2000)],
        "university": [f"University of {c}" for c in rng.sample(cities, 100)],
        "major": MAJORS,
    }


def build_facts(n_facts: int, seed: int = 0) -> list[Fact]:
    """Deterministically generate exactly n_facts facts (people x attributes, in person order).

    The seed fixes the "world", so facts for N=1k are a prefix of facts for N=4k.
    """
    # Separate streams for pools, names, and each person's values. With one shared stream,
    # generating more names would shift every value, and the N=1k world would not be a
    # subset of the N=4k world (a confound for H1).
    pools = _value_pools(random.Random(f"{seed}:pools"))
    name_rng = random.Random(f"{seed}:names")
    n_people = -(-n_facts // len(ATTRIBUTES))
    names = _unique_words(name_rng, n_people, lambda: f"{_word(name_rng, 2)} {_word(name_rng, 3)}")
    facts: list[Fact] = []
    for i, name in enumerate(names):
        value_rng = random.Random(f"{seed}:person:{i}")
        for attr in ATTRIBUTES:
            if len(facts) == n_facts:
                return facts
            facts.append(Fact(len(facts), name, attr, value_rng.choice(pools[attr])))
    return facts


def train_examples(facts: list[Fact]) -> list[Example]:
    return [
        Example(format_prompt(t.format(name=f.name)), f.value, f.fact_id)
        for f in facts
        for t in TEMPLATES[f.attribute][:N_TRAIN_TEMPLATES]
    ]


def eval_examples(facts: list[Fact], max_facts: int = 1000, seed: int = 0) -> list[Example]:
    """One held-out-template question per sampled fact."""
    rng = random.Random(seed)
    sampled = rng.sample(facts, min(max_facts, len(facts)))
    return [
        Example(
            format_prompt(rng.choice(TEMPLATES[f.attribute][N_TRAIN_TEMPLATES:]).format(name=f.name)),
            f.value,
            f.fact_id,
        )
        for f in sampled
    ]


def score(prediction: str, answer: str) -> bool:
    """Exact match on the first generated line, case- and whitespace-insensitive."""
    first_line = prediction.strip().split("\n", 1)[0].strip().rstrip(".")
    return first_line.lower() == answer.lower()
