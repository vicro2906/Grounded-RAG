"""ABBREVIATION -> full name, loaded from the specialty profiles.

The guidelines use siglas and full names interchangeably («DTG» appears 144 times,
«dolutegravir» 23), so treating the two as one term is what lets a question written either way
match a fragment written the other. This dictionary has the widest blast radius in the project:
seven system prompts embed it, `evidence.py` uses it to catch a citation that swapped one drug
for another, and the graph modes use it to collapse the six spellings of one concept into a
single node.

TWO VIEWS, AND THE SPLIT IS A SAFETY DECISION.

`ABBREVIATIONS` is the UNION over every specialty. It is what the integrity checks use, because
there more patterns can only make the check STRICTER: `evidence` compares the drug tokens of a
quote against those of the sentence it matched, so an abbreviation it does not know is one it
cannot notice being swapped.

`for_specialty(id)` is the per-specialty view, and it is what the prompts and query expansion
use. Two reasons. Size: seven prompts each carrying every specialty's dictionary does not scale
past the first few. Correctness: `expand_abbrevs` REWRITES the query, so a sigla that means one
thing in cardiology and another here would inject a wrong drug name into the search.

The values stay in Spanish (they are guideline terms) and now live in
`data/specialties/<id>.toml` rather than in this file, so adding a specialty is a data change.
"""
from functools import lru_cache

import corpus


@lru_cache(maxsize=None)
def for_specialty(specialty_id: str) -> dict:
    """The abbreviations of ONE specialty: what the prompts and query expansion should see."""
    return dict(corpus.specialty(specialty_id).abbreviations)


@lru_cache(maxsize=1)
def _union() -> dict:
    """Every specialty's abbreviations merged.

    A collision between specialties keeps the FIRST definition and is not an error: the union
    only ever feeds checks that get stricter with more patterns, so an imperfect merge degrades
    to "one spelling is not recognised", never to "the wrong drug name was substituted" — that
    risk lives in `for_specialty`, which is per-specialty precisely to avoid it."""
    merged: dict = {}
    for specialty_id in corpus.specialties():
        for abbr, name in corpus.specialty(specialty_id).abbreviations.items():
            merged.setdefault(abbr, name)
    return merged


ABBREVIATIONS = _union()
