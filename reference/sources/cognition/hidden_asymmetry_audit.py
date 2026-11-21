#!/usr/bin/env python3
'Finite audit of covariant branch selectors.\n\nThis module deliberately does not import any BST numerical/physical stage.\nIt tests a small mathematical claim used by the cognition/arbitration track:\n\nA deterministic covariant selector on the orbit of a state exists for a\nbranch b iff b is fixed by every symmetry that stabilizes that state.\n\nPermutations are tuples p with p[i] = image of i.'

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


Permutation = tuple[int, ...]
Coloring = tuple[int, ...]


def identity(n: int) -> Permutation:
    return tuple(range(n))


def compose(p: Permutation, q: Permutation) -> Permutation:
    'Return p after q.'
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p: Permutation) -> Permutation:
    out = [0] * len(p)
    for i, image in enumerate(p):
        out[image] = i
    return tuple(out)


def generated_group(n: int, generators: Iterable[Permutation]) -> tuple[Permutation, ...]:
    gens = tuple(generators)
    seen = {identity(n)}
    frontier = [identity(n)]
    while frontier:
        a = frontier.pop()
        for g in gens:
            for candidate in (compose(g, a), compose(a, g)):
                if candidate not in seen:
                    seen.add(candidate)
                    frontier.append(candidate)
    return tuple(sorted(seen))


def cyclic_group(n: int) -> tuple[Permutation, ...]:
    rotation = tuple((i + 1) % n for i in range(n))
    return generated_group(n, (rotation,))


def dihedral_group(n: int) -> tuple[Permutation, ...]:
    rotation = tuple((i + 1) % n for i in range(n))
    reflection = tuple((-i) % n for i in range(n))
    return generated_group(n, (rotation, reflection))


def symmetric_group(n: int) -> tuple[Permutation, ...]:
    cycle = tuple((i + 1) % n for i in range(n))
    swap = tuple(1 if i == 0 else 0 if i == 1 else i for i in range(n))
    return generated_group(n, (cycle, swap))


def act_on_coloring(p: Permutation, colors: Coloring) -> Coloring:
    out = [0] * len(colors)
    for old, new in enumerate(p):
        out[new] = colors[old]
    return tuple(out)


def stabilizer(group: Sequence[Permutation], colors: Coloring) -> tuple[Permutation, ...]:
    return tuple(p for p in group if act_on_coloring(p, colors) == colors)


def fixed_branches(group: Sequence[Permutation]) -> tuple[int, ...]:
    n = len(group[0])
    return tuple(i for i in range(n) if all(p[i] == i for p in group))


def orbits(group: Sequence[Permutation]) -> tuple[tuple[int, ...], ...]:
    n = len(group[0])
    remaining = set(range(n))
    result = []
    while remaining:
        seed = min(remaining)
        orbit = tuple(sorted({p[seed] for p in group}))
        result.append(orbit)
        remaining.difference_update(orbit)
    return tuple(result)


def selector_is_well_defined(
    group: Sequence[Permutation], colors: Coloring, selected_branch: int
) -> bool:
    'Check whether F(p.colors)=p.branch is single-valued on the state orbit.'
    assignments: dict[Coloring, int] = {}
    for p in group:
        transformed_state = act_on_coloring(p, colors)
        transformed_branch = p[selected_branch]
        previous = assignments.get(transformed_state)
        if previous is not None and previous != transformed_branch:
            return False
        assignments[transformed_state] = transformed_branch
    return True


def admissible_selector_branches(
    group: Sequence[Permutation], colors: Coloring
) -> tuple[int, ...]:
    return tuple(
        b for b in range(len(colors)) if selector_is_well_defined(group, colors, b)
    )


def canonical_index_selector_is_covariant(group: Sequence[Permutation]) -> bool:
    "The arbitrary rule 'always choose branch 0' should fail on transitive actions."
    return all(p[0] == 0 for p in group)


@dataclass(frozen=True)
class FamilyAudit:
    family: str
    n: int
    group_order: int
    coloring_count: int
    theorem_mismatches: int
    states_without_covariant_selector: int
    states_with_one_selector_branch: int
    states_with_multiple_selector_branches: int
    uniform_fixed_branches: tuple[int, ...]
    branch_orbits: tuple[tuple[int, ...], ...]


def audit_family(
    family: str, n: int, group: Sequence[Permutation], alphabet_size: int = 2
) -> FamilyAudit:
    mismatches = 0
    none_count = 0
    one_count = 0
    multiple_count = 0
    for colors in itertools.product(range(alphabet_size), repeat=n):
        colors = tuple(colors)
        by_stabilizer = fixed_branches(stabilizer(group, colors))
        by_construction = admissible_selector_branches(group, colors)
        if by_stabilizer != by_construction:
            mismatches += 1
        if not by_construction:
            none_count += 1
        elif len(by_construction) == 1:
            one_count += 1
        else:
            multiple_count += 1
    uniform = (0,) * n
    return FamilyAudit(
        family=family,
        n=n,
        group_order=len(group),
        coloring_count=alphabet_size**n,
        theorem_mismatches=mismatches,
        states_without_covariant_selector=none_count,
        states_with_one_selector_branch=one_count,
        states_with_multiple_selector_branches=multiple_count,
        uniform_fixed_branches=fixed_branches(stabilizer(group, uniform)),
        branch_orbits=orbits(group),
    )


def run_audit(max_n: int = 7, alphabet_size: int = 2) -> dict:
    families = []
    constructors = (
        ("cyclic", cyclic_group),
        ("dihedral", dihedral_group),
        ("symmetric", symmetric_group),
    )
    for n in range(2, max_n + 1):
        for name, constructor in constructors:
            families.append(audit_family(name, n, constructor(n), alphabet_size))

    return {
        "scope": {
            "n_min": 2,
            "n_max": max_n,
            "state_alphabet_size": alphabet_size,
            "families": [name for name, _ in constructors],
            "claim": "exhaustive over all colorings in the declared finite scope",
        },
        "criteria": {
            "stabilizer_theorem_has_zero_mismatches": all(
                row.theorem_mismatches == 0 for row in families
            ),
            "uniform_transitive_states_have_no_fixed_branch": all(
                not row.uniform_fixed_branches for row in families
            ),
            "arbitrary_index_tiebreak_breaks_covariance": all(
                not canonical_index_selector_is_covariant(
                    dict(constructors)[row.family](row.n)
                )
                for row in families
            ),
            "finite_invariant_randomization_exists": all(
                len(row.branch_orbits) >= 1 for row in families
            ),
        },
        "families": [asdict(row) for row in families],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=7)
    parser.add_argument("--alphabet-size", type=int, default=2)
    args = parser.parse_args()
    report = run_audit(args.max_n, args.alphabet_size)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(report["criteria"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
