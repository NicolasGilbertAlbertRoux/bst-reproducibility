'Check the additivity of the changes in action during S2 scan U(1).'
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def action(field):
    'Compact action of Wilson, beta-1, periodic edges.'
    total = 0.0
    for mu, nu in combinations(range(4), 2):
        angle = (field[mu] + np.roll(field[nu], -1, axis=mu)
                 - np.roll(field[mu], -1, axis=nu) - field[nu])
        total += np.sum(1.0 - np.cos(angle))
    return float(total)


def check():
    delta = 0.4
    cost = 1.0 - np.cos(delta)
    results = {}
    for separation in (1, 2):
        a = np.zeros((4, 6, 6, 6, 6))
        b = np.zeros_like(a)
        a[0, 0, 0, 0, 0] = delta
        b[0, 0, separation, 0, 0] = delta
        sum_field = action(a) + action(b)
        simultaneous = action(a + b)
        expected = (10 if separation == 1 else 12) * cost
        np.testing.assert_allclose([sum_field, simultaneous], [12 * cost, expected], rtol=0, atol=1e-13)
        results[str(separation)] = {'sum_separee': sum_field, 'variation_simultaneous': simultaneous}
    return {'model': 'U(1) compact, zero initial field, periodic network 6-4',
            "beta": 1.0, "rotation_radians": delta, 'separations_in_mailles': results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(check(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
