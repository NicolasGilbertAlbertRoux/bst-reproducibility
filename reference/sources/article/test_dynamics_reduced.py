'Two preparations S14c, same separation and same speed visible at baseline.'
import argparse
import json
from pathlib import Path

import numpy as np
from example_battement import simulate


def compare():
    series = []
    for stiffness in (3.0, 0.0):
        for dt in (0.004, 0.002, 0.001, 0.0005):
            n = round(2.0 / dt)
            distances = {}
            trials = {}
            for sign, name in ((1, "sortant"), (-1, "entrant")):
                traj, drift = simulate(direction_battement=sign, stiffness_contact=stiffness,
                                      dt=dt, nb_step=n, sample=1)
                distances[name] = traj[:, 1]
                np.testing.assert_allclose(traj[0, [0, 1, 2, 3, 6]], [0, 2, 0, 0, 0.1],
                                           rtol=0, atol=1e-14)
                trials[name] = {"d_a_1": float(traj[n//2, 1]), "d_a_2": float(traj[-1, 1]),
                               'drift_energy_max': float(drift)}
            deviation = distances["sortant"] - distances["entrant"]
            if stiffness == 0:
                np.testing.assert_array_equal(distances["sortant"], distances["entrant"])
            series.append({'stiffness_contact': stiffness, "dt": dt, 'trials': trials,
                           'deviation_d_a_1': float(deviation[n//2]), 'deviation_d_a_2': float(deviation[-1])})
    return {'model': 'S14c, reduced units of Chapter 2',
            "preparations": "d=2, p_d=0, b=0, p_b=+sqrt(0.18) ou -sqrt(0.18)",
            "observateur": "séparation d des centres et vitesse initiale p_d/m",
            "horizon": 2.0, "series": series}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='JSON control file')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compare(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
