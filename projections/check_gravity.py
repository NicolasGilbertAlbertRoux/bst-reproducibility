"""Compare the two default gravity trajectories with an independent DOP853 integration."""
import json
from itertools import combinations

import numpy as np
from scipy.integrate import solve_ivp

from gravity import simulate


def compare(preparation):
    trajectory = simulate(preparation)
    bodies = trajectory['bodies']
    samples = trajectory['records']
    count = len(bodies)
    masses = np.array([body['mass'] for body in bodies])
    radii = [body['radius'] for body in bodies]
    initial_positions = np.array(samples[0]['xy'])
    initial_velocities = np.array([[0, -70], [0, 151], [-118, 2]][:count], dtype=float)
    initial_velocities -= np.average(initial_velocities, weights=masses, axis=0)
    pairs = list(combinations(range(count), 2))
    gravitational_constant = 6.67430e-11

    def derivatives(time, state):
        pos = state[:2*count].reshape(count, 2)
        velocities = state[2*count:]
        acceleration = np.zeros_like(pos)
        for first, second in pairs:
            displacement = pos[second] - pos[first]
            force_per_mass_product = gravitational_constant * displacement / np.linalg.norm(displacement)**3
            acceleration[first] += masses[second] * force_per_mass_product
            acceleration[second] -= masses[first] * force_per_mass_product
        return np.r_[velocities, acceleration.ravel()]

    def surface_contact(first, second):
        def event(time, state):
            pos = state[:2*count].reshape(count, 2)
            return np.linalg.norm(pos[first] - pos[second]) - radii[first] - radii[second]
        event.terminal = True
        event.direction = -1
        return event

    reference = solve_ivp(
        derivatives, (0, 96*3600), np.r_[initial_positions.ravel(), initial_velocities.ravel()],
        method='DOP853', rtol=3e-12, atol=1e-8, dense_output=True,
        events=[surface_contact(i, j) for i, j in pairs],
    )
    assert reference.success
    shared = [sample for sample in samples if sample['t'] <= reference.t[-1]]
    positions = np.array([sample['xy'] for sample in shared])
    reference_positions = reference.sol([s['t'] for s in shared])[:2*count].T.reshape(-1, count, 2)
    difference = float(np.linalg.norm(positions-reference_positions, axis=2).max())
    span_difference = float(abs(reference.t[-1]-samples[-1]['t']))
    final_contact = next(([i+1,j+1] for (i,j),times in zip(pairs, reference.t_events) if len(times)), None)
    assert final_contact == samples[-1]['contact']
    assert difference < 3500 and span_difference < 6
    return {'sampled_position_difference_m': difference, 'duration_difference_s': span_difference,
            'hours': samples[-1]['t']/3600, 'contact': final_contact}


if __name__ == '__main__':
    print(json.dumps({key: compare(key) for key in ('binary', 'triple')}, indent=2))
