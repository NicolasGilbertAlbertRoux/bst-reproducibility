"""Build four crystallographic reference structures and a reduced harmonic displacement sequence."""
import argparse
import base64
import json
import zlib
from itertools import product
from pathlib import Path

import numpy as np

FCC = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
ICE = [[1/3, 2/3, 1/16], [2/3, 1/3, 9/16], [2/3, 1/3, 15/16], [1/3, 2/3, 7/16]]
AVOGADRO = 6.02214076e23
# Two SF6 centres per conventional cell, from the adopted solid molar volume at the triple point.
SF6_EDGE = (2 * 64.10e24 / AVOGADRO) ** (1/3)
REFERENCES = {
    'H2O': ('Ice Ih', [4.523, 4.523, 7.367], ICE, [3, 3, 2], 4, 9623),
    'CO2': ('Pa-3', [5.578]*3, FCC, [3, 3, 3], 12, 9624),
    'SF6': ('Im-3m', [SF6_EDGE]*3, [[0, 0, 0], [.5, .5, .5]], [3, 3, 3], 8, 9625),
    'Kr': ('Fm-3m', [5.646]*3, FCC, [3, 3, 3], 12, 9626),
}

def nearest_graph(points, box, degree):
    delta = points[None, :, :] - points[:, None, :]
    distance2 = np.full(delta.shape[:2], np.inf)
    shortest = np.zeros_like(delta)
    for offset in product((-1, 0, 1), repeat=3):
        trial = delta + np.array(offset) @ box
        d2 = np.sum(trial * trial, axis=-1)
        shorter = d2 < distance2
        shortest[shorter] = trial[shorter]
        distance2[shorter] = d2[shorter]
    np.fill_diagonal(distance2, np.inf)
    neighbours = np.argsort(distance2, axis=1)[:, :degree]
    graph = np.zeros_like(distance2, dtype=bool)
    graph[np.arange(len(points))[:, None], neighbours] = True
    if not np.array_equal(graph, graph.T):
        raise ValueError('The periodic first neighbour shell must be reciprocal')
    return neighbours, shortest, np.array(np.nonzero(np.triu(graph, 1))).T

def unit(v):
    return v / np.linalg.norm(v)

def water_axes(neighbours, directions, rng):
    # Orient an Euler circuit of the four-coordinated oxygen network: two donors and
    # two acceptors per oxygen and one donor per bond, including periodic boundaries.
    available = [list(map(int, rng.permutation(row))) for row in neighbours]
    used = set()
    stack, circuit = [0], []
    while stack:
        site = stack[-1]
        while available[site]:
            other = available[site].pop()
            edge = tuple(sorted((site, other)))
            if edge not in used:
                used.add(edge)
                stack.append(other)
                break
        else:
            circuit.append(stack.pop())
    if len(circuit) != 2*len(neighbours)+1 or circuit[0] != circuit[-1]:
        raise ValueError('Incomplete ice network')
    donors = [[] for _ in neighbours]
    for a, b in zip(circuit, circuit[1:]):
        donors[a].append(b)
    axes = []
    for i, pair in enumerate(donors):
        if len(pair) != 2:
            raise ValueError('Ice rule broken')
        left, right = [unit(directions[i,j]) for j in pair]
        x, y = unit(right-left), -unit(right+left)
        axes.append([x,y,np.cross(x,y)])
    return axes, donors

def crystal_axes(symbol, n, neighbours, direction, rng):
    if symbol == 'H2O':
        return water_axes(neighbours, direction, rng)
    if symbol == 'CO2':
        # Pa-3: carbon on 4a, oxygen on 8c, with molecular axes along body diagonals.
        diagonals = [[1,1,1], [-1,1,-1], [-1,-1,1], [1,-1,-1]]
        axes=[]
        for j in range(n):
            x=unit(np.asarray(diagonals[j%4], dtype=float))
            y=unit(np.cross([0,0,1],x))
            axes.append([x,y,np.cross(x,y)])
        return axes, None
    if symbol == 'SF6':
        # A seeded isotropic orientation snapshot represents disorder in the plastic phase.
        axes=[]
        for _ in range(n):
            x=unit(rng.normal(size=3))
            y=unit(np.cross(x,rng.normal(size=3)))
            axes.append([x,y,np.cross(x,y)])
        return axes, None
    return np.tile(np.eye(3),(n,1,1)), None

def vibrations(n, edges, rng):
    # Identical unit mass and vector spring coefficient; times and amplitudes are illustrative.
    left, right = edges.T
    u = np.zeros((n,3))
    v = rng.normal(size=u.shape)
    v -= v.mean(axis=0)
    v *= .14/np.sqrt(np.mean(np.sum(v*v,axis=1)))
    def acceleration(pos):
        dif = pos[right]-pos[left]
        acc = np.zeros_like(pos)
        np.add.at(acc,left,dif)
        np.add.at(acc,right,-dif)
        return acc
    acc = acceleration(u)
    states, energy = [], []
    dt = .025
    for step in range(72*4):
        if step%4 == 0:
            states.append(np.column_stack((u,np.linalg.norm(v,axis=1))))
            energy.append((np.sum(v*v) + np.sum((u[left]-u[right])**2))/2)
        v += .5*dt*acc
        u += dt*v
        acc = acceleration(u)
        v += .5*dt*acc
    if np.max(np.abs(v.mean(axis=0))) > 1e-11:
        raise ValueError('Nonzero centre-of-mass drift')
    drift = (max(energy)-min(energy))/energy[0]
    if drift > .005:
        raise ValueError('Excessive harmonic energy drift')
    return np.round(states,7).astype('<f4'), drift

def build():
    output={}
    for symbol,(phase, lengths, motif, repeats, degree, seed) in REFERENCES.items():
        cell=np.diag(lengths)
        if symbol == 'H2O':
            cell[1]=[-lengths[1]/2,lengths[1]*np.sqrt(3)/2,0]
        boxes=np.array(list(np.ndindex(*repeats)))
        points=(boxes[:,None,:]+np.array(motif)).reshape(-1,3)@cell
        big=np.diag(repeats)@cell
        neighbours,directions,edges=nearest_graph(points,big,degree)
        rng=np.random.default_rng(seed)
        axes,donors=crystal_axes(symbol,len(points),neighbours,directions,rng)
        u,drift=vibrations(len(points),edges,rng)
        points-=points.mean(axis=0)
        record=dict(phase=phase,cell=np.round(cell,7).tolist(),repeats=repeats,degree=degree,n=len(points),
                    centers=np.round(points,7).tolist(),axes=np.round(axes,7).tolist(),seed=seed,
                    relativeEnergySpan=round(float(drift),9),frames=len(u),step=.1,
                    motion=base64.b64encode(zlib.compress(u.tobytes(),6)).decode('ascii'))
        # Choose an interior site so the displayed shell needs no periodic wrapping.
        center_index = int(np.argmin(np.linalg.norm(points, axis=1)))
        shell = neighbours[center_index]
        if not np.allclose(points[shell]-points[center_index], directions[center_index,shell]):
            raise ValueError('The displayed first shell crosses a supercell boundary')
        record['shell'] = [center_index, *map(int,shell)]
        if donors is not None:
            record['donors']=donors
        output[symbol]=record
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.write_text('window.BST_CRYSTALS = '+json.dumps(build(),allow_nan=False,separators=(',',':'))+';\n')
