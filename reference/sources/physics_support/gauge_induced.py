'Build curved links from local subspaces of rank three.'
from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
sys.path.insert(0,str(INPUT))
from cosmology_R449 import write
from gauge_su3 import I, adj, exponential, loop, random_algebra, shift, trace


def projectors(x, coefficients=(1.,1.,1.)):
    x0,x1,x2 = np.moveaxis(x,-1,0)
    a = coefficients[0]*(.24*np.sin(x0)+.13*np.cos(x2))
    b = coefficients[1]*(.29*np.sin(x1)+.08*np.cos(x0))
    c = coefficients[2]*(.25*np.sin(x2)+.11*np.sin(x1+x0))
    # (sqrt(1-a²-b²-c²),a,b,c) is a smooth S³ chart. The complex assignment
    # to the normal is the additional choice of this induced connection.
    normal = np.stack((np.sqrt(1-a*a-b*b-c*c),a+1j*b,b+1j*c,c+1j*a),axis=-1)
    normal /= np.linalg.norm(normal,axis=-1,keepdims=True)
    return np.eye(4)-normal[...,None]*normal[...,None,:].conj()


def polar(matrix):
    u,s,vh = np.linalg.svd(matrix,full_matrices=False)
    assert np.min(s) > .5
    return u@vh


def local_frame(x, coefficients=(1.,1.,1.)):
    p = projectors(x,coefficients)
    return polar(p[...,1:])


def links(frame):
    m = np.array([polar(adj(frame)@shift(frame,k)) for k in range(3)])
    phase = np.angle(np.linalg.det(m))
    assert np.max(abs(phase)) < .5
    return m*np.exp(-1j*phase/3)[...,None,None]


def traceless(m):
    return m-trace(m)[...,None,None]*I/3


def run(n):
    step = 2*np.pi/n
    x = np.stack(np.meshgrid(*[step*np.arange(n)]*3,indexing="ij"),axis=-1)
    frame = local_frame(x)
    u = links(frame)
    g = exponential(random_algebra(np.random.default_rng(453+n),(n,n,n,3,3),.4))
    rotated = links(frame@g)
    predicted = np.array([adj(g)@m@shift(g,k) for k,m in enumerate(u)])
    error = float(np.max(abs(rotated-predicted)))
    q_err = float(np.max(abs(adj(u)@u-I)))
    det_err = float(np.max(abs(np.linalg.det(u)-1)))
    assert max(error,q_err,det_err)<1e-13
    defects, magnitudes, plaquettes = [], [], []
    delta = 2e-5
    for a,b in ((0,1),(0,2),(1,2)):
        holonomy = loop(u,a,b)
        plaquettes.append(holonomy)
        lattice = (holonomy-adj(holonomy))/(2*step**2)
        dp_a,dp_b=[(projectors(x+delta*np.eye(3)[k])-projectors(x-delta*np.eye(3)[k]))/(2*delta)
                   for k in (a,b)]
        continuum=traceless(adj(frame)@(dp_a@dp_b-dp_b@dp_a)@frame)
        magnitudes.append(float(np.max(abs(continuum))))
        defects.append(float(np.linalg.norm(lattice-continuum)/np.linalg.norm(continuum)))
    first,second=plaquettes[:2]
    comm = float(np.max(abs(first@second-second@first)))/step**4
    assert min(magnitudes)>.02 and comm>.001
    return dict(N=n,step_field=step,covariance=error,unitarite=q_err,determinant=det_err,
                deviation_relative_curvature=defects,max_curvature=magnitudes,commutator_normalized=comm)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--sortie",type=Path,default=ROOT/'results')
    cfg=parser.parse_args()
    stages = [run(n) for n in (8,16,24)]
    error=np.array([r['deviation_relative_curvature'] for r in stages])
    assert np.all(error[1:]<.8*error[:-1])
    tail = [r['commutator_normalized'] for r in stages[-2:]]
    assert .8<tail[1]/tail[0]<1.2
    write(cfg.output/"JAUGE_INDUITE_R453.json",dict(revision=453,grids=stages,
          derivation=('Link defined by the unit factor of F(x)--F(y), then determining to 1. '
                      'F is an orthonormal base of the complement to a complex normal vector. The curvature of the '
                      'projected connection is the traceless part of F------------------------------------------------------'),
          scope=('The projection allows curved and non-switching links from a field in a map '
                  'S3, with a complex dip chosen for the test. This kinematic configuration does not '
                  'demonstrates neither the uniqueness of this plunge nor an effective dynamic of Yang-Mills. Verification '
                  'Gauss with a dynamic substrate of the separate bank relates to independent links.')))
    print(stages,flush=True)


if __name__=="__main__":
    main()
