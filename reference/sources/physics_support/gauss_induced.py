'Integrate the dynamics of links actually induced by mobile fields.'
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor,cho_solve

from gauge_induced import ROOT,local_frame,links
from gauge_su3 import I,adj,algebra,exponential,loop,random_algebra,shift,trace
from cosmology_R449 import write


def generators():
    a,b,c=np.eye(3)
    g=[np.diag([1.,-1.,0.])/np.sqrt(2),np.diag([1.,1.,-2.])/np.sqrt(6)]
    for i,j in ((a,b),(a,c),(b,c)):
        p=np.outer(i,j)
        g.extend(((p+p.T)/np.sqrt(2),1j*(p-p.T)/np.sqrt(2)))
    return np.asarray(g,dtype=complex)


BASIS=generators()
MASSES=np.array([.55,.59,.64,.65])


def atlas(n):
    return np.stack(np.meshgrid(*[np.arange(n)*2*np.pi/n]*3,indexing="ij"),axis=-1)


def fields(x,coordinates):
    xx,yy,zz=np.moveaxis(x,-1,0)
    frame=local_frame(x,coordinates[:3])
    phase=coordinates[-1]
    spin=np.stack((.12*np.exp(1j*phase*np.sin(xx)),.13*np.exp(-1j*phase*np.cos(yy)),
                   .11*np.exp(1j*phase*np.cos(zz+xx))),axis=-1)
    psi=spin[...,None]*I
    potential=.48*np.sum((coordinates-np.array([1.,1.,1.,.12]))**2)
    return links(frame),psi,potential


def electric_coordinates(w):
    return np.einsum("...ij,aji->...a",w,BASIS,optimize=True).real


def stack(electric,matter):
    return np.r_[electric_coordinates(electric).ravel(),matter.real.ravel(),matter.imag.ravel()]


def operator(u,psi):
    n=psi.shape[0]
    shape=psi.shape[:3]
    size=n**3
    order=np.arange(size).reshape(shape)
    cols=8*order.ravel()[:,None]+np.arange(8)
    base=np.arange(8*size).reshape(size,8)
    D=np.zeros((3*8*size+18*size,8*size))
    for index,link in enumerate(u):
        rows=8*size*index+base
        right=8*shift(order,index).ravel()[:,None]+np.arange(8)
        moved=(link.reshape(size,1,3,3)@BASIS@adj(link.reshape(size,1,3,3)))
        rotations=np.swapaxes(electric_coordinates(moved),1,2)
        D[rows[:,:,None],cols[:,None,:]]+=np.eye(8)
        D[rows[:,:,None],right[:,None,:]]-=rotations
    m=1j*BASIS@psi.reshape(size,1,3,3)
    real=np.arange(9*size).reshape(size,9)+3*8*size
    D[real[:,:,None],cols[:,None,:]]=m.real.reshape(size,8,9).transpose(0,2,1)
    D[(real+9*size)[:,:,None],cols[:,None,:]]=m.imag.reshape(size,8,9).transpose(0,2,1)
    return D


def flux(u,psi,of,dpsi):
    return stack(algebra(of@adj(u)),dpsi)


def eliminate(D,w):
    factor=cho_factor(D.T@D,lower=True)
    temporal=-cho_solve(factor,D.T@w)
    horizontal=w+D@temporal
    gauge_residual=D.T@horizontal
    assert np.max(abs(gauge_residual))<2e-11
    return horizontal,temporal,float(np.max(abs(gauge_residual)))


def reduced(x,c,eps=1e-5):
    u,psi,potential=fields(x,c)
    p1=sum(np.sum(3.-trace(loop(u,a,b)).real) for a,b in ((0,1),(0,2),(1,2)))
    p2=sum(np.linalg.norm(psi-u[j]@shift(psi,j))**2 for j in range(3))
    columns=[]
    for coordinate in np.eye(4):
        plus,minus=[fields(x,c+d*eps*coordinate) for d in (1.,-1.)]
        columns.append(flux(u,psi,(plus[0]-minus[0])/(2*eps),(plus[1]-minus[1])/(2*eps)))
    w=np.column_stack(columns)
    D=operator(u,psi)
    h,temporal,gauss=eliminate(D,w)
    n=x.shape[0]
    M=h.T@h/n**3+np.diag(MASSES)
    V=potential+(p1+.6*p2)/n**3
    return V,M,gauss,(u,psi,w,D,temporal)


def gauge_transform_check(x,a,v):
    u,p,w,D,control = reduced(x,a)[-1]
    np.testing.assert_allclose(D.T@(w+D@control),0.,atol=2e-11)
    g=exponential(random_algebra(np.random.default_rng(578),p.shape,.35))
    freq=random_algebra(np.random.default_rng(579),p.shape,.1)
    dg=1j*freq@g
    eps=1e-5

    def transformed(t):
        u,p,_=fields(x,a+t*v)
        G=exponential(t*freq)@g
        return np.array([G@uh@adj(shift(G,j)) for j,uh in enumerate(u)]),G@p

    u_t,p_t=transformed(0.)
    acc=transformed(eps)
    rev=transformed(-eps)
    w_t=flux(u_t,p_t,*( (A-B)/(2*eps) for A,B in zip(acc,rev,strict=True)))
    D_t=operator(u_t,p_t)
    h_t,temporal,err=eliminate(D_t,w_t)
    h=w@v+D@(control@v)
    weights = np.einsum("...a,aij->...ij",(control@v).reshape(*x.shape[:3],8),BASIS)
    transformed_a=g@weights@adj(g)+1j*dg@adj(g)
    diff=float(np.max(abs(electric_coordinates(transformed_a).ravel()-temporal)))
    assert diff<3e-8
    np.testing.assert_allclose(h@h,h_t@h_t,rtol=2e-7,atol=1e-12)
    return dict(deviation_A0=diff,defect_Gauss_transforme=err,
                source_Gauss_initial=float(np.max(abs(D.T@(w@v)))) )


def run(n):
    x=atlas(n)

    def rhs(t,y):
        c,v=y.reshape(2,4)
        _,M,_,_=reduced(x,c)
        eps=3e-5
        dV,dM=[],[]
        for axis in np.eye(4):
            p,q=[reduced(x,c+a*eps*axis)[:2] for a in (1.,-1.)]
            dV.append((p[0]-q[0])/(2*eps))
            dM.append((p[1]-q[1])/(2*eps))
        deriv=np.asarray(dM)
        cforce=.5*np.einsum("ijk,j,k->i",deriv,v,v)-np.einsum("jik,j,k->i",deriv,v,v)
        return np.r_[v,np.linalg.solve(M,cforce-np.asarray(dV))]

    base=np.array([.92,1.05,.90,.08,.085,-.041,.055,.12])
    times=np.linspace(0.,.6,11)
    answers=[]
    for step in (.12,.06):
        calc=solve_ivp(rhs,(0.,times[-1]),base,t_eval=times,method="DOP853",rtol=2e-8,atol=3e-10,max_step=step)
        assert calc.success
        energies,gauss=[],[]
        for record in calc.y.T:
            c,v=record.reshape(2,4)
            V,M,err,_=reduced(x,c)
            energies.append(float(V+.5*v@M@v))
            gauss.append(err)
        answers.append(dict(step_max=step,drift_energy=max(abs(np.array(energies)-energies[0])),
                            gauss=max(gauss),variation_substrate=float(np.max(abs(calc.y[:4,-1]-base[:4]))),fin=calc.y[:,-1].tolist()))
    identity=gauge_transform_check(x,*np.array(answers[-1]["fin"]).reshape(2,4))
    final=answers[-1]
    initial=answers[0]
    np.testing.assert_allclose(initial["fin"],final["fin"],rtol=1e-6,atol=1e-8)
    assert final['drift_energy']<1e-7 and final['variation_substrate']>.04 and identity['source_Gauss_initial']>.003
    return dict(N=n,évolution=answers,invariance=identity)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument("--sortie",type=Path,default=ROOT/'results')
    cfg=cli.parse_args()
    rows=[run(n) for n in (3,4)]
    write(cfg.output/"GAUSS_LIENS_INDUITS_R453.json",dict(revision=453,grids=rows,
          derivation=('The spatial connection is the polar overlap of the local subspaces. The temporal component '
                      'A0 has no derivative in L. In the local reference points, E and the material speed have the shape '
                      "w+D A0. Variation de L en A0 donne Dᵀ(w+D A0)=0. Après élimination de A0, la masse de la "
                      'reduced dynamic range is derived from the orthogonal projection on the DT core. The term platelet '
                      'and material energy are back-to-back with the potential of the same reduced action.'),
          scope=('Gun group bench with four mobile coordinates with three spatial profiles of the substrate and a material profile. '
                  'Selected diver and local saturation tested. The evolved links are all calculated from the subspaces. '
                  'at each step. The time gauge uses the SU(3) generators on each site. The system remains a '
                  'actual finite map. The microscopic selection of the dive and the spectrum of strong states are open.')))
    print(rows,flush=True)


if __name__=="__main__":
    main()
