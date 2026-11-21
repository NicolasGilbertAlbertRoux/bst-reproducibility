'Test the singlet reader on the dynamics already induced by the substrate.'
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import minimize

from gauss_induced import ROOT, atlas, fields, reduced
from gauge_su3 import adj, exponential, random_algebra, shift
from cosmology_R449 import write


def reader(u,matter):
    legs=(matter[...,0],(u[0]@shift(matter,0))[...,1],
          (u[1]@shift(matter,1))[...,2])
    return np.linalg.det(np.stack(legs,axis=-1)).mean()


def calculate(n):
    x=atlas(n)

    def potential(c):
        from gauge_su3 import loop, trace
        u,p,V=fields(x,c)
        f=sum(np.sum(3.-trace(loop(u,a,b)).real) for a,b in ((0,1),(0,2),(1,2)))
        d=sum(np.linalg.norm(p-u[j]@shift(p,j))**2 for j in range(3))
        return V+(f+.6*d)/n**3

    opt=minimize(potential,[1.,1.,1.,.12],method="BFGS",options={"gtol":5e-8})
    assert np.max(abs(opt.jac))<2e-7
    c=opt.x
    V,M,g,(u,p,*_)=reduced(x,c)
    np.testing.assert_allclose(V,potential(c),rtol=0.,atol=1e-12)
    gauge=exponential(random_algebra(np.random.default_rng(454+n),p.shape,.45))
    u2=[gauge@link@adj(shift(gauge,k)) for k,link in enumerate(u)]
    gauge_error=float(abs(reader(u2,gauge@p)-reader(u,p)))
    assert gauge_error<1e-14
    identity=np.eye(4)
    last=None
    for h in (4e-4,2e-4,1e-4):
        hess=np.empty((4,4))
        for i in range(4):
            for j in range(i,4):
                di=h*identity[i]
                dj=h*identity[j]
                value=(potential(c+di+dj)-potential(c+di-dj)-potential(c-di+dj)+potential(c-di-dj))/(4*h*h)
                hess[i,j]=hess[j,i]=value
        ev,modes=eigh(hess,M)
        assert ev.min()>0.
        if last is not None:
            np.testing.assert_allclose(last,ev,rtol=1.e-6)
        last=ev

    selected=[]
    eps=4e-3
    for w,v in zip(np.sqrt(ev),modes.T,strict=True):
        def response(v,delta):
            U,psi,_=fields(x,c+v*delta)
            return reader(U,psi)
        plus,minus=(response(v,sign*eps) for sign in (1.,-1.))
        derivative=(plus-minus)/(2*eps)
        small=(response(v,eps/2)-response(v,-eps/2))/eps
        np.testing.assert_allclose(derivative,small,atol=8e-8,rtol=.002)
        selected.append(dict(pulsation_native=float(w),coupling_readout=abs(derivative)))
    # Evolve a true collective trajectory and compare its singlet signal
    # with the linear trace using all normal-mode participations.
    start=np.linspace(.003,.006,4)
    t=np.linspace(0.,.48,17)
    def rhs(_t,y):
        at,vel=y.reshape(2,4)
        _,metric,_,_=reduced(x,at)
        dM,dV=[],[]
        for direction in identity:
            first,last=[reduced(x,at+sign*3e-5*direction)[:2] for sign in (1.,-1.)]
            dV.append((first[0]-last[0])/6e-5)
            dM.append((first[1]-last[1])/6e-5)
        dM=np.array(dM)
        cforce=.5*np.einsum("ijk,j,k->i",dM,vel,vel)-np.einsum("jik,j,k->i",dM,vel,vel)
        return np.r_[vel,np.linalg.solve(metric,cforce-np.asarray(dV))]
    sol=solve_ivp(rhs,(t[0],t[-1]),np.r_[c+start,np.zeros(4)],t_eval=t,
                  method="DOP853",atol=2e-10,rtol=2e-8,max_step=.12)
    assert sol.success
    exact=np.array([reader(*fields(x,q)[:2]) for q in sol.y[:4,:].T])
    forecast=c[:,None]+modes@((modes.T@M@start)[:,None]*np.cos(np.sqrt(ev)[:,None]*t[None,:]))
    trial=np.array([reader(*fields(x,q)[:2]) for q in forecast.T])
    err=float(np.max(abs(exact-trial)))
    assert err<.1*max(abs(exact-exact[0]))
    return dict(N=n,coordinates_rest=c.tolist(),mode=selected,defect_gauge=gauge_error,
                gauss=g,error_temporal_readout=err,amplitude_temporal_readout=float(max(abs(exact-exact[0]))))


if __name__=="__main__":
    cases=[calculate(n) for n in (3,4)]
    write(ROOT/'results'/"SINGULET_COLLECTIF_R454.json",dict(
        revision=454,cases_field=cases,
        scope='The determinant of the three triplets transported is read at the fixed point and in a trajectory '
               'of the reduced action R453. The pulses are those of the four common coordinates, '
               'with the Gauss metric. It is a classical reader response spectrum. '
               'The establishment of separate strong states and a physical mass scale remains to be done.'))
    print(cases,flush=True)
