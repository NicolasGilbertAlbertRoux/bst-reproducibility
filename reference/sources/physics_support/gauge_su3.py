'Preservation of Gauss in a non-switching SU(3) evolution in three dimensions.'
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
I=np.eye(3)


def adj(u):
    return np.swapaxes(u.conj(),-1,-2)


def trace(u):
    return np.trace(u,axis1=-2,axis2=-1)


def algebra(z):
    y=(z-adj(z))/(2j)
    return y-trace(y)[...,None,None]*I/3.


def exponential(x):
    l,V=np.linalg.eigh(x)
    return (V*np.exp(1j*l)[...,None,:])@adj(V)


def shift(x,j):
    return np.roll(x,-1,axis=j)


def loop(U,a,b):
    return U[a]@shift(U[b],a)@adj(shift(U[a],b))@adj(U[b])


def kick(U):
    force=[]
    for a in range(3):
        X=np.zeros_like(U[a])
        for b in range(3):
            if a==b:
                continue
            X+=loop(U,a,b)
            X+=U[a]@np.roll(adj(shift(U[b],a))@adj(U[a])@U[b],1,axis=b)
        force.append(-algebra(X))
    return np.array(force)


def gauss(U,E):
    return sum(E[a]-np.roll(adj(U[a])@E[a]@U[a],1,axis=a) for a in range(3))


def energy(U,E):
    return float((.5*trace(E@E).real).sum()+sum(np.sum(3.-trace(loop(U,a,b)).real)
                                               for a in range(3) for b in range(a)))


def evolve(U,E,dt,duration):
    G0=gauss(U,E)
    H0=energy(U,E)
    max_gauss,max_unit,max_energy=0.,0.,0.
    for _ in range(round(duration/dt)):
        half=E+dt/2*kick(U)
        U=exponential(dt*half)@U
        E=half+dt/2*kick(U)
        max_gauss=max(max_gauss,float(np.max(abs(gauss(U,E)-G0))))
        max_unit=max(max_unit,float(np.max(abs(adj(U)@U-I))),float(np.max(abs(np.linalg.det(U)-1.))))
        max_energy=max(max_energy,abs(energy(U,E)-H0)/max(H0,1.))
    assert max_gauss<1e-11 and max_unit<1e-11
    return U,E,dict(step=dt,defect_gauss=max_gauss,defect_SU3=max_unit,drift_energy=max_energy)


def random_algebra(rng,shape,strength):
    z=rng.normal(size=shape)+1j*rng.normal(size=shape)
    return strength*algebra(z)


def outer(a,b):
    return a[...,None]*b[...,None,:].conj()


def apply(a,x):
    return (a@x[...,None])[...,0]


def matter_force(U,psi):
    # Positive, gauge-invariant fundamental scalar Hamiltonian. Link
    # forces and matter forces are gradients of the same hopping energy.
    kappa,mu,quartic = .38,.65,.22
    length = np.sum(abs(psi)**2,axis=-1,keepdims=True)
    field = -(mu*mu+quartic*length)*psi
    link = np.empty_like(U)
    for a in range(3):
        forwarded=apply(U[a],shift(psi,a))
        field += kappa*(forwarded+np.roll(apply(adj(U[a]),psi),1,axis=a)-2.*psi)
        link[a] = -kappa*algebra(outer(forwarded,psi))
    return link,field


def matter_energy(U,psi,momentum):
    kappa,mu,quartic=.38,.65,.22
    length=np.sum(abs(psi)**2,axis=-1)
    hopping=sum(np.sum(abs(psi-apply(U[a],shift(psi,a)))**2) for a in range(3))
    return float(.5*np.sum(abs(momentum)**2+mu**2*abs(psi)**2)+
                 .25*quartic*np.sum(length**2)+.5*kappa*hopping)


def evolve_matter(U,E,psi,pi,dt,duration):
    H0=energy(U,E)+matter_energy(U,psi,pi)
    Gauss0=gauss(U,E)+algebra(outer(pi,psi))
    max_gauss,max_energy,max_source,max_unit=0.,0.,0.,0.
    for _ in range(round(duration/dt)):
        fU,fq=matter_force(U,psi)
        a=E+dt/2*(kick(U)+fU)
        b=pi+dt/2*fq
        U=exponential(dt*a)@U
        psi=psi+dt*b
        fU,fq=matter_force(U,psi)
        E=a+dt/2*(kick(U)+fU)
        pi=b+dt/2*fq
        source=algebra(outer(pi,psi))
        max_gauss=max(max_gauss,float(np.max(abs(gauss(U,E)+source-Gauss0))))
        max_source=max(max_source,float(np.max(abs(source))))
        max_energy=max(max_energy,abs((energy(U,E)+matter_energy(U,psi,pi))/H0-1.))
        max_unit=max(max_unit,float(np.max(abs(adj(U)@U-I))),float(np.max(abs(np.linalg.det(U)-1.))))
    assert max_gauss<1e-11 and max_unit<1e-11 and max_source>.001
    return (U,E,psi,pi),dict(step=dt,defect_Gauss_couple=max_gauss,
                              charge_matter_max=max_source,drift_energy=max_energy,defect_SU3=max_unit)


def main():
    cases=[]
    rng=np.random.default_rng(437)
    for N in (5,7):
        shape=(3,N,N,N,3,3)
        U=exponential(random_algebra(rng,shape,.17))
        E=np.zeros(shape,complex)
        direction=random_algebra(rng,shape,.25)
        K=kick(U)
        h=2e-5
        fd=(energy(exponential(h*direction)@U,E)-energy(exponential(-h*direction)@U,E))/(2*h)
        ex=-np.sum(trace(K@direction).real)
        np.testing.assert_allclose(fd,ex,rtol=5e-7,atol=2e-8)
        t=.6
        evol=[evolve(U,E,dt,t) for dt in (.02,.01,.005)]
        d=[x[2]['drift_energy'] for x in evol]
        assert d[0]>3.8*d[1] and d[1]>3.8*d[2]
        uf,ef,ctrl=evol[-1]
        ub,eb,_=evolve(uf,-ef,.005,t)
        reverse=max(float(np.max(abs(ub-U))),float(np.max(abs(eb+E))))
        assert reverse<3e-12
        g=exponential(random_algebra(rng,shape[1:],.5))
        gauged = lambda links:np.array([g@links[a]@adj(shift(g,a)) for a in range(3)])
        Uout,Eout,_=evolve(gauged(U),E,.005,t)
        cov=max(float(np.max(abs(Uout-gauged(uf)))),float(np.max(abs(Eout-g@ef@adj(g)))))
        assert cov<3e-12
        p1,p2=loop(uf,0,1),loop(uf,0,2)
        comm=float(np.max(abs(p1@p2-p2@p1)))
        assert comm>.01
        vacuum_U,vacuum_E,_=evolve(np.broadcast_to(I,shape).astype(complex).copy(),E,.02,.04)
        assert np.max(abs(vacuum_U-I))<1e-14 and np.max(abs(vacuum_E))<1e-14
        psi=.18*(rng.normal(size=shape[1:-1])+1j*rng.normal(size=shape[1:-1]))
        pi=np.zeros_like(psi)
        force_link,force_matter=matter_force(U,psi)
        fd_link=(matter_energy(exponential(h*direction)@U,psi,pi)-
                 matter_energy(exponential(-h*direction)@U,psi,pi))/(2*h)
        dpsi=.2*(rng.normal(size=psi.shape)+1j*rng.normal(size=psi.shape))
        fd_m=(matter_energy(U,psi+h*dpsi,pi)-matter_energy(U,psi-h*dpsi,pi))/(2*h)
        np.testing.assert_allclose([fd_link,fd_m],[-np.sum(trace(force_link@direction).real),
                 -np.vdot(force_matter,dpsi).real],rtol=1e-7,atol=1e-8)
        (_,_,_,_), m_coarse=evolve_matter(U,E,psi,pi,.01,t)
        (um,em,qm,pm), m_fine=evolve_matter(U,E,psi,pi,.005,t)
        assert m_coarse['drift_energy']>3.8*m_fine['drift_energy']
        (umg,emg,qmg,pmg),_=evolve_matter(gauged(U),E,apply(g,psi),pi,.005,t)
        cov_m=max(float(np.max(abs(x-y))) for x,y in
                  zip((umg,emg,qmg,pmg),(gauged(um),g@em@adj(g),apply(g,qm),apply(g,pm)),strict=True))
        assert cov_m<3e-12
        cases.append(dict(N=N,duration_field=t,energy_initial=energy(U,E),
            defect_force_diff_finite=float(abs(fd-ex)),commutator_plaquettes_max=comm,
            defect_covariance=cov,defect_return=reverse,step=[r[2] for r in evol],
            matter_fundamental=dict(defect_covariance=cov_m,step=[m_coarse,m_fine])))
        print(N,comm,ctrl,cov,m_fine,flush=True)
    out=dict(revision=437,scope='Conventional compact Hamiltonian SU(3) added on periodic network. '
             'Non-switching blisters, Gauss in a vacuum and with a conventional dynamic scalar triplet. '
             'The derivation of these fields in BST, their quantification and the observables of the spectrum remain open.',cases_field=cases,
             source='https://doi.org/10.1103/PhysRevD.11.395')
    (ROOT/'resultats/JAUGE_SU3_R437.json').write_text(json.dumps(out,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


if __name__=='__main__':
    main()
