"Conditional witnesses for the user's latent-initiation proposal.\n\nDimensionless ports, prescribed gates, and abstract phase memory. These are\ncounterexample/coherence tests, not a brain, raphe, subject or consciousness\nsimulation. A clock-based latent controller can reproduce the supplied gates."
from pathlib import Path
import json,itertools
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True)
SCHEDULE=[(0.,2.,0),(2.,5.,1),(5.,7.,0),(7.,10.,2),(10.,12.,0)]
def matrix(route,broken=False):
    J=np.zeros((6,6))
    for j,w in enumerate([1.,1.2,.85]):J[2*j,2*j+1]=w;J[2*j+1,2*j]=-w
    if route:
        k=.75 if route==1 else .48;target=2*route
        J[target,1]=k
        if not broken:J[1,target]=-k
    return J
def propagate(broken=False):
    x=np.array([1.,0.,0.,0.,0.,0.]);times=[];states=[];routes=[];powers=[]
    for a,b,route in SCHEDULE:
        ts=np.linspace(a,b,501);J=matrix(route,broken);step=expm(J*(ts[1]-ts[0]))
        for i,t in enumerate(ts):
            if i:x=step@x
            # Keep each boundary once; state itself is continuous.
            if times and t==times[-1]:continue
            coupling=J-matrix(0);dx=coupling@x
            times.append(t);states.append(x.copy());routes.append(route)
            powers.append([float(x[2*j:2*j+2]@dx[2*j:2*j+2]) for j in range(3)])
    x=np.array(states).T;t=np.array(times);p=np.array(powers).T
    E=np.array([.5*np.sum(x[2*j:2*j+2]**2,axis=0) for j in range(3)])
    return t,x,E,np.array(routes),p

def memory():
    # Between writes, known rotations transport an encoded 3-bit memory.
    # All memories have the same norm/energy. The norm alone cannot recover it.
    messages=list(itertools.product([0,1],repeat=3));omega=np.array([.31,.57,.83])
    good=0;erased=0;trials=0;transport_errors=[];max_norm_error=0.;coarse_reader_error=0.
    for bits in messages:
        z0=np.exp(1j*np.pi*np.array(bits))
        for t in np.linspace(0,50,41):
            z=z0*np.exp(1j*omega*t);recovered=z*np.exp(-1j*omega*t)
            decoded=(recovered.real<0).astype(int)
            # Ideal phase reset is a non-invertible intervention; its physical
            # erasure device and thermodynamic cost are outside this witness.
            z_erased=np.exp(1j*omega*t)
            decoded_erased=((z_erased*np.exp(-1j*omega*t)).real<0).astype(int)
            good+=int(np.array_equal(decoded,bits));erased+=int(np.array_equal(decoded_erased,bits));trials+=1
            coarse_reader_error=max(coarse_reader_error,float(max(abs(abs(z)-abs(z_erased)))))
            transport_errors.append(float(max(abs(recovered-z0))))
            max_norm_error=max(max_norm_error,float(max(abs(abs(z)-1))))
    # Exact same coarse reader after either phase preservation or phase reset.
    # A later phase-sensitive reader separates the two. Its clock is required.
    return dict(bits=3,messages=8,times=41,trials=trials,correct_recall_preserving_model=good,
                correct_recall_phase_erased_model=erased,coarse_norm_reader_identical=bool(coarse_reader_error<1e-12),coarse_norm_reader_max_error=coarse_reader_error,
                transported_memory_max_error=max(transport_errors),norm_max_error=max_norm_error,
                recovery_requires_known_phase_reference=True,
                invariant_applies_between_writes=True,new_learning_changes_memory=True,
                identity_of_a_subject_not_derived=True,physical_erasure_cost_not_modeled=True,interpretation='Conservation of a scalar norm does not prove preservation of memory content; a content-bearing transport and a readout are separate requirements.')

def main():
    rng=np.random.default_rng(526);power_errors=[];exchange_errors=[]
    for _ in range(1024):
        route=int(rng.integers(0,3));x=rng.normal(size=6);J=matrix(route)
        power_errors.append(abs(float(x@J@x))/max(float(x@x),1.))
        dx=(J-matrix(0))@x
        exchange_errors.append(abs(sum(x[2*j:2*j+2]@dx[2*j:2*j+2] for j in range(3)))/max(float(x@x),1.))
    t,x,E,a,p=propagate();tb,xb,Eb,ab,pb=propagate(True)
    mem=memory();drift=float(max(abs(E.sum(axis=0)-.5))/.5);bad=float(max(abs(Eb.sum(axis=0)-.5))/.5)
    # The gate trajectory can be generated internally from a stored schedule and
    # an evolving clock. Equality shows nonidentifiability, not that humans are clocks.
    algorithmic=np.array([next(route for lo,hi,route in SCHEDULE if lo-1e-12<=tt<=hi+1e-12) for tt in t])
    # Boundary values are observational conventions; compare open segments only.
    boundaries=np.array([q[1] for q in SCHEDULE[:-1]])
    interior=np.min(abs(t[:,None]-boundaries[None,:]),axis=1)>1e-10
    identical=bool(np.array_equal(algorithmic[interior],a[interior]))
    assert max(power_errors)<1e-12 and max(exchange_errors)<1e-12 and drift<1e-9 and bad>.1
    assert mem['correct_recall_preserving_model']==328 and mem['correct_recall_phase_erased_model']==41 and identical
    out=dict(revision=526,family='H526: ideal latent-controlled conservative routing',
             assumptions=['Gates and routes are prescribed, not derived from a free agent.','Hamiltonian H=||x||^2/2 is independent of gate.','Switching an ideal skew interconnection has no stored-energy jump in this abstract model.','Real controller, neural dynamics, thermal bath and switching costs are not included.'],
             state_tests=1024,energy_power_identity_max=max(power_errors),exchange_identity_max=max(exchange_errors),
             trajectory_relative_energy_drift=drift,missing_mirror_relative_energy_drift=bad,
             deterministic_latent_controller_same_gate_trace=identical,memory=mem,
             pass_all_declared_tests=True,physical_raphe_model=False,empirical_cognition_validation=False,
             metaphysical_agency_identified=False,personal_identity_invariant_derived=False)
    (O/'COGNITION_FLUX_ET_MEMOIRE_R526.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    np.savez_compressed(O/'COGNITION_FLUX_R526.npz',time=t,state=x,energy=E,route=a,exchange_power=p,negative_control_energy=Eb)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,ax=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    ax[0,0].step(t,a,where='post',color='#4b566c');ax[0,0].set(yticks=[0,1,2],yticklabels=['Non-engagement','Voie 1','Voie 2'],xlabel='Reduced time',title='Porte et routage prescrits')
    for j,(name,col) in enumerate([('Secteur latent','#6e5ba5'),('Voie observable 1','#285ddd'),('Voie observable 2','#199a90')]):
        ax[0,1].plot(t,E[j],color=col,label=name)
        ax[1,0].plot(t,p[j],color=col,label=name)
    ax[0,1].set(xlabel='Reduced time',ylabel='Reduced energy',title='A reserve is redistributed between sectors');ax[0,1].legend(fontsize=8)
    ax[1,0].set(xlabel='Reduced time',ylabel='Reduced power',title='Flows opposite to ports, including return')
    ax[1,1].plot(t,E.sum(axis=0)/.5,color='#199a90',label='Returns')
    ax[1,1].plot(tb,Eb.sum(axis=0)/.5,color='#ba643e',label='A sense of coupling removed')
    ax[1,1].set(xlabel='Reduced time',ylabel='Total energy / initial energy',title='Le bilan contrôle le raccord');ax[1,1].legend(fontsize=9)
    fig.suptitle('BST R526 - témoin mathématique de commande latente\nCommandes prescrites ; aucun cerveau ni vécu simulé.',fontsize=13)
    fig.savefig(R/'FLUX_COGNITIFS_CONDITIONNELS_R526.png',dpi=160);plt.close(fig)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
