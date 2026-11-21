'H528: conditional bistable trace coupled to the archived R466 spectral bath.\n\nThis is a NEW constitutive candidate, not a fitted brain or a new canonical\nBST law. Native bath frequencies/weights are copied unchanged. All other\ncoefficients are declared dimensionless assumptions. No human data are used.'
from pathlib import Path
import csv, hashlib, itertools, json, platform
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
DATA = np.load(ROOT / 'entrees/MESURES_SPECTRALES_NH3_R466.npz')
K0 = 3.337**2  # explicit diagnostic reference, NOT a biological calibration


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def make_case(tag='m1', eta=.5, lam=1., mass=1., impulse=.1, sign=1,
              latent_momentum=0., gamma=0., broken=False):
    return dict(tag=tag, eta=eta, lam=lam, mass=mass, impulse=impulse,
                sign=sign, latent_momentum=latent_momentum, gamma=gamma, broken=broken)


def coefficients(cases):
    w = np.array([DATA[c['tag']+'_kernel_weight'] for c in cases])
    omega = np.array([DATA[c['tag']+'_kernel_rate'] for c in cases])
    g = omega*np.sqrt(w)
    return w, omega, g


def energy(Q, P, cases, omega, g):
    x, q = Q[:, 0], Q[:, 1]
    mass = np.array([c['mass'] for c in cases])
    lam = np.array([c['lam'] for c in cases])
    eta = np.array([c['eta'] for c in cases])
    s = Q[:, 2:] - g/omega**2*x[:, None]
    return (.5*P[:, 0]**2 + .5*P[:, 1]**2/mass + .5*np.sum(P[:, 2:]**2, axis=1)
            + lam/4*(q*q-1)**2 + .5*K0*np.exp(eta*q)*x*x
            + .5*np.sum(omega**2*s*s, axis=1))


def gradient(Q, cases, omega, g):
    x, q = Q[:, 0], Q[:, 1]
    lam = np.array([c['lam'] for c in cases])
    eta = np.array([c['eta'] for c in cases])
    broken = np.array([c['broken'] for c in cases])
    s = Q[:, 2:] - g/omega**2*x[:, None]
    k = K0*np.exp(eta*q)
    grad = np.empty_like(Q)
    grad[:, 0] = k*x-np.sum(g*s, axis=1)
    grad[:, 1] = lam*q*(q*q-1)+(~broken)*.5*eta*k*x*x
    grad[:, 2:] = omega**2*s
    return grad


def integrate(cases, dt=.002, end=8., sample=.01):
    w, omega, g = coefficients(cases)
    n = len(cases)
    Q = np.zeros((n, 794)); P = np.zeros_like(Q)
    Q[:, 1] = [c['sign'] for c in cases]
    P[:, 0] = [c['impulse'] for c in cases]
    P[:, 1] = [c['latent_momentum'] for c in cases]
    invmass = np.ones_like(Q); invmass[:, 1] = [1/c['mass'] for c in cases]
    gamma = np.array([c['gamma'] for c in cases])
    half_decay = np.exp(-gamma*dt/2)
    E0 = energy(Q, P, cases, omega, g)
    loss = np.zeros(n)
    records = []; stride = round(sample/dt); steps = round(end/dt)
    maximum = np.zeros(n); min_signed_q = np.ones(n)
    signs = np.array([c['sign'] for c in cases])
    def record(t):
        E = energy(Q, P, cases, omega, g)
        records.append(np.stack((np.full(n,t),Q[:,0],Q[:,1],P[:,0],P[:,1],E,loss),axis=1))
    record(0.)
    for step in range(1, steps+1):
        # Exact damping subflow with energy loss booked to an explicit reservoir.
        before = .5*P[:,0]**2; P[:,0] *= half_decay; loss += before-.5*P[:,0]**2
        P -= .5*dt*gradient(Q,cases,omega,g)
        Q += dt*P*invmass
        P -= .5*dt*gradient(Q,cases,omega,g)
        before = .5*P[:,0]**2; P[:,0] *= half_decay; loss += before-.5*P[:,0]**2
        min_signed_q = np.minimum(min_signed_q, signs*Q[:,1])
        # Audit every step, not just plotted samples.
        E = energy(Q,P,cases,omega,g)
        maximum = np.maximum(maximum, np.abs(E+loss-E0)/np.maximum(E0,1e-30))
        if step % stride == 0: record(step*dt)
    return np.array(records), maximum, min_signed_q


def independent(c, end=8.):
    cases=[c];w,omega,g=coefficients(cases);n=794
    start=np.zeros(2*n+1);start[1]=c['sign'];start[n]=c['impulse'];start[n+1]=c['latent_momentum']
    def f(t, state):
        Q=state[:n][None,:];P=state[n:2*n][None,:]
        dQ=P.copy();dQ[:,1]/=c['mass']
        dP=-gradient(Q,cases,omega,g);dP[:,0]-=c['gamma']*P[:,0]
        return np.r_[dQ[0],dP[0],c['gamma']*P[0,0]**2]
    t=np.arange(round(end/.01)+1)*.01
    sol=solve_ivp(f,(0,end),start,method='DOP853',rtol=2e-10,atol=2e-12,t_eval=t)
    assert sol.success
    E=energy(sol.y[:n].T,sol.y[n:2*n].T,[c]*len(t),np.repeat(omega,len(t),axis=0),np.repeat(g,len(t),axis=0))
    initial=E[0];drift=np.max(abs(E+sol.y[-1]-initial))/initial
    return t,sol.y,drift,sol.nfev


def finite_difference_audit():
    rng=np.random.default_rng(528);c=make_case();cases=[c]
    _,omega,g=coefficients(cases);max_grad=0.;max_mirror=0.;missing=0.
    for _ in range(64):
        Q=rng.normal(0,.02,size=(1,794));P=rng.normal(0,.01,size=(1,794));Q[0,1]=rng.uniform(-1.2,1.2)
        grad=gradient(Q,cases,omega,g)[0]
        for j in [0,1,int(rng.integers(2,794))]:
            h=1e-6;plus=Q.copy();minus=Q.copy();plus[0,j]+=h;minus[0,j]-=h
            fd=(energy(plus,P,cases,omega,g)-energy(minus,P,cases,omega,g))[0]/(2*h)
            max_grad=max(max_grad,abs(fd-grad[j])/max(1,abs(grad[j])))
        h=1e-5
        qp=Q.copy();qm=Q.copy();qp[0,1]+=h;qm[0,1]-=h
        d_x_d_q=(gradient(qp,cases,omega,g)[0,0]-gradient(qm,cases,omega,g)[0,0])/(2*h)
        qp=Q.copy();qm=Q.copy();qp[0,0]+=h;qm[0,0]-=h
        d_q_d_x=(gradient(qp,cases,omega,g)[0,1]-gradient(qm,cases,omega,g)[0,1])/(2*h)
        max_mirror=max(max_mirror,abs(d_x_d_q-d_q_d_x))
        missing=max(missing,abs(d_x_d_q))
    assert max_grad<2e-7 and max_mirror<1e-7 and missing>.01
    return dict(random_states=64,gradient_relative_or_absolute_error_max=max_grad,
                mixed_derivative_reciprocity_error_max=max_mirror,
                missing_mirror_detected_max=missing)


def run_memory():
    cases=[make_case(tag,eta,lam,mass,j,sign) for tag,eta,lam,mass,j,sign in itertools.product(
        ['m1','m2'],[.2,.5,.8],[.5,2.],[.5,2.],[.02,.1,.2],[-1,1])]
    rec,drift,mins=integrate(cases)
    # At t=0.1 the leading odd response identifies the content. Threshold is
    # fixed analytically, not tuned separately for each content or trajectory.
    h=.1;i=round(h/.01);correct=0;read_margins=[];pairs=[]
    for k,c in enumerate(cases):
        w=DATA[c['tag']+'_kernel_weight'].sum()
        curvature=6*(1-rec[i,k,1]/(c['impulse']*h))/h**2
        midpoint=K0*np.cosh(c['eta'])+w
        decoded=1 if curvature>midpoint else -1
        correct+=int(decoded==c['sign']);read_margins.append(c['sign']*(curvature-midpoint))
        c.update(energy_error_max=float(drift[k]),minimum_signed_trace=float(mins[k]),
                 energy_initial=.5*c['impulse']**2,barrier=c['lam']/4,
                 decoded_sign=decoded,read_margin=float(read_margins[-1]))
    for k in range(0,len(cases),2):
        j=cases[k]['impulse']
        pairs.append(float(np.max(abs(rec[:,k+1,1]-rec[:,k,1]))/j))
    assert np.max(drift)<5e-5 and np.min(mins)>0 and correct==len(cases)
    # Fixed-linear control: eta=0 removes content dependence even with the bath.
    controls=[make_case(eta=0.,sign=s) for s in [-1,1]]
    ctrl,_,_=integrate(controls)
    null=float(np.max(abs(ctrl[:,0,1]-ctrl[:,1,1])));assert null==0
    # Same equations, no cue: both preparations are exactly stationary.
    quiet,_,_=integrate([make_case(impulse=0.,sign=s) for s in [-1,1]],end=2.)
    assert np.max(abs(quiet[:,:,1]))==0 and np.max(abs(quiet[:,:,2]-np.array([-1,1])))==0
    refs=[make_case(tag=tag,eta=.8,lam=.5,mass=.5,impulse=.2,sign=sign)
          for tag,sign in itertools.product(['m1','m2'],[-1,1])]
    fine,fd,_=integrate(refs,dt=.0005)
    checks=[]
    for k,c in enumerate(refs):
        t,y,e,n=independent(c)
        err=float(np.max(abs(fine[:,k,1]-y[0]))/max(np.max(abs(y[0])),1e-15))
        qerr=float(np.max(abs(fine[:,k,2]-y[1])))
        checks.append(dict(case=c,relative_signal_error=err,trace_abs_error=qerr,
                           dop853_balance_error=float(e),verlet_balance_error=float(fd[k]),nfev=n))
    assert max(c['relative_signal_error'] for c in checks)<3e-5
    # Deliberately remove return force while latent coordinate moves.
    negs=[make_case(eta=.8,impulse=.2,latent_momentum=.15,broken=b) for b in [False,True]]
    neg,ne,nn=integrate(negs,dt=.0005)
    assert ne[0]<1e-5 and ne[1]>1e-3
    dampcases=[make_case(tag=tag,sign=s,gamma=.15,impulse=.2)
               for tag,s in itertools.product(['m1','m2'],[-1,1])]
    damp,of,dm=integrate(dampcases,dt=.001)
    assert max(of)<2e-5 and min(dm)>0 and np.min(damp[-1,:,6])>0
    # A force budget cannot substitute for an information budget: eta=0
    # preserves the latent sign but makes this specified readout blind to it.
    result=dict(revision=528,hypothesis='H528_NEW_CONDITIONAL_CANDIDATE',
        biological_calibration=False,canonical_BST_coefficients_modified=False,
        native_bath_modes_per_case=792,trajectories=len(cases),paired_content_comparisons=len(pairs),
        same_coefficients_for_both_contents=True,perfect_resting_readout_indistinguishability=True,
        maximum_energy_relative_error=float(max(drift)),minimum_signed_trace=float(min(mins)),
        correct_analytic_decodings=correct,total_decodings=len(cases),minimum_read_margin=float(min(read_margins)),
        minimum_pair_difference_per_unit_impulse=min(pairs),linear_control_interaction=null,
        independent_integrator_checks=checks,finite_difference_audit=finite_difference_audit(),
        negative_control=dict(return_present_balance_error=float(ne[0]),return_missing_balance_error=float(ne[1])),
        dissipative_checks=dict(cases=len(dampcases),balance_error_max=float(max(of)),
          minimum_heat_collected=float(min(damp[-1,:,6])),minimum_signed_trace=float(min(dm))),
        analytic_interval_certificate=dict(lam=[.5,2.],eta=[.2,.8],mass=[.5,2.],impulse=[.02,.2],
          maximum_initial_energy=.02,minimum_crossing_barrier=.125,strict_energy_margin=.105,
          sign_invariant_between_writes=True,requires_nonpositive_unbooked_energy_input=True,
          noise_or_thermal_perturbations_excluded=True,
          amplitude_pointwise_bound='sqrt(1-sqrt(4*E/lambda)) <= abs(q) <= sqrt(1+sqrt(4*E/lambda))'),
        limits=['New bistable variable q and its coupling are assumed, not derived from primordial BST waves.',
                'The R466 bath realization is exact for its scalar kernel, not the full NH3 molecular state.',
                'No neural/non-neural carrier, subject, raphe, imagination or free agency identified.',
                'No human observations. Numerical success is conditional existence, not empirical confirmation.',
                'Preparation/write/erase hardware and their energy costs are not simulated.',
                'The analytic sign certificate covers the continuous interval; the numerical decoder is tested only at the reported grid points.'])
    dump('PERSISTANCE_ET_CANAL_R528.json',result)
    with (OUT/'BALAYAGE_COGNITIF_R528.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    np.savez_compressed(OUT/'TRAJECTOIRES_COGNITIVES_R528.npz',records=rec,
                        negative_control=neg,dissipative=damp,linear_control=ctrl,
                        reference_fine=fine,columns=np.array(['t','x','q','p_x','p_q','H','Q_loss']))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    selected=[next(k for k,c in enumerate(cases) if c['tag']=='m1' and c['eta']==.5 and
                   c['lam']==.5 and c['mass']==.5 and c['impulse']==.2 and c['sign']==s) for s in [-1,1]]
    for k,col in zip(selected,['#3567a8','#c56b28']):
        c=cases[k];axes[0,0].plot(rec[:,k,0],rec[:,k,2],color=col,label=f"Trace initiale {c['sign']:+d}")
        axes[0,1].plot(rec[:,k,0],rec[:,k,1]/c['impulse'],color=col,label=f"Trace initiale {c['sign']:+d}")
    axes[0,0].set(title='Le contenu persiste pendant la lecture',xlabel='Reduced time',ylabel='Latent coordinate q')
    axes[0,1].set(title='Same impulse, separate responses',xlabel='Reduced time',ylabel='Response x/pulse')
    axes[1,0].plot(rec[:,0,0],(rec[:,selected[1],1]-rec[:,selected[0],1])/.2,color='#734e98',label='Trac-dependent plasting')
    axes[1,0].plot(ctrl[:,0,0],(ctrl[:,1,1]-ctrl[:,0,1])/.1,color='#4c8573',ls='--',label='Couplage fixe : contraste nul')
    axes[1,0].set(title='Contraste contenu × impulsion',xlabel='Reduced time',ylabel='Difference/pulse')
    for k,lab,col in [(0,'Return','#4c8573'),(1,'Return deleted','#b74d45')]:
        axes[1,1].plot(neg[:,k,0],(neg[:,k,5]-neg[0,k,5])/neg[0,k,5],label=lab,color=col)
    axes[1,1].set(title='Counter-test mirror',xlabel='Reduced time',ylabel='Relative change in energy')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    fig.suptitle('BST R528 — candidat de trace persistante et de lecture réciproque\nHypothèse constitutive nouvelle ; unités réduites ; aucune donnée cérébrale',fontsize=13)
    fig.savefig(ROOT/'PERSISTANCE_ET_RELECTURE_R528.png',dpi=165);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['trajectories','maximum_energy_relative_error','correct_analytic_decodings','negative_control']},indent=2),flush=True)
    return result


def kernel(tag,t,fast=True):
    w=DATA[tag+'_kernel_weight'];om=DATA[tag+'_kernel_rate'];total=w.sum()
    mask=w>1e-12*total if fast else np.ones(len(w),dtype=bool)
    return np.sum(np.cos(np.asarray(t)[:,None]*om[mask])*w[mask],axis=1)/total


def registration():
    train=np.linspace(.25,2.,351);hold=np.linspace(2.2,8.,1161);late=np.linspace(200,204,801)
    target=kernel('m1',train);rows=[]
    for label,bounds in [('translation',[(-.2,.2),(-.25,.25)]),
                         ('translation_gain',[(-.2,.2),(-.25,.25),(.5,2.)]),
                         ('translation_gain_dilatation',[(-.2,.2),(-.25,.25),(.5,2.),(.5,2.)])]:
        def unpack(p):return p[0],p[1],(p[2] if len(p)>2 else 1.),(p[3] if len(p)>3 else 1.)
        def pred(p,t):
            delta,offset,gain,scale=unpack(p)
            return offset+gain*kernel('m2',(t-delta)/scale)
        def cost(p):return float(np.max(abs(pred(p,train)-target)))
        fit=differential_evolution(cost,bounds,seed=528,tol=1e-8,popsize=12,maxiter=350,polish=False)
        refined=minimize(cost,fit.x,method='Nelder-Mead',bounds=bounds,options={'maxiter':3000,'xatol':1e-11,'fatol':1e-11})
        p=refined.x if refined.fun<=fit.fun else fit.x
        delta,offset,gain,scale=unpack(p)
        rows.append(dict(family=label,shift=float(delta),offset=float(offset),gain=float(gain),time_scale=float(scale),
             training_max_abs_error=cost(p),holdout_max_abs_error=float(max(abs(pred(p,hold)-kernel('m1',hold)))),
             late_max_abs_error=float(max(abs(pred(p,late)-kernel('m1',late)))),
             optimizer_success=bool(fit.success),global_optimum_certified=False))
    # Evaluate winning transform against all 792 modes, not just compressed sum.
    best=rows[-1];compression={}
    for tag in ['m1','m2']:
        w=DATA[tag+'_kernel_weight'];compression[tag]=float(w[w<=1e-12*w.sum()].sum()/w.sum())
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,t,title in zip(axes,[train,hold,late],['Calibration: 0.25–2','Reservation: 2.2–8','Prolongement : 200–204']):
        target_all=kernel('m1',t,False)
        raw=kernel('m2',t,False)
        aligned=best['offset']+best['gain']*kernel('m2',(t-best['shift'])/best['time_scale'],False)
        ax.plot(t,target_all,color='#3567a8',label='Cas inertie 1')
        ax.plot(t,raw,color='#999999',alpha=.65,lw=.7,label='Cas inertie 2 brut')
        ax.plot(t,aligned,color='#c56b28',label='Case 2 reset, fixed parameters')
        ax.set(title=title,xlabel='Temps natif',ylabel='K(t) / K(0)');ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Méthode de la translation graphique — deux noyaux R466\nRecalage minimax exploratoire ; les fenêtres réservées ne servent pas au réglage',fontsize=13)
    fig.savefig(ROOT/'TRANSLATIONS_MEMOIRE_R528.png',dpi=165);plt.close(fig)
    result=dict(revision=528,rows=rows,calibration_time=[.25,2.],holdout_time=[2.2,8.],late_time=[200,204],
      normalization='Each kernel divided by its own archived K(0); errors are max absolute fractions of that reference, not pointwise percentages.',
      full_spectrum_modes=792,uniform_truncation_error_bound= compression,
      physical_time_calibration=False,biological_translation=False,physical_laws_changed=False,
      global_minimax_certified=False,fit_type='Seeded numerical search; a found residual is an upper bound on the optimum, not a proof no better transform exists.')
    dump('TRANSLATIONS_MEMOIRE_R528.json',result);print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    plt.rcParams.update({'font.size':10})
    memory=run_memory();curves=registration()
    dump('VERIFICATION_NUMERIQUE_R528.json',dict(all_declared_tests_passed=True,revision=528,
         inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'inputs').iterdir() if p.is_file()},
         environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
         human_data_used=False,empirical_validation=False))
