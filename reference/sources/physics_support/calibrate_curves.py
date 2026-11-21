'Calibrate common nonlinear curves with two mass constraints and reserve rotation.'

from pathlib import Path

from sweep_sf4 import read
from collection_curves import CASE,EDGES,CHI_EDGES,OUTPUT,matrices
from fields_and_references import dump
from trials_nodal import digest
from clock_inertia import C_SI,H_SI,MASSES_U,REF,U_SI
from inertia_and_entrainment import M_RATIO,SF6_REF,Rotor,search,spin_free_mass
from free_rotations import ROOT,reduced
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import qmc


MAIN4 = 'SF4_R349_C2'
MAIN6 = 'SF6_R336_C2'
M4 = (MASSES_U[0]+4*MASSES_U[1])*U_SI
M6 = (MASSES_U[0]+6*MASSES_U[1])*U_SI
M64 = M6/M4
GIVEN_R6 = 1.5607
X = np.sqrt(EDGES[:-1]*np.minimum(EDGES[1:],1e5))
X[0] = 0.
Y = (CHI_EDGES[1:]+CHI_EDGES[:-1])/2
P_VALUES = (1,2,3,4,6,8)
Q_VALUES = (0,1,2,4,-1,-2,-4)
LOG_KAPPA = np.linspace(np.log(1e-5),np.log(200.),179)
AMP_MAX = 1e4


class Bank:
    def __init__(self,tag):
        path = OUTPUT/f'BANQUE_{tag}_R352.npz'
        self.meta = read(path.with_suffix('.json'))
        with np.load(path) as z:
            if not np.array_equal(z['edges'],EDGES) or not np.array_equal(z['chi_edges'],CHI_EDGES):
                raise ValueError('Binned integration grids differ')
            self.bins = z['bins']
            self.const = z['constant_matrices']
            self.linear = z['linear_x_matrices']
        self.tag = tag
        self.path = path
        self.geometry = self.meta['geometry']
        self.cache = {}

    def collapse(self,q):
        if q not in self.cache:
            v = Y**q if q>=0 else (1.-Y)**(-q)
            self.cache[q] = np.einsum('axyj,y->axj',self.bins,v)
        return self.cache[q]

    def extra(self,p,q,k):
        return matrices(np.einsum('axj,x->aj',self.collapse(q),hill(X,p,k)))

    def gram(self,curve):
        return self.const+curve['a']*self.extra(curve['p'],curve['q'],curve['kappa'])


def hill(x,p,k):
    r = (x/k)**p
    return r/(1.+r)


def isotropic(blocks):
    'Return the unique scalar if the local translation tensor is isotropic.'
    w = blocks[:,:3,:3]
    c = blocks[:,:3,3:]
    k = blocks[:,3:,3:]
    return np.trace(k-c.transpose(0,2,1)@np.linalg.solve(w,c),axis1=-2,axis2=-1)/3.


def ratios(G4,G6):
    a,b = isotropic(G4),isotropic(G6)
    return a[0]/a[1],b.sum()/a.sum()


def mass_residuals(G4,G6):
    a,b = ratios(G4,G6)
    return np.array([a/M_RATIO-1.,b/M64-1.])


def isotope_amplitude(G,extra):
    def f(a):
        m = isotropic(G[:2]+a*extra[:2])
        return m[0]/m[1]-M_RATIO
    if f(0.)>=0.:
        raise ArithmeticError('Expected the unweighted sulfur mass to be too low')
    if f(AMP_MAX)<=0.:
        return None
    return float(brentq(f,0.,AMP_MAX,xtol=1e-10))


def roots_for(p,q,train4,train6):
    def solve(log_k):
        k = float(np.exp(log_k))
        E = train4.extra(p,q,k)
        a = isotope_amplitude(train4.const,E)
        if a is None:
            return None
        err = mass_residuals(train4.const+a*E,train6.const+a*train6.extra(p,q,k))[1]
        return float(err),{'p':p,'q':q,'kappa':k,'a':a}
    rows = [(k,r) for k in LOG_KAPPA if (r:=solve(k)) is not None]
    found = []
    for (a,r),(b,s) in zip(rows[:-1],rows[1:],strict=True):
        if b-a>1.01*(LOG_KAPPA[1]-LOG_KAPPA[0]) or r[0]*s[0]>0.:
            continue
        root = brentq(lambda k:solve(k)[0],a,b,xtol=2e-11)
        found.append(solve(root)[1])
    return found,[{'kappa':r[1]['kappa'],'a':r[1]['a'],'relative_mass6_residual':r[0]} for _,r in rows]


def threshold_envelope(bank):
    'Enclose every nondecreasing weight in x on this quadrature cloud.'
    xbins = matrices(bank.bins.sum(axis=2))
    tails = np.cumsum(xbins[:,::-1],axis=1)[:,::-1]
    lower = np.concatenate((tails[:,1:],np.zeros_like(tails[:,:1])),axis=1)
    # The optimum over frame spin for a sum of positive quadratic forms
    # is bounded below by the sum of the independent optima.
    conservative = []
    for t in range(tails.shape[1]):
        U = tails[:,t]
        L = lower[:,t]
        if np.trace(U[0,3:,3:])<=0.:
            continue
        low = []
        for mat in L[1:]:
            w,c = mat[:3,:3],mat[:3,3:]
            smallest = np.linalg.eigvalsh(w)[0]
            if smallest<=1e-12:
                low.append(0.)
                continue
            y = np.linalg.solve(w,c)
            residual = w@y-c
            value = np.trace(mat[3:,3:]-2*c.T@y+y.T@w@y)
            error = np.sum(residual**2)/smallest + 1e-11*max(1.,np.max(abs(mat)))
            low.append(max(0.,float((value-error)/3.)))
        hi = max(0.,float(np.trace(U[0,3:,3:])/3.+1e-12))
        F = float(np.sum(low)/(len(L)-1))
        if F<=0.:
            raise ArithmeticError('No positive lower fluorine trace for a contributing sulfur tail')
        conservative.append((hi/F,t))
    upper,t = max(conservative)
    baseline = np.trace(bank.const[0,3:,3:])/3./np.mean(isotropic(bank.const)[1:])
    # Numerical ellipsoids may carry only tiny energy in the last tails.
    return {'SF6_S_to_F_site_mean_ratio_upper_for_nondecreasing_weight':max(float(upper),float(baseline)),
            'maximizing_threshold_interval':[float(EDGES[t]),float(EDGES[t+1])],
            'baseline_upper_ratio':float(baseline),'quadrature_node_scope':True}


def axis_geometry(bank,common_scale,imposed_sf6):
    row = bank.geometry
    X = np.array(row['centers'])
    if imposed_sf6:
        lengths = np.linalg.norm(X[1:],axis=1)
        X[1:] *= (GIVEN_R6*common_scale/lengths)[:,None]
    return {**row,'centers':X}


def read_case(bank,G,theta,scale,common_mass_unit):
    sf6 = bank.tag.startswith('SF6')
    row = axis_geometry(bank,scale,sf6)
    ref = np.full(3,SF6_REF) if sf6 else REF
    model = Rotor(row,G,scale,ref)
    standard = model.read(theta)
    native = standard['rotation_moments_native']
    I = native*(1e-10/scale)**2*common_mass_unit
    ABC = H_SI/(8*np.pi**2*100*C_SI*I)
    m = float(np.trace(standard['mass_translation_native'])/3.)
    masses = isotropic(G)
    return {'translation_eigenvalues_native':standard['mass_translation_eigenvalues_native'],
            'translation_anisotropy':standard['translation_anisotropy'],
            'S_over_F1_conditional_mass_ratio':masses[0]/masses[1],
            'local_isotropic_mass_components_native':masses,
            'mass_relative_error_with_same_unit':m*common_mass_unit/(M6 if sf6 else M4)-1.,
            'rotation_constants_same_units_cm_inverse':ABC,
            'rotation_relative_errors_same_units':ABC/ref-1.,
            'unmodified_bond_lengths_angstrom':np.linalg.norm(np.asarray(bank.geometry['centers'])[1:],axis=1)/scale,
            'SF6_experimental_geometry_imposed_in_kinetic_comparison':sf6,
            'SF4_reference_from_this_preparation_used_for_scale':False,
            'nearest_bisector_axis_error_deg':None if sf6 else standard['bisector_nearest_inertia_axis_deg']}


def qualify(curve,bank4,bank6,held):
    a,b = bank4.gram(curve),bank6.gram(curve)
    m4 = isotropic(a).sum()
    iso,formula = ratios(a,b)
    held_mass = []
    for bank in held:
        G = bank.gram(curve)
        m = isotropic(G)
        if bank.tag.startswith('SF6'):
            r = m.sum()/m4/M64-1.
        else:
            r = max(abs(m[0]/m[1]/M_RATIO-1.),abs(m.sum()/m4-1.))
        held_mass.append(float(abs(r)))
    return {'mass_target_residuals':[iso/M_RATIO-1.,formula/M64-1.],
            'held_mass_errors_by_case':dict(zip((k.tag for k in held),held_mass,strict=True)),
            'maximum_held_mass_relative_error':max(held_mass,default=0.)}


def recheck(curve,bank4,bank6,holdout):
    scale = bank4.geometry['source_length_scale']
    G0 = bank4.gram(curve)
    model = Rotor(bank4.geometry,G0,scale,REF)
    points = (2*qmc.Sobol(4,scramble=True,seed=352).random_base2(10)-1)*np.pi
    theta,_ = search(model,np.zeros(4),points)
    mass_unit = M4/isotropic(G0).sum()
    results = {}
    for b in (bank4,bank6,*holdout):
        results[b.tag] = read_case(b,b.gram(curve),np.zeros(6) if b.tag.startswith('SF6') else theta,scale,mass_unit)
    return {'curve':curve,'absolute_mass_unit_kg_from_SF4':mass_unit,
            'common_length_unit_m_from_SF4':1e-10/scale,'SF4_twists_fitted_only_to_SF4_on_cubic_Gauss2':theta,
            'cases':results}


def main():
    banks = {tag:Bank(tag) for tag in CASE}
    train4,train6 = banks[MAIN4],banks[MAIN6]
    reserved = [b for k,b in banks.items() if k not in (MAIN4,MAIN6)]
    curves,sweeps = [],[]
    for q in Q_VALUES:
        for p in P_VALUES:
            found,trace = roots_for(p,q,train4,train6)
            for curve in found:
                curves.append({**curve,**qualify(curve,train4,train6,reserved)})
            sweeps.append({'p':p,'q':q,'sampled_curve':trace,'mass_calibrated_roots':len(found)})
        print('common shape q',q,'mass-calibrated roots',sum(c['q']==q for c in curves),flush=True)
    if not curves:
        raise ArithmeticError('No Hill curve met both calibration mass constraints')
    pure = [r for r in curves if r['q']==0]
    def order(r):
        return (r['maximum_held_mass_relative_error'],abs(r['q']),r['p'],r['a'])
    leading = min(curves,key=order)
    simplest = min(pure or curves,key=lambda r:(abs(r['q']),r['p'],r['a']))
    picks = [leading] if leading==simplest else [simplest,leading]
    selected = {name:recheck({k:v for k,v in p.items() if k in ('a','p','q','kappa')},train4,train6,reserved)
                for name,p in zip(('one_variable_or_simplest','most_stable_mass_transfer') if len(picks)>1 else ('one_variable_or_simplest',),
                                  picks,strict=True)}
    bound = {tag:threshold_envelope(banks[tag]) for tag in CASE if tag.startswith('SF6')}
    results = {'revision':352,'full':True,'physical_validation':False,
        'targets':{'isotope_S32_over_F19':M_RATIO,'formula_mass_ratio_SF6_over_SF4':M64,
                   'cubic_Gauss2_anchors':[MAIN4,MAIN6],'reference_SF4_rotations_cm_inverse':REF,
                   'reserved_SF6_rotation_cm_inverse':SF6_REF,
                   'definition_of_mass':'isotropic part of free translation in the retained fields'},
        'tested_Hill_curves':sweeps,'multiple_mass_calibrated_curves':curves,'selected':selected,
        'monotone_density_only_effective_local_SF6_mass_bounds':bound,
        'chi_definition':'27 det(n,Dn)^2 / tr(Dn Dn^T)^3',
        'search_shape_window':{'p':P_VALUES,'q':Q_VALUES,'kappa':[float(np.exp(LOG_KAPPA[0])),float(np.exp(LOG_KAPPA[-1]))],
                                'amplitude_upper':AMP_MAX},
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__).resolve(),
                *(b.path for b in banks.values()),*(b.path.with_suffix('.json') for b in banks.values()))}}
    dump(OUTPUT/'COURBES_CANDIDATES_R352.json',results)
    print('selected',[(name,c['curve'],c['cases'][MAIN6]['rotation_relative_errors_same_units']) for name,c in selected.items()],flush=True)
    print('monotone bounds',bound,flush=True)


if __name__=='__main__':
    main()
