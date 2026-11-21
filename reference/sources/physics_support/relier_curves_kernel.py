'Separate continuous-profile calibration from the native finite-stencil experiment.'

from pathlib import Path

from sweep_sf4 import read
from collection_curves import EDGES,OUTPUT,matrices
from collection_native_curves import NATIVE,native_fields
from calibrate_curves import M4,M6,REF,SF6_REF,X,isotropic
from fields_and_references import dump
from curves_of_distribution import POSITIONS,WIDTHS,constraints,contract,laws,shape,solve
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from clock_inertia import C_SI,H_SI
from inertia_and_entrainment import Rotor,search
from kernel_curves import CurveKinetic,SOURCE as C_SOURCE
from free_rotations import ROOT
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc


DIRECT = OUTPUT/'CALAGE_DIRECT_R352.json'
PORTS = OUTPUT/'PORTS_MULTICAS_R352.json'
F4,F6 = 'SF4_N48','SF6_N48'


def load_bank(name):
    p = OUTPUT/f'BANQUE_{name}_R352.npz'
    with np.load(p) as z:
        if not np.array_equal(z['edges'],EDGES):
            raise ValueError('Native bin locations changed')
        return z['bins'],z['constant'],read(p.with_suffix('.json')),p


def exact_blocks(fields,g,beta,terms):
    model = CurveKinetic(beta,terms)
    return np.array([model.mass(n,h)[2][3:,3:] for n,h in zip(fields,g['h_native'],strict=True)])


def candidates(banks,direct):
    catalog = [(t['center'],t['width_log'],0) for t in direct]
    for scheme in ('same_centers','native_catalog'):
        if scheme=='native_catalog':
            catalog = [(c,w,0) for w in WIDTHS for c in POSITIONS]
        B4,B6 = [matrices(np.array([np.einsum('axj,x->aj',banks[name][0],shape(X,np.ones_like(X),c,w,q))
                                   for c,w,q in catalog])) for name in (F4,F6)]
        weights = np.array([1.+.22/w for _,w,_ in catalog])
        a,test = solve(banks[F4][1],banks[F6][1],B4,B6,weights)
        if a is not None:
            return scheme,laws(catalog,a),test
    raise ArithmeticError(f'No positive law reached the native training masses: {test}')


def variation(field,h,beta,terms):
    kernel = CurveKinetic(beta,terms)
    n = np.ascontiguousarray(field)
    t = np.zeros_like(n)
    t[:,1:] = np.cross(n[:,1:],np.array([.012,-.006,.004]))
    v = .018*np.sin(.57*np.arange(9)+.13)
    q,gn,gt,gv = kernel.quadratic(n,h,t,v)
    A,C,G = kernel.mass(n,h)
    quadratic = np.sum(np.einsum('ni,nij,nj->n',t,A,t))+2*np.einsum('ni,nia,a->',t,C,v)+v@G@v
    mass_defect = abs(quadratic-q)/max(1.,abs(q))
    dn = np.sin(.021*np.arange(n.size)).reshape(n.shape)
    dn -= n*np.sum(n*dn,axis=1)[:,None]
    dn /= np.linalg.norm(dn)
    dt = np.cos(.017*np.arange(t.size)).reshape(t.shape)
    dt /= np.linalg.norm(dt)
    dv = np.sin(.27*np.arange(9)+.43)
    exact = float(np.sum(gn*dn)+np.sum(gt*dt)+gv@dv)
    diffs = []
    for eps in (4e-6,2e-6):
        q1 = kernel.quadratic(n+eps*dn,h,t+eps*dt,v+eps*dv)[0]
        q2 = kernel.quadratic(n-eps*dn,h,t-eps*dt,v-eps*dv)[0]
        fd = (q1-q2)/(2*eps)
        diffs.append({'step':eps,'analytic':exact,'numeric':fd,'abs_defect':abs(fd-exact)})
    if mass_defect>2e-11 or max(z['abs_defect'] for z in diffs)>1e-6*max(1.,abs(exact)):
        raise ArithmeticError(f'Nonlinear variational check failed: {mass_defect}, {diffs}')
    return {'native_mass_kinetic_relative_defect':mass_defect,'finite_differences':diffs}


def read_rotor(G,row,scale,munit,sf6,theta,ported_lengths):
    geometry = {**row}
    if sf6:
        X0 = np.asarray(row['centers'])
        X = X0.copy()
        X[1:] *= (ported_lengths/np.linalg.norm(X[1:],axis=1))[:,None]
        geometry['centers'] = X
    ref = np.full(3,SF6_REF) if sf6 else REF
    r = Rotor(geometry,G,scale,ref).read(theta)
    native = np.asarray(r['rotation_moments_native'])
    abc = H_SI/(8*np.pi**2*100*C_SI*native*(1e-10/scale)**2*munit)
    return {'rotation_constants_cm_inverse':abc,'rotation_errors_relative':abc/ref-1.,
            'total_mass_error_with_common_unit':isotropic(G).sum()*munit/(M6 if sf6 else M4)-1.,
            'translation_anisotropy':r['translation_anisotropy'],
            'translation_eigenvalues_native':r['mass_translation_eigenvalues_native'],
            'SF6_radial_replay_geometry_applied':sf6}


def fit_native(constant,bases,initial):
    fixed4,fixed6 = constant[F4],constant[F6]
    B4,B6 = bases[F4],bases[F6]
    m4,m6 = isotropic(fixed4),isotropic(fixed6)
    scale = np.array([m4[0],m6[0],m6.sum()])

    def grams(a):
        return fixed4+np.tensordot(a,B4,axes=1),fixed6+np.tensordot(a,B6,axes=1)

    def residual(a):
        g4,g6 = grams(a)
        return constraints(isotropic(g4),isotropic(g6))/scale

    def jacobian(a):
        g4,g6 = grams(a)
        return constraints(contract(B4,g4),contract(B6,g6))/scale[:,None]

    opt = least_squares(residual,initial,jac=jacobian,bounds=(0.,1e4),
                        ftol=1e-13,xtol=1e-13,gtol=1e-13,max_nfev=250)
    return opt.x,{'relative_equation_residuals':opt.fun,
                  'initial_relative_equation_residuals':residual(initial),
                  'mass_constraints_met':bool(max(abs(opt.fun))<2e-8),
                  'number_active_windows':int(sum(opt.x>1e-9)),
                  'jacobian_singular_values':np.linalg.svd(opt.jac,compute_uv=False)}


def main():
    continuum = read(DIRECT)
    direct = continuum['fitted_curve']
    direct_mass_unit = continuum['calibration_and_holdouts']['common_mass_unit_kg']
    direct_angles = np.array(continuum['calibration_and_holdouts']['torsions_trained_only_on_SF4'])
    ported_lengths = np.array([r['radius_native'] for r in read(PORTS)['relative_offset_after_conditioning_on_molecule']['SF6']['radial_replay']])
    banks = {name:load_bank(name) for name in NATIVE}
    scheme,terms,initial = candidates(banks,direct)
    model = NodalDynamics(48)
    scale = banks[F4][2]['geometry']['source_length_scale']
    results,controls,fields,bases,native_const = {},[],{},{},{}
    for name in NATIVE:
        n,g,beta = native_fields(model,name)
        raw = exact_blocks(n,g,beta,direct)
        sf6 = name.startswith('SF6')
        results[name] = {'continuous_curve_without_shape_refit':read_rotor(raw,g,scale,direct_mass_unit,sf6,
                            np.zeros(6) if sf6 else direct_angles,ported_lengths)}
        print(name,'continuum curve transfer',results[name]['continuous_curve_without_shape_refit']['rotation_errors_relative'],flush=True)
        if name in (F4,F6):
            fields[name] = n,g,beta
    for name,(n,g,beta) in fields.items():
        G = banks[name][1]
        baseline_defect = float(np.max(abs(exact_blocks(n,g,beta,[])-G)))
        if baseline_defect>2e-8:
            raise ArithmeticError(f'Native baseline differs for {name}: {baseline_defect}')
        native_const[name] = G
        bases[name] = np.array([exact_blocks(n,g,beta,[{**term,'amplitude':1.}])-G for term in terms])
    a,proof = fit_native(native_const,bases,np.array([t['amplitude'] for t in terms]))
    initial_exact = proof
    if not proof['mass_constraints_met']:
        # Binning can shift the optimum across the edge of the positive cone.
        # Add neighbours on both sides of the two selected low-density centres.
        low = [t for t in terms if t['center']<1e-5]
        nearby = [{**t,'center':t['center']*np.exp(delta),'amplitude':0.}
                  for t in low for delta in (-.60,.60)]
        for name,(n,g,beta) in fields.items():
            G = native_const[name]
            extra = [exact_blocks(n,g,beta,[{**t,'amplitude':1.}])-G for t in nearby]
            if extra:
                bases[name] = np.concatenate((bases[name],extra))
        terms = [*terms,*nearby]
        a,proof = fit_native(native_const,bases,np.r_[a,np.zeros(len(nearby))])
        scheme += '_exact_neighbours'
    terms = [{**t,'amplitude':float(k)} for t,k in zip(terms,a,strict=True) if k>1e-10]
    print('exact native mass constraints',proof,flush=True)
    a4 = native_const[F4]+np.tensordot(a,bases[F4],axes=1)
    common = M4/isotropic(a4).sum()
    rotor = Rotor(fields[F4][1],a4,scale,REF)
    points = (2*qmc.Sobol(4,scramble=True,seed=3352).random_base2(10)-1)*np.pi
    angles,_ = search(rotor,np.zeros(4),points)
    for name in NATIVE:
        n,g,beta = fields[name] if name in fields else native_fields(model,name)
        local = exact_blocks(n,g,beta,terms)
        if name==F4:
            if np.max(abs(a4-local))>2e-8:
                raise ArithmeticError('Native kinetic reassembly differs from calibrated matrices')
            controls = [variation(a,h,beta,terms) for a,h in zip(n[:2],g['h_native'][:2],strict=True)]
        sf6 = name.startswith('SF6')
        result = read_rotor(local,g,scale,common,sf6,np.zeros(6) if sf6 else angles,ported_lengths)
        results[name]['native_adjusted_curve'] = result
        print(name,'native ABC',result['rotation_errors_relative'],
              'mass',result['total_mass_error_with_common_unit'],flush=True)
    paths = (DIRECT,PORTS,C_SOURCE,ROOT/'scripts'/'noyau_courbes_R352.py',Path(__file__).resolve(),
             *(r[3] for r in banks.values()))
    record = {'revision':352,'full':True,'physical_validation':False,'native_terms':terms,
              'native_shape_rule':scheme,'native_mass_anchor':'SF4 and SF6 at N48',
              'native_amplitude_fit':proof,'initial_exact_support_fit':initial_exact,
              'initial_binned_shape_fit':initial,'same_native_terms_on_N64':True,
              'native_common_mass_unit_kg':common,'native_scale_per_angstrom':scale,
              'native_SF4_torsions_rad':angles,'kernel_variations':controls,'cases':results,
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in paths}}
    dump(OUTPUT/'NOYAU_ET_TRANSFERT_R352.json',record)


if __name__=='__main__':
    main()
