'Fit smooth positive response windows across mass cases, reserving SF6 rotation.'

from pathlib import Path

from sweep_sf4 import read
from collection_curves import CASE,OUTPUT,matrices
from calibrate_curves import Bank,MAIN4,MAIN6,M4,M64,M_RATIO,REF,X,Y,isotropic,read_case
from fields_and_references import dump
from trials_nodal import digest
from clock_inertia import MASSES_U
from inertia_and_entrainment import Rotor,search
from free_rotations import ROOT
import numpy as np
from scipy.optimize import linprog,minimize
from scipy.stats import qmc


M6_FRACTION = MASSES_U[0]/(MASSES_U[0]+6*MASSES_U[1])
POSITIONS = np.geomspace(1e-8,8.,43)
WIDTHS = (.60,1.20,2.40)
Q = (0,1,2,4,-1,-2,-4)


def shape(x,y,center,width,power):
    use = x>0.
    log = np.zeros_like(x)
    np.log(x/center,out=log,where=use)
    gaussian = np.exp(-.5*(log/width)**2)*use
    anisotropy = y**power if power>=0 else (1.-y)**(-power)
    return gaussian*anisotropy


def window_matrices(bank,catalog):
    return matrices(np.array([np.einsum('axj,x->aj',bank.collapse(q),shape(X,np.ones_like(X) if q>=0 else np.zeros_like(X),
                                                               c,w,q))
                              for c,w,q in catalog]))


def constraints(m4,m6):
    return np.array([m4[...,0]-M_RATIO*m4[...,1],
                     m6[...,0]-M6_FRACTION*np.sum(m6,axis=-1),
                     np.sum(m6,axis=-1)-M64*np.sum(m4,axis=-1)])


def contraction(G):
    s = np.linalg.solve(G[:,:3,:3],-G[:,:3,3:])
    return np.concatenate((s,np.broadcast_to(np.eye(3),(len(s),3,3))),axis=1)


def contract(blocks,at):
    P = contraction(at)
    return np.einsum('aip,...aij,ajp->...a',P,blocks,P)/3.


def solve(G4,G6,B4,B6,weights):
    'Positive window amplitudes, with exact derivatives of the reduced masses.'
    x = np.zeros(len(weights))
    scale = np.array([M_RATIO*isotropic(G4)[1],isotropic(G6)[0],isotropic(G6).sum()])

    def gram(z,G,B):
        return G+np.tensordot(z,B,axes=1)

    for _ in range(10):
        t4,t6 = gram(x,G4,B4),gram(x,G6,B6)
        linear = constraints(contract(B4,t4),contract(B6,t6))/scale[:,None]
        base = constraints(contract(G4,t4),contract(G6,t6))/scale
        opt = linprog(weights,A_eq=linear,b_eq=-base,bounds=(0.,1e4),method='highs')
        if not opt.success:
            return None,{'linear_program_status':opt.status,'message':str(opt.message)}
        x = opt.x
        err = constraints(isotropic(gram(x,G4,B4)),isotropic(gram(x,G6,B6)))/scale
        if max(abs(err))<2e-10:
            break
    if max(abs(err))>=2e-10:
        def exact(z):
            return constraints(isotropic(gram(z,G4,B4)),isotropic(gram(z,G6,B6)))/scale

        def exact_jac(z):
            a,b = gram(z,G4,B4),gram(z,G6,B6)
            return constraints(contract(B4,a),contract(B6,b))/scale[:,None]

        opt = minimize(lambda a:a@weights,x,jac=lambda a:weights,
                       method='SLSQP',bounds=[(0.,1e4)]*len(x),
                       constraints={'type':'eq','fun':exact,'jac':exact_jac},
                       options={'maxiter':240,'ftol':1e-11})
        x,err = opt.x,exact(opt.x)
    if max(abs(err))>=2e-8:
        return None,{'maximum_exact_constraint_relative_residual':float(max(abs(err)))}
    return x,{'weighted_amplitude_objective':float(x@weights),'relative_equation_residuals':err,
              'number_active_windows':int(sum(x>1e-9))}


def laws(catalog,a):
    return [{'center':float(c),'width_log':float(w),'chi_power':int(q),'amplitude':float(z)}
            for (c,w,q),z in zip(catalog,a,strict=True) if z>1e-10]


def curve_multiplier(x,y,terms):
    return 1.+sum(row['amplitude']*shape(x,y,row['center'],row['width_log'],row['chi_power']) for row in terms)


def predict(banks,grams):
    b4 = banks[MAIN4]
    scale = b4.geometry['source_length_scale']
    munit = M4/isotropic(grams[MAIN4]).sum()
    points = (2*qmc.Sobol(4,scramble=True,seed=1352).random_base2(10)-1)*np.pi
    rotor = Rotor(b4.geometry,grams[MAIN4],scale,REF)
    twists,_ = search(rotor,np.zeros(4),points)
    measured = {name:read_case(b,grams[name],np.zeros(6) if name.startswith('SF6') else twists,scale,munit)
                for name,b in banks.items()}
    return {'common_mass_unit_kg':munit,'common_length_unit_m':1e-10/scale,
            'torsions_trained_only_on_SF4':twists,'cases':measured}


def main():
    banks = {tag:Bank(tag) for tag in CASE}
    outcomes = {}
    families = [('only_density_windows',(0,),WIDTHS),
                ('density_and_shape_windows',Q,WIDTHS),
                ('density_and_shape_broad_only',Q,(1.20,2.40))]
    for name,powers,widths in families:
        catalog = [(c,w,q) for q in powers for w in widths for c in POSITIONS]
        weights = np.array([1.+.22/w+.08*abs(q) for _,w,q in catalog])
        bases = {k:window_matrices(b,catalog) for k,b in banks.items()}
        a,test = solve(banks[MAIN4].const,banks[MAIN6].const,bases[MAIN4],bases[MAIN6],weights)
        if a is None:
            outcomes[name] = {'mass_constraints_met':False,'diagnostic':test}
            print(name,test,flush=True)
            continue
        G = {tag:b.const+np.tensordot(a,bases[tag],axes=1) for tag,b in banks.items()}
        terms = laws(catalog,a)
        out = predict(banks,G)
        outcomes[name] = {'mass_constraints_met':True,'diagnostic':test,'terms':terms,**out}
        print(name,'terms',terms,'SF6 reserved',[out['cases'][tag]['rotation_relative_errors_same_units']
                     for tag in (MAIN6,'SF6_R336_C3','SF6_R321_C2')],flush=True)
    record = {'revision':352,'full':True,'physical_validation':False,
        'constraints':['S32/F19 in SF4, first axial fluorine as reference',
                       'SF6 central share given by the isotope mass partition as an explicit model hypothesis',
                       'formula mass ratio SF6/SF4 in one unit'],
        'rotation_SF6_excluded_from_mass_fit':True,'catalog':{'centers':POSITIONS,'widths':WIDTHS,'chi_powers':Q},
        'families':outcomes,'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (
            Path(__file__).resolve(),ROOT/'scripts'/'calibrer_courbes_R352.py',
            *(b.path for b in banks.values()),*(b.path.with_suffix('.json') for b in banks.values()))}}
    dump(OUTPUT/'FENETRES_DE_REPARTITION_R352.json',record)


if __name__=='__main__':
    main()
