'Screen smooth vacuum-flat size laws on the received SF4 and SF6 states.'

from concurrent.futures import ThreadPoolExecutor
import ctypes
from pathlib import Path

from sweep_sf4 import read
from calibrate_curves import isotropic
from fields_and_references import dump
from dynamics_full import NodalKinetic
from dynamics_nodal_scripts import VACUUM,pad_cube
from dynamics_size import OUTPUT,ROOT,SIZES
from trials_nodal import digest
from flux_nodal import replace_once
from clock_inertia import MASSES_U,U_SI
from inertia_sizes import SF4,SF6,load,report
from scope_spatial import AMPLITUDES,CENTERS_X,PADDING,halo_patch
from refit_dependencies import ARRAY,load_c
from free_rotations import CALIBRATION
from size_dynamics import Radius
import numpy as np
from scipy.optimize import brentq


SOURCE = Path(__file__).resolve()
BASE = ROOT/'sources'/'cinetique_complete_R343.c'
BANK = ROOT/'scripts'/'banque_coeur_R357.c'
GATES = np.array([[-.6,0.],[-.5,.25],[-.25,.5],[0.,.5],[0.,.75],[.25,.75],
                  [.25,.90],[.5,.90],[.5,.95],[.75,.95]])
FILES = {'SF4_N48':SF4,'SF6_N48':SF6,
         **{f'{name}_N{n}':SF4.parent/f'ETAT_{name}_N{n}_R355.npz'
            for name in ('SF4','SF6') for n in (56,60,64)}}
REFS = {'SF4':'https://cccbdb.nist.gov/exp2x.asp?casno=7783600',
        'SF6':'https://cccbdb.nist.gov/exp2x.asp?casno=2551624&charge=0'}


def kernel_source():
    src = BASE.read_text()
    first = src.split('static void link(',1)[0]
    body = 'int bst_general_mass('+src.split('int bst_general_mass(',1)[1]
    body = replace_once(body,
        'int bst_general_mass(int n,double h,double alpha,double beta,const double*field,double*A,double*C,double*G)',
        'int bst_r357_core_bank(int n,double h,double alpha,double beta,int gates,const double*thresholds,const double*field,double*moments,double*grams)')
    body = replace_once(body,';const ptrdiff_t node=pos/4;',';')
    body = replace_once(body,'double K[4][4],R[9][4]={{0.}},KR[4][9];',
        '''double K[4][4],R[6][4]={{0.}},KR[4][6];
   double rho=fabs(dot4(base,(double[]){
       minor(d[0],d[1],d[2],1,2,3),-minor(d[0],d[1],d[2],0,2,3),
       minor(d[0],d[1],d[2],0,1,3),-minor(d[0],d[1],d[2],0,1,2)}));
   double radial=xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2];''')
    body = replace_once(body,';A[node*16+4*j+k]+=K[j][k]','')
    marker = '''    double axis[3]={0.},rot[3],adv[3];axis[a]=1.;cross3(axis,base+1,rot);cross3(axis,xi,adv);
    for(int j=0;j<3;j++)R[a][j+1]=-rot[j];
    for(int j=0;j<4;j++){R[a+6][j]=-d[a][j];for(int b=0;b<3;b++)R[a+3][j]-=adv[b]*d[b][j];}'''
    body = replace_once(body,marker,'''    double axis[3]={0.},adv[3];axis[a]=1.;cross3(axis,xi,adv);
    for(int j=0;j<4;j++){R[a+3][j]=-d[a][j];for(int b=0;b<3;b++)R[a][j]-=adv[b]*d[b][j];}''')
    old = '''   for(int j=0;j<4;j++)for(int a=0;a<9;a++){double v=0.;for(int k=0;k<4;k++)v+=K[j][k]*R[a][k];KR[j][a]=v;C[node*36+9*j+a]+=v;}
   for(int a=0;a<9;a++)for(int b=0;b<=a;b++){double v=0.;for(int j=0;j<4;j++)v+=R[a][j]*KR[j][b];G[9*a+b]+=v;if(a!=b)G[9*b+a]+=v;}'''
    new = '''   for(int j=0;j<4;j++)for(int a=0;a<6;a++){
    double v=0.;for(int k=0;k<4;k++)v+=K[j][k]*R[a][k];KR[j][a]=v;
   }
   double local[36];
   for(int a=0;a<6;a++)for(int b=0;b<=a;b++){
    double v=0.;for(int j=0;j<4;j++)v+=R[a][j]*KR[j][b];local[6*a+b]=local[6*b+a]=v;
   }
   for(int g=0;g<=gates;g++){
    double gate=(g==0)?1.:window(base[0],thresholds[2*(g-1)],thresholds[2*(g-1)+1]);
    if(gate==0.)continue;
    for(int j=0;j<36;j++)grams[36*g+j]+=gate*local[j];
    double density=gate*rho;
    moments[5*g]+=density;
    for(int a=0;a<3;a++)moments[5*g+1+a]+=density*xi[a];
    moments[5*g+4]+=density*radial;
   }'''
    body = replace_once(body,old,new)
    return first+'''#include <math.h>

static double minor(const double *a,const double *b,const double *c,int i,int j,int k){
 return a[i]*(b[j]*c[k]-b[k]*c[j])-a[j]*(b[i]*c[k]-b[k]*c[i])+a[k]*(b[i]*c[j]-b[j]*c[i]);
}

/* A C2 target-space cutoff, flat in the vacuum hemisphere near n0=1. */
static double window(double scalar,double lo,double hi){
 if(scalar>=hi)return 0.;
 if(scalar<=lo)return 1.;
 double t=(hi-scalar)/(hi-lo);
 return t*t*t*(10.+t*(-15.+6.*t));
}

'''+body


class Bank:
    def __init__(self,beta):
        self.beta = beta
        text = kernel_source()
        if BANK.exists() and BANK.read_text()!=text:
            raise ArithmeticError('The gated bank differs from the exact parent transformation')
        if not BANK.exists():
            BANK.write_text(text)
        self.f,self.sha = load_c(BANK,'bst_r357_core_bank',
            [ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ARRAY,ARRAY,ARRAY,ARRAY])

    def __call__(self,n,h):
        M = np.zeros((len(GATES)+1,5))
        G = np.zeros((len(GATES)+1,6,6))
        status = self.f(round(len(n)**(1/3)),h,1.,self.beta,len(GATES),GATES,np.ascontiguousarray(n),M,G)
        if status or min(M[:,0])<=0.:
            raise ArithmeticError('The core bank lacks positive support')
        radii = np.sqrt(M[:,4]/M[:,0]-np.sum((M[:,1:4]/M[:,0,None])**2,axis=1))
        return radii,G


def solve_exponent(R,G,i):
    iso = lambda matrix: float(isotropic(matrix[None])[0])
    a,b = iso(G[0,0]),iso(G[1,0])
    target = b*MASSES_U[0]/MASSES_U[1]
    if a>=target:
        raise ArithmeticError('The received native mass no longer calls for positive loading')
    def mismatch(extra):
        return iso(G[0,0]+extra*G[0,i])-target
    hi = 1.
    while mismatch(hi)<0. and hi<1024.:
        hi *= 2.
    if mismatch(hi)<0.:
        return None
    extra = brentq(mismatch,0.,hi,xtol=2e-14)
    log_lever = np.log(R[0,i]/R[1,i])
    if abs(log_lever)<1e-6:
        return None
    return np.log1p(extra)/log_lever,float(R[1,i]),float(MASSES_U[1]*U_SI/b)


def assess(data,i,power,ref,unit,scale):
    out = {}
    for name,(R,G,g) in data.items():
        w = (R[:,i]/ref)**power
        local = G[:,0]+(w-1.)[:,None,None]*G[:,i]
        if min(np.linalg.eigvalsh(local).ravel())<=0.:
            raise ArithmeticError('Gated kinetic matrix lost positivity')
        out[name] = {'site_RMS_radii_native':R[:,i],'core_multipliers_by_site':w,
                     **report(local,g,scale,unit,name.startswith('SF6'))}
    return out


def distant_control(bank,n,h):
    base = pad_cube(n,48,PADDING,VACUUM)
    R,G = bank(base,h)
    variants = []
    for center in CENTERS_X:
        v,_ = halo_patch(base,center,AMPLITUDES[-1])
        r,q = bank(v,h)
        defect = max(abs(r[1:]-R[1:]))
        if defect>2e-12 or np.max(abs(q[1:]-G[1:]))>3e-10:
            raise ArithmeticError('A near-vacuum patch changed the gated support or added-mass block')
        variants.append({'patch_center_native':center*h,'amplitude':AMPLITUDES[-1],
                         'full_ungated_radius_change_native':r[0]-R[0],
                         'gated_radius_max_change_native':defect,
                         'gated_additional_kinetic_block_max_defect':np.max(abs(q[1:]-G[1:]))})
    return variants


def main():
    beta = read(CALIBRATION)['candidate']['beta']
    r355 = read(SIZES)
    scale = r355['common_length_scale_native_per_angstrom']
    bank = Bank(beta)
    radius = Radius()
    unweighted = NodalKinetic(beta)
    data = {}
    for name,path in FILES.items():
        n,g = load(path)
        with ThreadPoolExecutor(max_workers=4) as pool:
            raw = list(pool.map(lambda a:bank(*a),zip(n,g['h_native'],strict=True)))
        R,G = np.array([z[0] for z in raw]),np.array([z[1] for z in raw])
        data[name] = (R,G,g)
        print('Core bank',name,'radius interval',R[:,1:].min(),R[:,1:].max(),flush=True)
        if name=='SF4_N48':
            r_checks = [abs(radius(v,h)[0]-R[i,0]) for i,(v,h) in enumerate(zip(n,g['h_native'],strict=True))]
            g_checks = [np.max(abs(unweighted.mass(n[i],g['h_native'][i])[2][3:,3:]-G[i,0])) for i in (0,1)]
            if max(r_checks)>2e-12 or max(g_checks)>3e-9:
                raise ArithmeticError('The ungated native mass and radius were not reproduced')
            proof = distant_control(bank,n[1],float(g['h_native'][1]))
    R,G,_ = data['SF4_N48']
    source_law = r355['calibration_of_size_law']
    old = assess(data,0,source_law['power'],source_law['reference_F_support_radius_native'],
                 source_law['common_mass_unit_kg'],scale)
    candidates = []
    for i,gate in enumerate(GATES,1):
        fit = solve_exponent(R,G,i)
        if fit is None:
            candidates.append({'thresholds':gate,'calibratable_from_two_anchors':False})
            continue
        power,ref,unit = fit
        p = assess(data,i,power,ref,unit,scale)
        training = p['SF4_N48']['translation_mass_site_errors_relative'][:2]
        if max(abs(training))>2e-11:
            raise ArithmeticError('The two received mass anchors did not replay')
        candidates.append({'thresholds':gate,'calibratable_from_two_anchors':True,
                           'exponent':power,'reference_radius_native':ref,'mass_unit_kg':unit,'cases':p})
        print('Gate',gate,'p',power,'SF6 held-out mass err',p['SF6_N48']['total_mass_error_with_common_unit'],
              'SF6 N60 mass err',p['SF6_N60']['total_mass_error_with_common_unit'],flush=True)
    paths = (SOURCE,BASE,BANK,SIZES,CALIBRATION,*FILES.values(),ROOT/'scripts'/'portee_spatiale_R357.py')
    dump(OUTPUT/'COURBES_INERTIE_COEUR_R357.json',
         {'revision':357,'full':True,'physical_validation':False,'anchor_sites':['SF4_N48_S','SF4_N48_axial_F1'],
          'scope':'static finite native kinetic matrices; gate applied to support and inertial loading; no dynamical validation',
          'gate_definition':'quintic S((hi-n0)/(hi-lo)) clipped to [0,1], where S(t)=t^3*(10-15*t+6*t^2)',
          'kinetic_multiplier':'1+[(R_gate/R_ref)^p-1]*gate(n0)',
          'all_caps_flat_in_vacuum':True,'vacuum_linear_wave_speed_native_for_all_gates':1.,
          'whole_site_law_vacuum_speed_native_by_site_at_SF6_N48':1./np.sqrt(old['SF6_N48']['core_multipliers_by_site']),
          'no_rotational_observables_or_fine_grids_used_in_anchor_fit':True,'rotation_references':REFS,
          'base_mass_block_replay_defects':g_checks,'base_radius_replay_defects':r_checks,
          'vacuum_patch_controls':proof,'whole_site_R355_law_comparison':old,
          'gate_catalog_all_results':candidates,'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in paths}})


if __name__=='__main__':
    main()
