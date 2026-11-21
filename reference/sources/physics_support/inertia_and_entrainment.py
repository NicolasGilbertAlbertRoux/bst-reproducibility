'Independent isotope-mass anchors for the local substrate response.'

from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from clock_inertia import C_SI,H_SI,U_SI,MASSES_U, REF
from free_rotations import CALIBRATION, OUTPUT as R349, ROOT, kinetic21, reduced
from rotations_spatial import skew
from sizes_substrate import OUTPUT, SHAPES, dilated_blocks
import numpy as np
from scipy.optimize import brentq, least_squares, minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


BANK = OUTPUT/'BANQUE_DILATATIONS_R351.json'
M_RATIO = MASSES_U[0]/MASSES_U[1]
SF6_REF = .09107


def rotate_frames(X,O,theta):
    O = O.copy()
    directions = X[1:]/np.linalg.norm(X[1:],axis=1)[:,None]
    O[1:] = Rotation.from_rotvec(directions*np.asarray(theta)[:,None]).as_matrix()@O[1:]
    return O


def free_local(G):
    permutation = [3,4,5,0,1,2]
    return reduced(G[np.ix_(permutation,permutation)],3)


def blocks(row):
    return tuple(np.array([a[k] for a in row['atoms']]) for k in ('G2_native','beta_G4_native'))


def spin_free_mass(G):
    return float(np.trace(free_local(G))/3.)


def rotor_abc(moments,mass_native,scale,formula_mass):
    physical = moments*formula_mass*(1e-10/scale)**2/mass_native
    return H_SI/(8*np.pi**2*100*C_SI*physical)


def anchor_load(reference):
    G2,G4 = blocks(reference)
    G = G2+G4
    mS,mF = map(spin_free_mass,G[:2])
    missing = M_RATIO*mF-mS
    if missing<=0:
        raise ArithmeticError('The atomic mass ratio does not require positive central entrainment')

    def dilation_error(s):
        S = dilated_blocks(G2[:1],G4[:1],[s])[0]
        return spin_free_mass(S)-M_RATIO*mF
    interval = [.12,4.]
    scan = np.geomspace(*interval,257)
    roots = [brentq(dilation_error,a,b,xtol=1e-12) for a,b in zip(scan[:-1],scan[1:],strict=True)
             if dilation_error(a)*dilation_error(b)<0]
    close = min(roots,key=lambda s:abs(np.log(s)))
    E2,E4 = [reference['atoms'][0][k] for k in ('E2_edges','E4')]
    gradient = E2-reference['beta']*E4/close**2
    return {'mass_isotope_ratio_S_over_F':M_RATIO,'native_angularly_isotropic_components_reference_S_and_F1':[mS,mF],
            'native_translation_eigenvalues_reference_S_and_F1':[np.linalg.eigvalsh(free_local(g)) for g in G[:2]],
            'additive_central_entrainment_native':missing,
            'whole_sulfur_mantle_factor':(mS+missing)/mS,
            'required_central_entrainment_fraction_of_isotopic_S':missing/(mS+missing),
            'sulfur_dilation_mass_anchor_scale':close,'size_roots':roots,'size_search_window':interval,
            'bare_sulfur_size_force_at_mass_anchor':gradient,
            'source_preparation':'R349 free N48. First axial F. No spectrum in the mass anchor.'}


def variants(row,calibration):
    G2,G4 = blocks(row)
    G = G2+G4
    s = np.ones(len(G))
    s[0] = calibration['sulfur_dilation_mass_anchor_scale']
    loaded = G.copy()
    # A dimensionless coefficient learned on N48 is transported with the
    # local native S mass, not as a different dimensional mass unit per grid.
    mass = (calibration['whole_sulfur_mantle_factor']-1)*spin_free_mass(G[0])
    loaded[0,3:,3:] += np.eye(3)*mass
    whole = G.copy()
    whole[0] *= calibration['whole_sulfur_mantle_factor']
    virial = [a['scale_free_self_energy_stationary'] for a in row['atoms']]
    affine = np.array([np.array(a['G2_affine_relaxed'])+a['beta_G4_affine_relaxed'] for a in row['atoms']])
    return {'original':G,'virial_scalar':dilated_blocks(G2,G4,virial),'virial_affine':affine,
            'taille_S_ancree_masse':dilated_blocks(G2,G4,s),
            'entrainment_center':loaded,'manteau_S_integer':whole}


def assess(D,scale,formula_mass,reference,X):
    inertia = reduced(D,3)
    moments,axes = np.linalg.eigh(inertia)
    mass_t = D[3:,3:]
    mt = np.linalg.eigvalsh(mass_t)
    trace_scale = float(np.trace(mass_t)/3.)
    abc = rotor_abc(moments,trace_scale,scale,formula_mass)
    interval = np.array([rotor_abc(moments,x,scale,formula_mass) for x in (mt[0],mt[-1])])
    error = abc/reference-1.
    forbidden = np.maximum(np.maximum(interval[0]/reference-1.,1.-interval[1]/reference),0.)
    bisector = X[1]/np.linalg.norm(X[1])+X[2]/np.linalg.norm(X[2])
    tilt = None
    if np.linalg.norm(bisector)>1e-8:
        bisector /= np.linalg.norm(bisector)
        tilt = float(np.degrees(np.arccos(np.clip(max(abs(bisector@axes)),0.,1.))))
    return {'rotation_moments_native':moments,'mass_translation_native':mass_t,
            'mass_translation_eigenvalues_native':mt,'translation_anisotropy':float(mt[-1]/mt[0]-1),
            'ABC_cm_inverse_if_isotropic_translation':abc,'ABC_envelope_from_translation_directions':interval,
            'relative_errors_if_isotropic_translation':error,'direction_envelope_lower_error':forbidden,
            'bisector_nearest_inertia_axis_deg':tilt}


class Rotor:
    def __init__(self,row,G,scale,reference,J=None):
        self.X,self.O = np.array(row['centers']),np.array(row['frames'])
        self.G = G
        self.Gfree = np.array([free_local(g) for g in G])
        self.V = np.array([np.column_stack((-skew(x),np.eye(3))) for x in self.X])
        self.formula_mass = U_SI*(MASSES_U[0]+(len(G)-1)*MASSES_U[1])
        self.scale,self.reference = scale,reference
        self.J = None if J is None else np.asarray(J)
        if self.J is not None and len(G)!=5:
            raise ValueError('Received contact tangent is specific to SF4')

    def matrix(self,theta):
        O = rotate_frames(self.X,self.O,theta)
        if self.J is None:
            P = O.transpose(0,2,1)@self.V
            return np.einsum('nai,nab,nbj->ij',P,self.Gfree,P)
        J = self.J.copy()
        for j,(old,new) in enumerate(zip(self.O[1:],O[1:],strict=True)):
            sl = slice(3*j+3,3*j+6)
            J[j,sl] = J[j,sl]@old@new.T
        M = kinetic21(self.X,O,self.G)
        # Projection of all 15 free frame velocities onto ker J in the kinetic metric.
        solution = np.linalg.solve(M[6:,6:],J.T)
        B = M[:6,6:]@solution
        return reduced(M,6)+B@np.linalg.solve(J@solution,B.T)

    def read(self,theta):
        return assess(self.matrix(theta),self.scale,self.formula_mass,self.reference,self.X)

    def errors(self,theta):
        D = self.matrix(theta)
        I = np.linalg.eigvalsh(reduced(D,3))
        return rotor_abc(I,float(np.trace(D[3:,3:])/3.),self.scale,self.formula_mass)/self.reference-1.


def search(rotor,starter,points):
    cost = np.array([np.max(abs(rotor.errors(theta))) for theta in points])
    starts = [starter,*points[np.argsort(cost)[:8]]]
    fits = [least_squares(rotor.errors,theta,bounds=(-np.pi,np.pi),max_nfev=240,
                          ftol=2e-11,gtol=2e-11,xtol=2e-11) for theta in starts]
    least = min(fits,key=lambda a:np.max(abs(a.fun)))
    start = np.r_[least.x,np.max(abs(least.fun))]
    solution = minimize(lambda a:a[-1],start,bounds=[(-np.pi,np.pi)]*4+[(0.,2.)],method='SLSQP',
                        constraints={'type':'ineq','fun':lambda x:np.r_[x[-1]-rotor.errors(x[:4]),x[-1]+rotor.errors(x[:4])]},
                        options={'maxiter':200,'ftol':1e-12})
    theta = solution.x[:4] if np.max(abs(rotor.errors(solution.x[:4])))<start[-1] else start[:4]
    return theta,float(min(cost))


def point_rotor(row,scale,target):
    X = np.array(row['centers'])
    masses = np.r_[MASSES_U[0],np.full(len(X)-1,MASSES_U[1])]*U_SI
    COM = masses@X/masses.sum()
    q = X-COM
    I = sum(m*((r@r)*np.eye(3)-np.outer(r,r)) for m,r in zip(masses,q,strict=True))
    values = rotor_abc(np.linalg.eigvalsh(I),1.,scale,1.)
    return {'ABC_cm_inverse':values,'relative_errors':values/target-1.}


def transfer():
    rows = read(BANK)['rows']
    ref = next(r for r in rows if r['preparation']==SHAPES[0] and r['count']==48)
    cal = anchor_load(ref)
    s4 = ref['length_scale_native_per_angstrom']
    slow = read(R349/'REPONSE_CONTACTS_R349.json')
    fast_starter = np.zeros(4)
    slow_starter = np.deg2rad(read(R349/'BALAYAGE_LENT_R349.json')['best_relative_twists_deg'])
    points = (2*qmc.Sobol(4,scramble=True,seed=351).random_base2(11)-1)*np.pi
    output = {}
    for kind,prep,starter in [('free',SHAPES[0],fast_starter),('slow',SHAPES[1],slow_starter),
                              ('prestress_free',SHAPES[2],np.zeros(4))]:
        states = [r for r in rows if r['preparation']==prep]
        variants_by_grid = [variants(row,cal) for row in states]
        output[kind] = {}
        for label in variants_by_grid[0]:
            if kind=='slow' and label=='virial_affine':
                continue
            rotors = []
            for index,row in enumerate(states):
                jacobian = slow['rows'][index]['contact_direction_jacobian'] if kind=='slow' else None
                rotors.append(Rotor(row,variants_by_grid[index][label],row['length_scale_native_per_angstrom'],REF,J=jacobian))
            angles,best_sample = search(rotors[0],starter,points)
            fitted = [r.read(angles) for r in rotors]
            prior = [r.read(starter) for r in rotors]
            output[kind][label] = {'twists_rad_fitted_on_N48':angles,'sobol_samples_N48':len(points),
                'min_sampled_error_N48':best_sample,'grid_counts':[r['count'] for r in states],
                'readings_after_torsion_fit':fitted,'readings_at_received_angles':prior}
            print(kind,label,'new N48/N64 max error',[max(abs(x['relative_errors_if_isotropic_translation'])) for x in fitted],
                  'mass anisotropy',[x['translation_anisotropy'] for x in fitted],flush=True)

    row6 = next(r for r in rows if r['species']=='SF6')
    geom6 = np.linalg.norm(np.array(row6['centers'])[1:],axis=1)/s4
    reference6 = np.full(3,SF6_REF)
    sf6 = {'geometry_angstrom_with_same_SF4_scale':geom6,'reference_length_angstrom':1.5607,
           'expected_B_cm_inverse':SF6_REF,'zero_extra_torsions':True,
           'results':{name:Rotor(row6,G,s4,reference6).read(np.zeros(6)) for name,G in variants(row6,cal).items()}}
    data = {'revision':351,'full':True,'mass_anchor':cal,'R349_fixed_geometries':True,
            'point_mass_control_SF4':point_rotor(ref,s4,REF),'point_mass_control_SF6':point_rotor(row6,s4,reference6),
            'results_SF4':output,'reserved_SF6':sf6,'physical_validation':False,
            'reading':'trace/3 is the unique scalar necessary in an isotropic translation sector; eigenvalues and anisotropy are reported separately',
            'scope':'kinetic-only tests; the slow tangent at changed scalar sizes assumes the normalized ports and reference centre distances held',
            'references':{'SF4':'https://cccbdb.nist.gov/exp2x.asp?casno=7783600',
                          'SF6':'https://cccbdb.nist.gov/exp2x.asp?casno=2551624&charge=0',
                          'isotopes':'https://physics.nist.gov/cgi-bin/MolSpec/mole.pl?molecule=SF&prefix=di&type=fancy'},
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (BANK,CALIBRATION,
                R349/'REPONSE_CONTACTS_R349.json',R349/'BALAYAGE_LENT_R349.json',Path(__file__).resolve())}}
    dump(OUTPUT/'ENTRAINEMENT_ET_ROTATIONS_R351.json',data)
    print('mass anchor',cal,'SF6 held out',geom6,flush=True)


if __name__=='__main__':
    transfer()
