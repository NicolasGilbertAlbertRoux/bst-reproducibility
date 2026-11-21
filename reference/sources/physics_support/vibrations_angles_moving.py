'Read the native SF6 tangent response with bond and port directions mobile.'

from pathlib import Path

from angles_internal import move,pullback
from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from clock_inertia import C_SI,H_SI
from reader_transported import FixedReader
from read_nodes import mass_laws
from modes_mixed import CONTRAST,PAIR,SIMPLE
from modes_radial import interpolate_rates
from kernel_curves import CurveKinetic
from refit_pose import Pose,tangent_basis
from release_collar import face_interior
from relax_nodes import ROOT,NodalSF6,projected
from free_rotations import reduced
import numpy as np
from scipy.linalg import eigh


OUTPUT = ROOT/'results'/'R355'
R354 = ROOT/'results'/'R354'
STATE = R354/'ETAT_FACES_FIXES_N48_R354.npz'
BASIS = R354/'BASE_REPONSE_POLI_N48_R354.npz'
OLD = R354/'POLES_FACES_FIXES_R354.json'
SIZES = OUTPUT/'INERTIE_COMMUNE_R355.json'
SOURCE = Path(__file__).resolve()


class Tangent:
    'Three field directions together with the radial, bond and port charts.'

    def __init__(self,model,n,r,d,O,Y):
        self.model,self.n,self.r,self.d,self.O,self.Y = model,n,r,d,O,Y
        self.P = Pose(model,n,d,O)
        self.B = tangent_basis(d)
        self.m = len(r)
        self.dimension = 5*self.m+len(Y)

    def evaluate(self,x):
        m = self.m
        dstep,fstep = x[m:3*m].reshape(m,2),x[3*m:5*m].reshape(m,2)
        w = np.einsum('nia,ni->na',self.B,dstep)
        v = np.einsum('nia,ni->na',self.P.B,fstep)
        d,f = move(self.d,w),move(self.P.f0,v)
        O = self.P.frames(d,f)
        q = self.n+np.einsum('i,inac->nac',x[5*m:],self.Y)
        norm = np.linalg.norm(q,axis=-1,keepdims=True)
        n = q/norm
        E,g = self.model.evaluate(n,True,r=self.r+x[:m],D=d,frames=O,pose=True)
        state = self.model.last
        torque = state['frame_rotation_gradient_world']
        gd = state['direction_gradient']+np.cross(d,torque)
        gf = np.cross(np.einsum('nji,nj->ni',O,torque),f)
        pd = np.einsum('nia,na->ni',self.B,pullback(self.d,w,gd)).ravel()
        pf = np.einsum('nia,na->ni',self.P.B,pullback(self.P.f0,v,gf)).ravel()
        pg = np.einsum('inac,nac->i',self.Y,g/norm)
        return E,np.r_[state['radial_force_native'],pd,pf,pg]

    def kinetic(self,fields,hs,Y,beta,terms,weights):
        m,dim = self.m,self.dimension
        total = dim+6+m
        model = CurveKinetic(beta,terms)
        out = np.zeros((total,total))
        h_fields = slice(5*m,dim)
        for i,(n,h,J,weight) in enumerate(zip(fields,hs,Y,weights,strict=True)):
            A,C,G = model.mass(n,h)
            V = np.zeros((9,total))
            O = np.eye(3) if not i else self.O[i-1]
            V[3:6,dim:dim+3] = O.T
            V[6:,dim+3:dim+6] = O.T
            if i:
                j = i-1
                d,r = self.d[j],self.r[j]
                f = self.P.f0[j]
                db,fb = self.B[j],self.P.B[j]
                V[6:,j] = O.T@d
                V[6:,m+2*j:m+2*j+2] = r*O.T@db.T
                V[6:,dim:dim+3] = O.T@np.cross(np.eye(3),r*d).T
                V[3:6,m+2*j:m+2*j+2] = O.T@np.cross(d,db).T
                V[3:6,3*m+2*j:3*m+2*j+2] = -np.cross(f,fb).T
                V[3:6,dim+6+j] = f
            out += weight*(V.T@G@V)
            flat = J.reshape(-1,J.shape[-1])
            out[h_fields,h_fields] += weight*(flat.T@(A@J).reshape(flat.shape))
            cross = weight*(flat.T@C.reshape(-1,9)@V)
            out[h_fields,:] += cross
            out[:,h_fields] += cross.T
        out = .5*(out+out.T)
        if np.linalg.eigvalsh(out)[0]<=0.:
            raise ArithmeticError('The native kinetic form is not positive')
        return reduced(out,dim),out[dim:,dim:]


def spectrum(K,M,indices):
    K,M = [q[np.ix_(indices,indices)] for q in (K,M)]
    s,V = eigh(K,M)
    r = V[:6]
    symmetric = (SIMPLE.T@PAIR.T@r).ravel()**2
    contrast = np.sum((CONTRAST.T@PAIR.T@r)**2,axis=0)
    symmetric /= symmetric.sum()
    contrast /= contrast.sum()
    i = int(np.argmax(np.where(s>0.,symmetric,-1.)))
    if s[i]<=0.:
        raise ArithmeticError('No positive native anchor')
    nu = 774.*np.sqrt(np.maximum(0.,s)/s[i])
    top = np.argsort(contrast)[-5:][::-1]
    return {'native_squared_rates':s,'signed_scaled_cm_inverse':np.sign(s)*np.sqrt(abs(s)/s[i])*774.,
            'symmetric_anchor_index':i,'symmetric_probe_fractions':symmetric,'contrast_probe_fractions':contrast,
            'top_five_contrast_indices':top,'top_five_contrast_cm_inverse':nu[top],
            'max_pencil_residual':float(np.max(abs(K@V-(M@V)*s))),
            'minimum_kinetic_eigenvalue':float(min(np.linalg.eigvalsh(M))),
            'negative_rate_count_below_minus_1e_minus_7':int(sum(s < -1e-7))}


def linear_returns(v):
    s = v['native_squared_rates']
    anchor = v['symmetric_anchor_index']
    if min(s)<=0.:
        return {'positive_subspace_required':True}
    omega = np.sqrt(s)
    native_period = 2.*np.pi/omega[anchor]
    q0 = 1e-4*np.sin(.71*np.arange(len(s))+.3)
    p0 = 1e-4*omega*np.cos(.59*np.arange(len(s))+.3)
    H0 = .5*(p0@p0+(s*q0)@q0)
    t = 8.*native_period
    ref = q0*np.cos(omega*t)+p0/omega*np.sin(omega*t)
    rows = []
    for resolution in (128,256,512):
        step = native_period/resolution
        if step*max(omega)>=1.:
            raise ArithmeticError('Integration step exceeds the linear stability window')
        def integrate(q,p):
            drift = 0.
            for _ in range(8*resolution):
                p -= .5*step*s*q
                q += step*p
                p -= .5*step*s*q
                drift = max(drift,abs(.5*(p@p+(s*q)@q)/H0-1.))
            return q,p,drift
        q,p,h = integrate(q0.copy(),p0.copy())
        err = float(np.linalg.norm(q-ref)/np.linalg.norm(ref))
        q,p,_ = integrate(q,-p)
        ret = float(np.linalg.norm(q-q0)/np.linalg.norm(q0))
        rows.append({'steps_per_symmetric_period':resolution,'dt_native':step,
                     'max_relative_energy_drift':h,'relative_displacement_error':err,'relative_reversed_position_error':ret})
    return {'duration_native':t,'duration_seconds_on_774_anchor':8./(774.*100.*C_SI),
            'seconds_per_native_time_on_774_anchor':omega[anchor]/(774.*100.*2.*np.pi*C_SI),
            'step_checks':rows}


def main():
    with np.load(STATE) as z:
        n,r,d,O,h = [z[k] for k in ('fields','radii_native','directions','frames','h_native')]
        intercept = float(z['contact_intercept_native'])
    with np.load(BASIS) as z:
        raw = z['fields_basis']
    model = NodalSF6(workers=6)
    try:
        model.mask = face_interior(48)
        model.model.geometry.d['r0'] = intercept
        q = np.array([projected(n,v,model.mask) for v in raw])
        Q,_ = np.linalg.qr(q.reshape(len(q),-1).T,mode='reduced')
        Y = Q.T.reshape(q.shape)
        T = Tangent(model,n,r,d,O,Y)
        x = np.zeros(T.dimension)
        E,g = T.evaluate(x)
        p = np.sin(.23*np.arange(T.dimension)+.11)
        p /= np.linalg.norm(p)
        changes = []
        for eps in (4e-5,2e-5):
            fd = float((T.evaluate(eps*p)[0]-T.evaluate(-eps*p)[0])/(2*eps))
            changes.append({'step':eps,'analytic':float(g@p),'difference':fd,'defect':abs(float(g@p)-fd)})
        if max(q['defect'] for q in changes)>2e-6:
            raise ArithmeticError('Joint gradient did not reproduce the potential')
        records = []
        for eps in (2e-4,1e-4):
            K = np.column_stack([(T.evaluate(eps*u)[1]-T.evaluate(-eps*u)[1])/(2*eps) for u in np.eye(T.dimension)])
            records.append(K)
            print('SF6 angular Hessian',eps,'asymmetry',float(np.max(abs(K-K.T))),flush=True)
        K = (records[1]+records[1].T)*.5
        step_change = float(np.max(abs(records[0]-records[1])))
        asym = float(np.max(abs(records[1]-records[1].T)))
        if step_change>3e-4 or asym>5e-6:
            raise ArithmeticError('Angular Hessian step or symmetry control failed')
        fine = FixedReader(n,64,3,None)
        rates = list(np.moveaxis(Y,0,-1))
        rates64 = interpolate_rates(fine,rates)
        laws = mass_laws()
        sizes = read(SIZES)
        calibrated = sizes['calibration_of_size_law']
        scale = sizes['common_length_scale_native_per_angstrom']
        unit = calibrated['common_mass_unit_kg']
        indexes = {'angular_and_field_mobile':np.arange(T.dimension),
                   'directions_held':np.r_[np.arange(6),np.arange(30,T.dimension)],
                   'fields_held':np.arange(30)}
        cases = {}
        beta = model.model.common['beta']
        for count,fields,hs,rate in ((48,n,h,rates),(64,fine.anchor,h*47/63,rates64)):
            support = np.array(sizes['matched_preparation_results'][f'SF6_N{count}']['topological_support_radii_native'])
            weights = (support/calibrated['reference_F_support_radius_native'])**calibrated['power']
            use = [(k,v[0],np.ones(len(fields))) for k,v in laws.items()]+[('size_power_R355',[],weights)]
            for name,terms,w in use:
                M,rigid = T.kinetic(fields,hs,rate,beta,terms,w)
                rows = {k:spectrum(K,M,v) for k,v in indexes.items()}
                cases[f'N{count}_{name}'] = {'reduced_metric':M,'spectra':rows}
                v = rows['angular_and_field_mobile']
                if name=='size_power_R355':
                    frozen = np.linalg.eigvalsh(reduced(reduced(rigid,6),3))
                    constants = H_SI/(8.*np.pi**2*100.*C_SI*unit*(1e-10/scale)**2*frozen)
                    cases[f'N{count}_{name}'].update({'co_transported_frame_rotation_constants_cm_inverse':constants,
                        'symplectic_linear_returns':linear_returns(v)})
                print('SF6',count,name,'contrast poles',v['top_five_contrast_cm_inverse'],
                      'negative',v['negative_rate_count_below_minus_1e_minus_7'],flush=True)
        files = (SOURCE,STATE,BASIS,OLD,SIZES,ROOT/'scripts'/'modes_faces_R354.py',ROOT/'scripts'/'relaxer_noeuds_R354.py',
                 ROOT/'scripts'/'noyau_courbes_R352.py',ROOT/'scripts'/'recaler_pose_R354.py')
        dump(OUTPUT/'VIBRATIONS_ANGLES_MOBILES_R355.json',{'revision':355,'full':True,'physical_validation':False,
            'coordinate_counts':{'radii':6,'bond_directions':12,'F_port_directions':12,'field_directions':3,
                                 'kinetic_elimination_global_SE3_and_bond_roll':12},
            'positive_rate_anchor_cm_inverse':774.,'reserved_Eg_cm_inverse':642.,
            'angular_tangents_and_frames_independent':True,'held_faces':True,'N64_is_only_kinetic_quadrature':True,
            'scope':'SF6 R354 static state, full mobile geometry and three transported field directions, zero global momenta',
            'energy_native':float(E),'joint_gradient_inf':float(np.max(abs(g))),'derivative_checks':changes,
            'centered_Hessian_step_max_difference':step_change,'Hessian_antisymmetry_max':asym,'native_Hessian':K,
            'cases':cases,'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in files}})
    finally:
        model.close()


if __name__=='__main__':
    main()
