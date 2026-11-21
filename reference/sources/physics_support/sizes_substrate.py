'Transport the existing E2/E4 kinetic action through local substrate dilations.'

from pathlib import Path
import itertools

from sweep_sf4 import read
from sweep_beta import base_fields
from fields_and_references import dump
from dynamics_full import NodalKinetic
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from clock_inertia import OUTPUT as R350
from reader_transported import FixedReader
from precision_sum import AccurateSumAction
from refit_dependencies import State
from free_rotations import CALIBRATION, OUTPUT as R349, ROOT
from rotations_spatial import centers_and_jacobian
import numpy as np


OUTPUT = ROOT/'results'/'R351'
SOURCE = Path(__file__).resolve()
SHAPES = ('R349_free','R349_contacts_nuls','R350_precontraint')


def positive_power(M,power):
    eig,V = np.linalg.eigh((M+M.T)/2)
    if eig[0]<=0:
        raise ArithmeticError('Affine texture tensor is not positive')
    return (V*eig**power)@V.T


def nodes(count,h):
    return (np.moveaxis(np.indices((count,)*3,dtype=float),0,-1)-.5*(count-1))*h


def stencils(field,h,positions):
    n = round(len(field)**(1/3))
    f = field.reshape(n,n,n,4)
    for sign in itertools.product((1,-1),repeat=3):
        axes = tuple(a for a,s in enumerate(sign) if s<0)
        z = np.flip(f,axis=axes) if axes else f
        x = np.flip(positions,axis=axes) if axes else positions
        core = z[:-1,:-1,:-1]
        ds = []
        for s,nb in zip(sign,(z[1:,:-1,:-1],z[:-1,1:,:-1],z[:-1,:-1,1:]),strict=True):
            dot = np.sum(core*nb,axis=-1)
            if np.min(dot)<=0:
                raise ValueError('Texture leaves the gnomonic chart')
            ds.append((s/h)*(nb/dot[...,None]-core))
        d = np.stack(ds,axis=-2).reshape(-1,3,4)
        yield d,x[:-1,:-1,:-1].reshape(-1,3)


def cof_rows(M):
    return np.stack([np.cross(M[...,1,:],M[...,2,:]),np.cross(M[...,2,:],M[...,0,:]),
                     np.cross(M[...,0,:],M[...,1,:])],axis=-2)


def affine_energy(P,Q,A,beta):
    inverse = np.linalg.inv(A)
    det = np.linalg.det(A)
    return .5*det*np.trace(inverse@inverse.T@P)+.5*beta/det*np.trace(A.T@A@Q)


def optimal_affine(P,Q,beta):
    qh,qi = positive_power(Q,.5),positive_power(Q,-.5)
    C = qi@positive_power(qh@P@qh/beta,.5)@qi
    return positive_power(C/np.linalg.det(C),.5)


def transport_local(block,s):
    # The last three velocities are translations. They acquire a factor 1/s.
    d = np.r_[np.ones(3),np.full(3,1/s)]
    return block*d[:,None]*d[None,:]


def dilated_blocks(G2,G4,s):
    return np.array([transport_local(s_i**3*g2+s_i*g4,s_i)
                     for g2,g4,s_i in zip(G2,G4,s,strict=True)])


def kinetic_blocks(d,x,alpha,beta,volume):
    n = len(d)
    A = np.zeros((n,6,4))
    A[:,3:,:] = -d
    for a,e in enumerate(np.eye(3)):
        A[:,a,:] = -np.einsum('ni,nic->nc',np.cross(e,x),d)
    flat = A.transpose(1,0,2).reshape(6,-1)
    G2 = volume*flat@flat.T
    g = np.sum(d*d,axis=(1,2))
    square = (A*np.sqrt(g)[:,None,None]).transpose(1,0,2).reshape(6,-1)
    dot = np.einsum('nac,nic->nai',A,d).transpose(1,0,2).reshape(6,-1)
    G4 = volume*(square@square.T-dot@dot.T)
    return alpha*G2,beta*G4


def probe_profile(field,h,beta,energy2,kinetic,control):
    count = round(len(field)**(1/3))
    coords = nodes(count,h)
    p,q = np.zeros((3,3)),np.zeros((3,3))
    volume = h**3/8
    for d,_ in stencils(field,h,coords):
        g = d@d.swapaxes(1,2)
        p += volume*np.sum(g,axis=0)
        q += volume*np.sum(cof_rows(g),axis=0)
    if energy2 is None:
        raise ValueError('The inherited edge energy is required')
    E2,E4 = energy2
    E2c,E4c = .5*np.trace(p),.5*np.trace(q)
    if abs(E4-E4c)>1e-8:
        raise ArithmeticError('Corner quartic energy differs from the parent action')
    A = optimal_affine(p,q,beta)
    B = np.linalg.inv(A).T
    G2,G4 = np.zeros((6,6)),np.zeros((6,6))
    aff2,aff4 = np.zeros_like(G2),np.zeros_like(G4)
    for d,x in stencils(field,h,coords):
        if control:
            g2,g4 = kinetic_blocks(d,x,1.,beta,volume)
            G2 += g2
            G4 += g4
        dnew = np.einsum('ij,njc->nic',B,d)
        g2,g4 = kinetic_blocks(dnew,x@A.T,1.,beta,volume*np.linalg.det(A))
        aff2 += g2
        aff4 += g4
    if not control:
        G2,G4 = kinetic
    mass_defect = float(max(np.max(abs(G2-kinetic[0])),np.max(abs(G4-kinetic[1]))))
    energy = affine_energy(p,q,A,beta)
    mass = aff2+aff4
    virial = float(np.max(abs(mass[3:,3:]-np.eye(3)*energy)))
    grad = positive_power(q,.5)@p@positive_power(q,.5)
    exact_min = float(np.sqrt(beta)*np.trace(positive_power(grad,.5)))
    if mass_defect>2e-8 or virial>2e-8 or abs(energy-exact_min)>2e-8:
        raise ArithmeticError(f'Affine action or virial control failed: {mass_defect}, {virial}, {energy-exact_min}')
    scalar = float(np.sqrt(beta*E4/E2))
    return {'G2_native':G2,'beta_G4_native':G4,'E2_edges':E2,'E2_corners':E2c,'E4':E4,
            'Q_P':p,'Q_E4_cofactor':q,'scale_free_self_energy_stationary':scalar,
            'scale_shape_relaxed_affine_axes':np.linalg.eigvalsh(A),'A_shape_relaxed':A,
            'G2_affine_relaxed':aff2,'beta_G4_affine_relaxed':aff4,
            'self_energy_affine_relaxed':energy,'virial_abs_defect_corner_action':virial,
            'native_G_crosscheck_abs_defect':mass_defect}


def row_bank(fields,h,beta,full_check):
    action = AccurateSumAction(beta)
    n = round(len(fields[0])**(1/3))
    free,all_terms = NodalKinetic(0.),NodalKinetic(beta)
    rows = []
    for i,(field,step) in enumerate(zip(fields,h,strict=True)):
        g = np.zeros_like(field)
        out = np.zeros(3,dtype=np.longdouble)
        if action.f(n,step,1.,beta,field,g,out):
            raise ArithmeticError('The native edge action failed')
        G2 = free.mass(field,step)[2][3:,3:]
        G4 = all_terms.mass(field,step)[2][3:,3:]-G2
        bank = probe_profile(field,step,beta,(float(out[1]),float(out[2])),(G2,G4),full_check and i in (0,1))
        if full_check and i in (0,1):
            s = np.array([1.19])
            check = all_terms.mass(field,step*s[0])[2][3:,3:]
            bound = float(np.max(abs(check-dilated_blocks(G2[None],G4[None],s)[0])))
            if bound>2e-8:
                raise ArithmeticError('Local scaling disagrees with the native C kinetic stencil')
            bank['independent_dilation_scale'] = s[0]
            bank['independent_dilation_block_inf_defect'] = bound
        rows.append(bank)
    return rows


def make():
    candidate = read(CALIBRATION)['candidate']
    contact = read(R349/'REPONSE_CONTACTS_R349.json')
    prestress = read(R350/'CONTACT_PRECONTRAINT_R350.json')['prestressed_seed']['candidate']
    atlas = [r for r in read(R349/'BANQUE_ROTATION_R349.json')['rows'] if r['stage']=='anchor']
    prep = dict(zip(SHAPES,(candidate,contact['candidate'],prestress),strict=True))
    model = NodalDynamics(48)
    initial_fields = model.unpack(model.project(model.initial())[0])[0]
    rows = []
    for label,c in prep.items():
        fields = base_fields(model,c,initial_fields)
        chart = np.array(c['center_chart'])
        X,_,_,O,_,_ = model.geometry(chart)
        for count in (48,64):
            n = fields if count==48 else FixedReader(fields,count,3,model).anchor
            h = np.array(model.h)*47/(count-1)
            obs = row_bank(n,h,c['beta'],label==SHAPES[0])
            prior = (atlas if label==SHAPES[0] else contact['rows']) if label!=SHAPES[2] else None
            check = None
            if prior is not None:
                expected = np.array(next(r['native_blocks'] for r in prior if r['count']==count))
                check = float(np.max(abs(expected-np.array([r['G2_native']+r['beta_G4_native'] for r in obs]))))
                if check>2e-8:
                    raise ArithmeticError('R349 anchor metric did not replay')
            rows.append({'preparation':label,'species':'SF4','count':count,'h_native':h,
                         'length_scale_native_per_angstrom':c['scale_native_per_angstrom'],
                         'centers':X,'frames':O,'beta':c['beta'],'atoms':obs,
                         'native_metric_replay_inf_defect':check})
            print(label,count,'s*',[r['scale_free_self_energy_stationary'] for r in obs],
                  'native rest values',[r['E2_edges']+c['beta']*r['E4'] for r in obs],
                  'affine minimum',[r['self_energy_affine_relaxed'] for r in obs],flush=True)
    state6 = State('SF6')
    centres6,_ = centers_and_jacobian(state6)
    with np.load(ROOT/'results'/'SF6_TRANSFERT_CADRES_calage_R342.npz') as source:
        frames6 = source['frames']
    spacing6 = [float(state6.m.d[('S' if i==0 else 'F')+'_h']) for i in range(7)]
    obs6 = row_bank(np.array(state6.fields),spacing6,candidate['beta'],False)
    rows.append({'preparation':'SF6_R336_reserve','species':'SF6','count':48,
                 'h_native':spacing6,'centers':centres6,'frames':frames6,
                 'beta':candidate['beta'],'atoms':obs6})
    dependencies = (CALIBRATION,R349/'REPONSE_CONTACTS_R349.json',R349/'BANQUE_ROTATION_R349.json',
                    R350/'CONTACT_PRECONTRAINT_R350.json',ROOT/'results'/'SF6_TRANSFERT_CADRES_calage_R342.npz',
                    ROOT/'sources'/'cinetique_complete_R343.c',ROOT/'sources'/'action_parametree_R331.c',SOURCE)
    dump(OUTPUT/'BANQUE_DILATATIONS_R351.json',{'revision':351,'full':True,
         'protocol':'R220 size scaling and R235 affine corner action; local fields at fixed molecular centres',
         'physical_validation':False,'rows':rows,
         'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dependencies}})
    print('R351 size bank complete',flush=True)


if __name__=='__main__':
    make()
