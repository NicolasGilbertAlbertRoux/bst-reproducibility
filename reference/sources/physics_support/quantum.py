'One conditional quantization for composition and collective excitations.\n\nCanonical commutators, tensor composition, Born probabilities and ideal\ndisplaced parity readers are ASSUMPTIONS. They are not inferred from BST.\nAll state covariances and level spacings below come from the received action.'
import itertools
from scipy import sparse
from scipy.sparse.linalg import eigsh,expm_multiply
from scipy.linalg import eigh,expm,block_diag
from collective import *

def covariance(M,K):
    ev,modes=eigh(K,M);omega=np.sqrt(ev)
    X=(modes/(2*omega))@modes.T
    mv=M@modes;P=(mv*(omega/2))@mv.T
    return block_diag(X,P),omega

def inverse_root(A):
    v,u=np.linalg.eigh(A);return (u/np.sqrt(v))@u.T

def frozen_settings():
    z=np.load(ROOT/'resultats/OPERATEUR_N4_L1.npz');V,_=covariance(z['M'],z['H']);G=np.linalg.inv(V)[:4,:4]
    A=inverse_root(G[:2,:2]);B=inverse_root(G[2:,2:]);u,s,vh=np.linalg.svd(A@G[:2,2:]@B)
    da=np.r_[A@u[:,0],np.zeros(2),np.zeros(4)]*.1
    db=np.r_[np.zeros(2),B@vh[0],np.zeros(4)]*.1
    assert da@np.linalg.inv(V)@db>0
    return da,db,dict(reference='N4_L1',dimensionless_displacement_t=.1,
        rule='Directions selected once on the position accuracy of the reference Hamiltonian; same settings for the fifteen cases. No experimental Bell results used.',
        reference_normalized_cross_correlation=float(s[0]),a=da.tolist(),b=db.tolist())

def parity(V,d):
    return float(np.exp(-.5*d@np.linalg.solve(V,d)) / np.sqrt(np.linalg.det(2*V)))

PART_A=np.array([0,1,4,5]);PART_B=np.array([2,3,6,7])

def probabilities(V,da,db):
    VA=V[np.ix_(PART_A,PART_A)];VB=V[np.ix_(PART_B,PART_B)];rows=[]
    for x,y in itertools.product([0,1],repeat=2):
        ax=x*da;by=y*db;ea=parity(VA,ax[PART_A]);eb=parity(VB,by[PART_B]);eab=parity(V,ax+by)
        p=np.array([[(1+a*ea+b*eb+a*b*eab)/4 for b in [1,-1]] for a in [1,-1]])
        rows.append(dict(x=x,y=y,p=p.tolist(),EA=ea,EB=eb,EAB=eab))
    chsh=sum(r['EAB']*(-1 if r['x']==r['y']==1 else 1) for r in rows)
    ns=max(float(np.max(abs(np.array(rows[0]['p']).sum(0)-np.array(rows[2]['p']).sum(0)))),
           float(np.max(abs(np.array(rows[1]['p']).sum(0)-np.array(rows[3]['p']).sum(0)))),
           float(np.max(abs(np.array(rows[0]['p']).sum(1)-np.array(rows[1]['p']).sum(1)))),
           float(np.max(abs(np.array(rows[2]['p']).sum(1)-np.array(rows[3]['p']).sum(1)))))
    return dict(settings=rows,CHSH=float(chsh),no_signalling_defect=ns,
                minimum_probability=min(np.min(r['p']) for r in rows),normalization_defect=max(abs(np.sum(r['p'])-1) for r in rows))

def tensor(matrices):
    out=sparse.csr_matrix([[1.]])
    for m in matrices:out=sparse.kron(out,m,format='csr')
    return out

def fock_hamiltonian(M,K,size):
    inv=np.linalg.inv(M);alpha=np.sqrt(np.diag(K)/np.diag(inv));I=sparse.eye(size,format='csr')
    lower=sparse.diags(np.sqrt(np.arange(1,size)),1,shape=(size,size),format='csr');raise_=lower.T
    diag=sparse.diags(2*np.arange(size)+1,format='csr')
    hopping=lower@lower+raise_@raise_
    q=[(lower+raise_)/np.sqrt(2*a) for a in alpha]
    p=[-1j*np.sqrt(a/2)*(lower-raise_) for a in alpha]
    q2=[(diag+hopping)/(2*a) for a in alpha];p2=[a*(diag-hopping)/2 for a in alpha]
    H=sparse.csr_matrix((size**4,size**4),dtype=complex)
    for i in range(4):
        ops=[I]*4;ops[i]=.5*inv[i,i]*p2[i]+.5*K[i,i]*q2[i];H+=tensor(ops)
        for j in range(i+1,4):
            ops=[I]*4;ops[i]=p[i];ops[j]=p[j];H+=inv[i,j]*tensor(ops)
            ops=[I]*4;ops[i]=q[i];ops[j]=q[j];H+=K[i,j]*tensor(ops)
    assert abs(H.imag).max()<1e-12
    return H.real.tocsr(),q,p

def apply_local(vector,matrices,size):
    z=vector.reshape((size,)*4)
    for i,op in enumerate(matrices):
        if op is not None:z=np.moveaxis(np.tensordot(op,z,axes=(1,i)),0,i)
    return z.ravel()

def fock_parity_actions(psi,q,p,da,db,size):
    parity0=np.diag((-1.)**np.arange(size));out=[]
    for x,y in itertools.product([0,1],repeat=2):
        opsa=[];opsb=[]
        for i in range(4):
            if i<2:
                displacement=x*da;op=expm(2j*(displacement[4+i]*q[i].toarray()-displacement[i]*p[i].toarray()))@parity0
                opsa.append(op);opsb.append(None)
            else:
                displacement=y*db;op=expm(2j*(displacement[4+i]*q[i].toarray()-displacement[i]*p[i].toarray()))@parity0
                opsb.append(op);opsa.append(None)
        apsi=apply_local(psi,opsa,size);bpsi=apply_local(psi,opsb,size);abpsi=apply_local(apsi,opsb,size)
        expectations=[np.vdot(psi,z) for z in [apsi,bpsi,abpsi]]
        assert max(abs(z.imag) for z in expectations)<1e-10
        probs=np.array([[np.linalg.norm((psi+a*apsi+b*bpsi+a*b*abpsi)/4)**2 for b in [1,-1]] for a in [1,-1]])
        out.append(dict(x=x,y=y,p=probs.tolist(),EA=float(expectations[0].real),EB=float(expectations[1].real),EAB=float(expectations[2].real)))
    return out

def run(case,da,db):
    n,length,weight=case;tag=label(*case);path=ROOT/f'resultats/QUANTIQUE_{tag}.json'
    if path.exists():return json.loads(path.read_text())
    z=np.load(ROOT/f'resultats/OPERATEUR_{tag}.npz');M,K=z['M'],z['H'];V,omega=covariance(M,K)
    J=np.block([[np.zeros((4,4)),np.eye(4)],[-np.eye(4),np.zeros((4,4))]])
    purity=float(np.max(abs(V@J@V-J/4)));assert purity<1e-11
    predicted=probabilities(V,da,db)
    assert predicted['minimum_probability']>-1e-12 and predicted['normalization_defect']<1e-12 and predicted['no_signalling_defect']<1e-12
    # The reference settings are NOT reoptimized when a case is tested.
    exact=sorted((np.array(occupation)+.5)@omega for occupation in itertools.product(range(5),repeat=4))[:12]
    previous=None;truncations=[];final=None
    for size in [7,9]:
        H,q,p=fock_hamiltonian(M,K,size);start=np.linspace(.3,1.,H.shape[0]);start/=np.linalg.norm(start)
        ev,vectors=eigsh(H,k=12,which='SA',tol=1e-12,v0=start);order=np.argsort(ev);ev=ev[order];vectors=vectors[:,order]
        error=float(np.max(abs(ev-exact)));residual=float(np.max(np.linalg.norm(H@vectors-vectors*ev,axis=0)))
        assert error<2e-6 and residual<2e-8
        psi=vectors[:,0];probs=fock_parity_actions(psi,q,p,da,db,size)
        probability_error=max(float(np.max(abs(np.array(a['p'])-b['p']))) for a,b in zip(probs,predicted['settings'],strict=True))
        assert probability_error<2e-7
        truncations.append(dict(local_basis_size=size,total_dimension=H.shape[0],eigenvalues=ev.tolist(),
            exact_level_error=error,eigen_residual=residual,probability_error=probability_error,
            step_difference=None if previous is None else float(np.max(abs(ev-previous)))))
        previous=ev;final=(H,psi,probs)
    # Imaginary-time projection is a formal preparation algorithm. No
    # physical cooling apparatus or unforced BST preparation is claimed.
    H,psi,probs=final;tau=8/min(omega);initial=np.zeros(H.shape[0]);initial[0]=1
    projected=expm_multiply(-(H-exact[0]*sparse.eye(H.shape[0],format='csr'))*tau,initial)
    projected/=np.linalg.norm(projected);fidelity=float(abs(np.vdot(psi,projected))**2);assert fidelity>1-1e-8
    # Same full action, several finite probes of the harmonic approximation.
    action=Action(n,length,weight);c=z['c'];Vc=action.potential(c);g=action.gradient(c)
    sigma=np.linalg.cholesky(V[:4,:4]);directions=[*np.eye(4),np.ones(4)/2,np.array([1,-1,1,-1])/2]
    probes=[]
    for hbar in [1e-4,1e-3]:
        for direction in directions:
            for sign in [-1,1]:
                d=sign*3*np.sqrt(hbar)*sigma@direction;dv=action.potential(c+d)-Vc;quad=float(.5*d@K@d)
                probes.append(dict(hbar=hbar,displacement=d.tolist(),relative_potential_error=abs(dv-g@d-quad)/quad))
    worst=max(p['relative_potential_error'] for p in probes)
    scales=[]
    for hbar in [1e-4,1e-3]:
        # Normalize by hbar in the parity formula; it is otherwise the same
        # covariance/reader law, proving homogeneity throughout hbar>0.
        test=probabilities((hbar*V)/hbar,(np.sqrt(hbar)*da)/np.sqrt(hbar),(np.sqrt(hbar)*db)/np.sqrt(hbar))
        scales.append(dict(hbar=hbar,CHSH=test['CHSH'],max_probability_difference=max(float(np.max(abs(np.array(a['p'])-b['p']))) for a,b in zip(test['settings'],predicted['settings'],strict=True))))
    # Removing every cross term between the declared two parties yields
    # a product ground state under the same quantization and reader.
    Mb=M.copy();Kb=K.copy()
    for i in [0,1]:
        for j in [2,3]:Mb[i,j]=Mb[j,i]=Kb[i,j]=Kb[j,i]=0.
    control,_=covariance(Mb,Kb);product=probabilities(control,da,db);assert product['CHSH']<=2+1e-12
    result=dict(revision=478,N=n,length_ratio=length,gauge_weight=weight,
        hbar_interval=[1e-4,1e-3],hbar_scaling_exact_in_quadratic_model=True,
        frequencies=omega.tolist(),purity_symplectic_defect=purity,gaussian_prediction=predicted,
        fock_projection_probabilities=probs,truncations=truncations,formal_preparation=dict(method='imaginary_time_projection',tau=tau,fidelity=fidelity,physical_preparation_derived=False),
        nonlinear_potential_probes=probes,maximum_probe_relative_error=worst,hbar_endpoint_checks=scales,
        uncoupled_control_CHSH=product['CHSH'],reader_frozen_from='N4_L1',
        assumptions_not_derived=['canonical_commutators','Born_rule','ideal_parity_readout','collective_subsystem_partition','quadratic_action_regime'],
        physical_validation=False,hadron_spectrum_claim=False,experimental_Bell_claim=False)
    save(path,result);np.savez_compressed(ROOT/f'resultats/QUANTIQUE_{tag}.npz',M=M,K=K,covariance_hbar1=V,displacement_A=da,displacement_B=db,frequencies=omega)
    print(tag,'S=',predicted['CHSH'],'levels=',truncations[-1]['exact_level_error'],'nonlinear=',worst,flush=True)
    return result

if __name__=='__main__':
    da,db,settings=frozen_settings();save(ROOT/'resultats/LECTEUR_GELE_R478.json',settings)
    rows=[run(case,da,db) for case in CASES]
    save(ROOT/'resultats/QUANTIFICATION_COMMUNE_R478.json',dict(revision=478,settings=settings,cases=rows))
