'Testing a common vectorial reading of contacts, hearts and coats in the measured spectra.'
import itertools
import numpy as np
from scipy.optimize import least_squares,linear_sum_assignment
from scipy.constants import Avogadro, Planck, c
from readout_ir import decode,observed,operators,save,I,WINDOWS,CORE
CONV=4184/(Planck*c*100*Avogadro)
def contact_readers():
    s=decode('ETAT_SF6_PARENT_R480.json');ch=s['chart'];x=np.array(s['coordinates'])
    d=np.array(s['directions']);bs=np.array(ch['contact_frame'])
    v0=np.array(ch['contact_base'])+np.einsum('nij,ni->nj',bs,x[48:60].reshape(6,2))
    dd=(bs-d[:,None,:]*np.einsum('nij,nj->ni',bs,d)[:,:,None])/np.linalg.norm(v0,axis=1)[:,None,None]
    B=[]
    def vector(z,site,shell):
        ds=np.array(ch['contact_base'])+np.einsum('nij,ni->nj',bs,z[48:60].reshape(6,2))
        return (ds/np.linalg.norm(ds,axis=1)[:,None]).T@z[:24].reshape(2,6,2)[site,:,shell]
    with np.load(I/'OPERATEURS_SF6_PARENT_R480.npz') as f:
        modes=f['modes']
        for site,shell in itertools.product(range(2),repeat=2):
            jac=np.zeros((3,66));indices=np.arange(6)*2+12*site+shell
            jac[:,indices]=d.T
            jac[:,48:60]=(x[indices,None,None]*dd).reshape(12,3).T
            z=np.cos(np.arange(66));eps=2e-6
            assert max(abs((vector(x+eps*z,site,shell)-vector(x-eps*z,site,shell))/(2*eps)-jac@z))<1e-8
            B.append(jac@modes)
    w,geom,_=operators()
    return w,np.concatenate([geom,np.array(B)])
def geometries():
    raw=decode('EXTENSION_DES_FAMILLES_R572.json')['polygones_par_scenario']
    return {name:[r for r in raw[key] if 16<=r['k_over_k0']<=32] for name,key in [('two_preparations','moderne_plus_cycle_measurement'),('1995_branche4','moderne_cycle_and_calculation_1995__branch_4')]}
def slice_at(rec,gain):
    verts=np.array(rec['vertices_C_et_C_hbar'])
    y=gain/CONV;xs=[]
    for a,b in zip(verts,np.roll(verts,-1,axis=0)):
        if min(a[1],b[1])-1e-12<=y<=max(a[1],b[1])+1e-12 and abs(a[1]-b[1])>1e-14:
            xs.append(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
    if not xs:return None
    return [min(xs),max(xs)]
def best():
    w,B=contact_readers();raw=observed()
    # Two wide regions and two inner windows; anchorages at both pressures.
    keys=[(b,reg) for b in WINDOWS for reg in ('region','core')]
    edges={ 'region':WINDOWS,'core':CORE}
    reference=np.array([[v[b][k]['area'] for b,k in keys] for v in raw.values()])
    centers=np.array([[v[b]['region']['center'] for b in WINDOWS] for v in raw.values()])
    cov=np.einsum('aim,bim->mab',B,B)
    records={}
    rt=decode('RACCORD_ENERGIE_ET_FREQUENCES_R572.json')
    target=np.array([ref for ref,rep in rt['comparateurs_NIST_cm1_et_multiplicites'] for _ in range(int(rep))])
    # Point inside each cell where no mode changes windows.
    bins=np.unique([q/v for e in edges.values() for b in e.values() for q in b for v in w])
    for name,polys in geometries().items():
        ranges=[(CONV*min(x[1] for x in r['vertices_C_et_C_hbar']),CONV*max(x[1] for x in r['vertices_C_et_C_hbar'])) for r in polys]
        bp=np.unique(np.r_[bins, np.ravel(ranges)])
        bank={}
        for a,z in zip(bp[:-1],bp[1:]):
            gain=(a+z)/2
            poly=next((i for i,(lo,hi) in enumerate(ranges) if lo<gain<hi),None)
            if poly is None:continue
            freq=gain*w;err=abs(freq[None,:]/target[:,None]-1)
            jj=linear_sum_assignment(np.where(err<=.10,err,1000.))[1]
            if max(err[np.arange(len(jj)),jj])>.1:continue
            masks=np.array([(freq>=edges[k][b][0])&(freq<=edges[k][b][1]) for b,k in keys])
            key=masks.tobytes()
            if key in bank:continue
            bank[key]=(gain,poly,jj,masks)
        winners={}
        for dim in (2,3,4,6):
            best_err=1e99; winner=None
            # a,b, mantle contrast F-S, heart contrast F-S; then four separate weights.
            if dim==6:L=np.eye(6)
            else:
                L=np.zeros((dim,6));L[:2,:2]=np.eye(2)
                if dim>=3:L[2,[3,5]]=[-1,1]
                if dim==4:L[3,[2,4]]=[-1,1]
            for g,idx,jj,masks in bank.values():
                freq=g*w; mats0=np.einsum('rm,mab->rab',masks.astype(float),cov)
                mats=L@mats0@L.T
                diag=np.sqrt(np.maximum(np.diagonal(mats[0]+mats[2]),1e-30));mats=mats/diag[None,:,None]/diag[None,None,:]
                norm=mats[None,...]/reference[:,:,None,None];norm=norm.reshape((-1,dim,dim))
                for seed in range(3):
                    rng=np.random.default_rng(seed+573);initial=rng.normal(size=dim)*13
                    def res(q):return np.einsum('i,rij,j->r',q,norm,q)-1.
                    def jac(q):return 2*(norm@q)
                    fit=least_squares(res,initial,jac=jac,max_nfev=140,ftol=2e-11,xtol=2e-11,gtol=1e-11)
                    coeff=L.T@(fit.x/diag);sig=np.einsum('a,mab,b->m',coeff,cov,coeff)
                    ratio=masks@sig/reference;factor=2/(ratio.min()+ratio.max())
                    coeff*=np.sqrt(factor);sig*=factor;er=float(np.max(abs(ratio*factor-1)))
                    if er>=best_err:continue
                    mass=masks@sig;mu=(masks[[0,2]]@(sig*freq))/mass[[0,2]]
                    pred=polys[idx];lo,hi=slice_at(pred,g);C0=(lo+hi)/2
                    y=g/CONV
                    verts=np.array(pred['vertices_C_et_C_hbar'])
                    winner=dict(gain=g,stiffness_over_k0=pred['k_over_k0'],C_kcal_mol=C0,hbar_native=y/C0,affectation_base1=(jj+1).tolist(),error_max_frequency_pct=100*max(abs(freq[jj]/target-1)),coefficients=coeff.tolist(),aires=mass.tolist(),centers=mu.tolist(),deviation_8_aires_pct=100*er,deviation_4_aires_larges_pct=100*np.max(abs(ratio[:,[0,2]]*factor-1)),deviation_4_centers_pct=100*np.max(abs(mu/centers-1)),frequencies=freq.tolist(),forces=sig.tolist(),polygon_C_and_C_hbar=verts.tolist(),interval_C_a_gain_fixed=[lo,hi],scenario=name)
                    best_err=er
            winners[str(dim)]=winner
        records[name]=dict(cells_distinct=len(bank),best_par_n_of_coefficients=winners)
    save('SPECTRE_ET_THERMOCHIMIE_R573.json',dict(measurements=raw,keys_aires=keys,unit_area='10^-19 cm/molecule',families=records,readout='Calibrated comparison operator. Each coefficient is common to the six arms and to the two pressures. The forces include the 66 modes. Energy and time in units of the SFx action. Areas include all instrumental contributions from windows. The rotational branch structures, the phases and the Raman polarization are not determined by these integrals. Finished digital research, without proof of optimality.'))
    for name,rec in records.items():
        print(name,rec['cells_distinct'],[(k,round(v['deviation_8_aires_pct'],3),round(v['deviation_4_centers_pct'],3)) for k,v in rec['best_par_n_of_coefficients'].items()])
if __name__=='__main__':best()
