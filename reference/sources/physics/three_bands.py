'Confront the same reading weights with the fundamentals and the weak band, on the same PTB spectrum.'
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
from crosscheck_spectrum import contact_readers,geometries,slice_at,WINDOWS,CORE,CONV
from readout_ir import decode,I,O,save
import json,csv
def internal_directions():
    state=decode('ETAT_SF6_PARENT_R480.json');chart=state['chart'];x=np.array(state['coordinates'])
    with np.load(I/'OPERATEURS_SF6_PARENT_R480.npz') as z:modes=z['modes']
    basis=[]
    for site,label in enumerate(('s','f')):
        start=24+12*site
        d0=np.array(chart[f'mesh_{label}_base']);frames=np.array(chart[f'mesh_{label}_frame'])
        raw=d0+np.einsum('nij,ni->nj',frames,x[start:start+12].reshape(6,2))
        norm=np.linalg.norm(raw,axis=1);d=raw/norm[:,None]
        dd=(frames-d[:,None,:]*np.einsum('nij,nj->ni',frames,d)[:,:,None])/norm[:,None,None]
        for shell in (1,0,None):
            vec=np.zeros((3,66))
            if shell is not None:
                ids=np.arange(6)*2+12*site+shell
                vec[:,ids]=d.T
                amp=x[ids]
            else:amp=np.ones(6)
            vec[:,start:start+12]=(dd*amp[:,None,None]).reshape(12,3).T
            basis.append(vec@modes)
    return np.array(basis)[[0,3,1,4,2,5]]
def main():
    spectral=json.loads((O/'SPECTRE_ET_THERMOCHIMIE_R573.json').read_text()); w,B=contact_readers()
    rv=spectral['measurements'];refs=np.array([[v[b][k]['area'] for b,k in spectral['keys_aires']] for v in rv.values()])
    template=list(csv.DictReader((I/'SPECTRES_SF6_R570.csv').open(encoding='utf-8-sig')))
    means=decode('CALIBRATION_SF6_R570.json')['externe'];window=(1100.,1160.)
    results={}
    for scenario,group in spectral['families'].items():
        win=group['best_par_n_of_coefficients']['4'];gain=win['gain'];freq=gain*w
        first=(freq>=window[0])&(freq<=window[1])
        norms=np.einsum('a,aim->im',win['coefficients'],B);native=np.sum(norms[:,first]**2)
        # Require linear silence of each mode in the combination band already recorded.
        # Projection into space of the coefficients, without elimination of one mode or coefficient per line.
        base=B
        attempts=[]
        directions=internal_directions()
        for count in (0,2,4,6):
            extended=np.concatenate([base,directions[:count]])
            nearly=extended[:,:,first].reshape(len(extended),-1).T
            null=null_space(nearly);of=np.einsum('ai,ajm->ijm',null,extended)
            masks=[]
            for b,k in spectral['keys_aires']:
                l,h=(WINDOWS if k=='region' else CORE)[b];masks.append((freq>=l)&(freq<=h))
            cov=np.einsum('aim,bim->mab',of,of);mats=np.einsum('rm,mab->rab',np.array(masks,float),cov)
            sc=np.sqrt(np.maximum(np.diagonal(mats[0]+mats[2]),1e-30));mats=mats/sc[None,None,:]/sc[None,:,None]
            arr=(mats[None,...]/refs[:,:,None,None]).reshape(-1,len(sc),len(sc))
            def fun(x):return np.einsum('i,rij,j->r',x,arr,x)-1
            def jac(x):return 2*arr@x
            candidates=[]
            for seed in range(25):
                q=least_squares(fun,np.random.default_rng(seed).normal(size=len(sc))*15,jac=jac,max_nfev=2000)
                c0=null@(q.x/sc);sig=np.sum(np.einsum('a,aim->im',c0,extended)**2,axis=0)
                r=(np.array(masks)@sig)/refs; fac=2/(r.min()+r.max());r*=fac
                candidates.append((max(abs(r.ravel()-1)),c0*np.sqrt(fac),sig*fac))
            err,c0,sig=min(candidates,key=lambda z:z[0])
            attempts.append(dict(count_coefficients=len(extended),deviation_huit_aires_pct=float(err)*100))
            if err<.09999:break
        shape={}
        for p in ['225.53','745.66']:
            t=[r for r in template if r['pressure_Torr']==p];x=np.array([float(r['nombre_onde_cm-1']) for r in t]);bg=np.array([float(r['fond_G']) for r in t]);y=np.array([float(r['section_PTB_cm2_par_molecule']) for r in t]);u=np.array([float(r['U_k2']) for r in t]);ys=np.array([float(r['absorption_G']) for r in t])
            # Circuit of R570 added to a residual line (ero length with rounding noise).
            width=.12;extra=np.zeros_like(x)
            for z,amp in zip(freq[first],sig[first]):extra+=amp*1.e-19*np.exp(-.5*((x-z)/width)**2)/(np.sqrt(2*np.pi)*width)
            er=(bg+ys+extra-y)/u;shape[p]={'count':len(x),'difference_max_sur_U':max(abs(er)),'area_physics':np.trapezoid(ys+extra,x),'points_in_U':int(sum(abs(er)<=1.))}
        results[scenario]=dict(gain=gain,indices_linear_in_window=(np.flatnonzero(first)+1).tolist(),area_spurious_first_readout_cm_mol=native*1e-19,
              ratio_first_readout_over_PTB=native*1e-19/means['PTB_value_combinee'][0],
              coefficients_common=c0.tolist(),trials_with_grids=attempts,dimension_space_silent=null.shape[1],force_linear_window_cm_mol=sum(sig[first])*1e-19,
              deviation_huit_aires_pct=100*err,aires_strong=(np.array(masks)@sig).tolist(),
              centers_strong=[float(sig[m]@freq[m]/sum(sig[m])) for m in (masks[0],masks[2])],profile_published=shape,
              conditions='Linear reader of the arms and of the mesh directions S and F calibrated on the two strong bands under the constraint of silence in the weak band. The continuous combination contribution comes from the R570 calibrated reading and retains its instrumental funds and unity. The calculation does not derive its 139 knots from the vibratory action alone.')
    save('TROIS_BANDES_ET_CONTROLE_R573.json',dict(results_field=results,method='The same vector moment coefficients are used for the three bands at each pressure, with the additional combination profile R570. Completed research on the two published witnesses; a failure of this family does not exclude another assignment.'))
    print([(k,v['rapport_premier_readout_sur_PTB'],v['deviation_huit_aires_pct'],[(p,a['points_in_U']) for p,a in v['profile_published'].items()]) for k,v in results.items()])
if __name__=='__main__':main()
