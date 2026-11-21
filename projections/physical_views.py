"""Derived viewing records for neutral atoms, reduced thermal ensembles and sky beams."""
import base64
import json
import zlib
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.spatial import cKDTree
from scipy.special import gammaln, lpmv
from trajectories import carbon_hessian
from atlas_geometry import assemble

HERE = Path(__file__).resolve().parent
INPUTS = HERE / 'inputs'

def regular(x, digits=7):
    return np.round(x, digits).tolist()

def compressed(array):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(array).tobytes(),6)).decode('ascii')

def atom_models():
    models={}
    with np.load(INPUTS/'light-element-relaxations.npz') as records:
        for z,symbol in enumerate(records['symbols'].tolist(),1):
            key=f'Z{z:02d}_{symbol}_';p=records[key+'relaxed_position'].copy();p-=p.mean(axis=0)
            h=carbon_hessian(p);rigid=np.array([np.tile(e,(len(p),1)).ravel() for e in np.eye(3)]+[np.cross(e,p).ravel() for e in np.eye(3)]).T
            basis=null_space(rigid.T);lam,q=np.linalg.eigh(basis.T@h@basis);modes=basis@q
            radial=p.ravel();overlap=radial@modes;positive=lam>1e-7
            if not positive.any():raise ValueError(f'No positive internal mode for {symbol}')
            index=np.argmax(np.where(positive,np.abs(overlap),-1));v=modes[:,index]*np.sign(overlap[index]);v=v.reshape(-1,3)
            assert np.max(np.abs(v.mean(axis=0)))<1e-8
            radius=np.max(np.linalg.norm(p,axis=1));v/=np.max(np.linalg.norm(v,axis=1));p/=radius
            models[symbol]=dict(points=regular(p),u=regular(v),omega=round(float(np.sqrt(lam[index])),8),spacing=float(np.median(cKDTree(p).query(p,k=2)[0][:,1])),surface=records[key+'surface'].astype(int).tolist())
    return models

def thermal_ensemble(density, target, seed):
    # Identical mass and length units in a periodic force-shifted Lennard-Jones fluid.
    rng=np.random.default_rng(seed);cells=3;n=4*cells**3;length=(n/density)**(1/3)
    motif=np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
    grid=np.array(list(np.ndindex(cells,cells,cells)));pos=((grid[:,None,:]+motif+.12)*length/cells).reshape(n,3)
    vel=rng.normal(size=pos.shape);vel-=vel.mean(axis=0);vel*=np.sqrt(3*(n-1)*target/np.sum(vel**2));a,b=np.triu_indices(n,1)
    dt=.003;cut=min(2.5,.49*length);fc=24*(2/cut**13-1/cut**7);vc=4*(cut**-12-cut**-6)
    def forces(p):
        diff=p[a]-p[b];diff-=length*np.rint(diff/length);d2=(diff*diff).sum(axis=1);active=d2<cut**2
        delta=diff[active];radius=np.sqrt(d2[active]);inv=1/radius;u6=inv**6
        f=(24*(2*u6*u6-u6)*inv-fc)*inv;pair=delta*f[:,None];ia,ib=a[active],b[active]
        acc=np.column_stack([np.bincount(ia,weights=col,minlength=n)-np.bincount(ib,weights=col,minlength=n) for col in pair.T]);pot=np.sum(4*(u6*u6-u6)-vc+(radius-cut)*fc)
        return acc,pot,float(np.sum(f*d2[active]))
    acc,energy,virial=forces(pos);states=[];metrics=[];energies=[];origin=None;warm=550;stride=15
    for i in range(warm+72*stride):
        vel+=.5*dt*acc;pos+=dt*vel;acc,energy,virial=forces(pos);vel+=.5*dt*acc
        if i<warm and i%15==0:vel*=np.sqrt(3*(n-1)*target/np.sum(vel**2))
        if i>=warm and (i-warm)%stride==0:
            if origin is None:origin=pos.copy()
            kinetic=.5*np.sum(vel**2);energies.append(energy+kinetic)
            states.append(np.concatenate([(pos/length)%1,np.linalg.norm(vel,axis=1)[:,None]],axis=1))
            metrics.append([i*dt-warm*dt,2*kinetic/(3*(n-1)),np.mean(np.sum((pos-origin)**2,axis=1)),(2*kinetic+virial)/(3*length**3)])
    drift=np.ptp(energies)/n;assert drift<.004 and np.linalg.norm(vel.mean(axis=0))<1e-10
    return dict(n=n,length=length,density=density,target=target,seed=seed,energy_span_per_particle=drift,values=regular(metrics,6),raw=compressed(np.array(states,dtype='<f4')))

def sky_maps(sky):
    # Use only supported TT multipoles between measured band centres; omit all extrapolation.
    ell=np.arange(48,85);width,height=224,105;mu=np.sin(np.linspace(-np.pi/2,np.pi/2,height));rng=np.random.default_rng(29113)
    basis=np.zeros((len(ell),height,width),dtype='f4');beams=[0.,.5,1.,1.5,2.,2.5,3.]
    for k,l in enumerate(ell):
        coef=np.zeros((height,width//2+1),complex)
        for m in range(l+1):
            norm=np.sqrt((2*l+1)/(4*np.pi))*np.exp(.5*(gammaln(l-m+1)-gammaln(l+m+1)))
            rand=rng.normal() if m==0 else (rng.normal()+1j*rng.normal())/np.sqrt(2)
            coef[:,m]=norm*lpmv(m,l,mu)*rand
        basis[k]=np.fft.irfft(coef,n=width,axis=1)*width
    assert np.isfinite(basis).all();result={};area=np.cos(np.linspace(-np.pi/2,np.pi/2,height));area/=area.sum()
    for key,values in sky.items():
        rows=np.asarray(values['TT']);cl=np.interp(ell,rows[:,0],rows[:,3])*2*np.pi/(ell*(ell+1));assert cl.min()>0
        sigma=np.radians(beams)/np.sqrt(8*np.log(2));w=np.sqrt(cl)[None,:]*np.exp(-.5*sigma[:,None]**2*(ell*(ell+1)))
        maps=(w@basis.reshape(len(ell),-1)).reshape(len(beams),height,width);rms=np.sqrt(np.einsum('bxy,x->b',maps**2,area)/width);assert np.all(np.diff(rms)<0)
        result[key]=dict(map=compressed(np.rint(maps*32).astype('<i2')),rms=regular(rms,4))
    return dict(w=width,h=height,ell=[48,84],seed=29113,beams=beams,families=result)

def build(dataset, destination):
    data=dict(light=atom_models(),thermal={k:thermal_ensemble(r,t,s) for k,r,t,s in [('solid',1.03,.26,918),('liquid',.82,.90,919),('gas',.065,1.3,920)]},sky=sky_maps(dataset['sky']),material=json.loads((INPUTS/'substrate-grid.json').read_text()))
    data['atlas']=assemble(data['light'],dataset)
    Path(destination).write_text('window.BST_EXTENDED = '+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n');return data

if __name__=='__main__':
    import sys
    path=HERE.parent/'observatory/data.js';s=path.read_text();build(json.loads(s[s.index('{'):s.rindex('}')+1]),sys.argv[1])
