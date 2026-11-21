'Fixed-coefficient collective gauge action and its quadratic operators.\n\nThe R453 embedding and bare collective potential are received hypotheses.\nSpatial derivative factors are transported with mesh spacing. At N=3 and\nL/L0=1 this agrees with R453. This does not turn the collective chart into\na microscopic derivation of Yang--Mills matter.'
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'entrees/R454'
sys.path.insert(0,str(PARENT))
from gauss_induced import atlas,fields,operator,flux,eliminate,MASSES,reduced
from gauge_su3 import loop,trace,shift,adj,exponential,random_algebra

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def label(n,length,weight=1.):return f'N{n}_L{length:g}'+(f'_G{weight:g}' if weight!=1. else '')

class Action:
    def __init__(self,n,length=1.,weight=1.):
        self.n=n;self.length=length;self.weight=weight;self.x=atlas(n)
        self.spacing_ratio=3*length/n;self.volume_ratio=length**3
    def energy_fields(self,c,u,p,base):
        f=sum(np.sum(3.-trace(loop(u,a,b)).real) for a,b in ((0,1),(0,2),(1,2)))
        matter=sum(np.linalg.norm(p-u[j]@shift(p,j))**2 for j in range(3))
        return float(self.volume_ratio*(base+(self.weight*f/self.spacing_ratio**4+.6*matter/self.spacing_ratio**2)/self.n**3))
    def potential(self,c):
        u,p,base=fields(self.x,c);return self.energy_fields(c,u,p,base)
    def gradient(self,c,h=1e-5):
        eye=np.eye(4);return np.array([(self.potential(c+h*v)-self.potential(c-h*v))/(2*h) for v in eye])
    def metric(self,c,eps=1e-5):
        u,p,base=fields(self.x,c);columns=[]
        for v in np.eye(4):
            plus,minus=[fields(self.x,c+sgn*eps*v) for sgn in [1,-1]]
            columns.append(flux(u,p,(plus[0]-minus[0])/(2*eps),(plus[1]-minus[1])/(2*eps)))
        w=np.column_stack(columns);D=operator(u,p)
        weights=np.r_[np.full(24*self.n**3,np.sqrt(self.weight)/self.spacing_ratio),np.ones(18*self.n**3)]
        Dw=D*weights[:,None];ww=w*weights[:,None]
        horizontal,A0,err=eliminate(Dw,ww)
        M=self.volume_ratio*(horizontal.T@horizontal/self.n**3+np.diag(MASSES))
        return (M+M.T)/2,dict(gauss_residual=err,temporal_potential_norm=float(np.linalg.norm(A0)))
    def hessian(self,c,h):
        eye=np.eye(4);H=np.empty((4,4))
        for i in range(4):
            for j in range(i,4):
                a=h*eye[i];b=h*eye[j]
                H[i,j]=H[j,i]=(self.potential(c+a+b)-self.potential(c+a-b)-self.potential(c-a+b)+self.potential(c-a-b))/(4*h*h)
        return H

CASES=[(n,length,weight) for weight in [1.,.8,1.2] for n,length in [(3,1.),(4,1.),(5,1.),(4,.8),(4,1.2)]]

def calculate(n,length,weight=1.):
    tag=label(n,length,weight);dest=ROOT/f'resultats/OPERATEUR_{tag}.json'
    if dest.exists():
        row=json.loads(dest.read_text())
        if 'gauge_weight' not in row:
            row['gauge_weight']=weight;save(dest,row)
        return row
    action=Action(n,length,weight)
    fit=minimize(action.potential,np.array([1.,1.,1.,.12]),jac=action.gradient,method='BFGS',options=dict(gtol=2e-9,maxiter=120))
    c=fit.x;g=action.gradient(c);assert max(abs(g))<1e-7
    M,checks=action.metric(c);Mfine,_=action.metric(c,5e-6)
    Hs=[action.hessian(c,h) for h in [4e-4,2e-4,1e-4]];H=Hs[-1]
    lam,modes=eigh(H,M);assert lam.min()>0 and np.linalg.eigvalsh(M)[0]>0
    herr=float(np.linalg.norm(Hs[-2]-H)/np.linalg.norm(H));merr=float(np.linalg.norm(Mfine-M)/np.linalg.norm(M))
    assert herr<2e-5 and merr<1e-6
    # Gauge covariance is checked at each physical size, at the same point.
    u,p,base=fields(action.x,c);G=exponential(random_algebra(np.random.default_rng(478+n),(n,n,n,3,3),.35))
    ug=np.array([G@link@adj(shift(G,k)) for k,link in enumerate(u)])
    gauge_error=abs(action.energy_fields(c,ug,G@p,base)-action.potential(c));assert gauge_error<2e-12
    parent_errors=None
    if n==3 and length==1. and weight==1.:
        V0,M0,_,_=reduced(action.x,c)
        parent_errors=dict(energy=abs(V0-action.potential(c)),metric=float(np.max(abs(M0-M))))
        assert max(parent_errors.values())<2e-12
    row=dict(revision=478,N=n,length_ratio=length,gauge_weight=weight,spacing_ratio=action.spacing_ratio,volume_ratio=action.volume_ratio,
        minimum=c.tolist(),energy=action.potential(c),gradient_inf=float(max(abs(g))),
        mass_min=float(np.linalg.eigvalsh(M)[0]),hessian_min=float(np.linalg.eigvalsh(H)[0]),
        frequencies=np.sqrt(lam).tolist(),hessian_step_error=herr,metric_step_error=merr,gauge_energy_error=gauge_error,
        parent_reproduction=parent_errors,**checks)
    np.savez_compressed(ROOT/f'resultats/OPERATEUR_{tag}.npz',c=c,M=M,H=H,H_coarse=Hs[-2],omega=np.sqrt(lam),modes=modes)
    save(dest,row);print(tag,'PASS',row['frequencies'],flush=True);return row

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--N',type=int);cli.add_argument('--length',type=float,default=1.);cfg=cli.parse_args()
    cases=CASES if cfg.N is None else [(cfg.N,cfg.length)]
    rows=[calculate(*case) for case in cases]
    if cfg.N is None:save(ROOT/'resultats/OPERATEURS_R478.json',dict(revision=478,cases=rows))
