'Inner prepared-state relaxation on a prescribed Gram, then an unheld test.\n\nThe nominal angles are calibration constraints. Four radii, relative pose,\nall 60 finite-field coordinates and four momentum orientations remain free.\nConstraint-normal gradients are exported and never counted as physical forces.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.spatial.transform import Rotation
from refit_dependencies import ROOT,State,UpdatedRouth,ReactiveContacts,install,split_primitives,assemble,force_components,fit,skew
from rotations_spatial import centers_and_jacobian
from verify_final import geometry_check
from mix import dump

def left_jacobian(v):
    t=np.linalg.norm(v);K=skew(v)
    if t<1e-4:a=.5-t*t/24+t**4/720;b=1/6-t*t/120+t**4/5040
    else:a=(1-np.cos(t))/t**2;b=(t-np.sin(t))/t**3
    return np.eye(3)+a*K+b*K@K

def direction_chart(model,D):
    dot=np.clip(np.sum(model.D0*D,axis=1),-1.,1.);s=np.linalg.norm(np.cross(model.D0,D),axis=1);t=np.arctan2(s,dot)
    if np.any((s<1e-10)&(dot<0)):raise ValueError('Carte antipodale inconnue')
    tangent=D-dot[:,None]*model.D0;w=tangent*np.divide(t,s,out=np.ones_like(t),where=s>1e-10)[:,None]
    return np.einsum('iac,ic->ia',model.tangents,w)

class Conditioned:
    def __init__(self,row,seed='fit',point=None):
        self.row=row;self.core=UpdatedRouth(row);self.ns=self.core.ns;self.N=self.core.state.m.N;self.nq=self.core.nq;self.seed=seed
        nominal=np.array(row['full_coordinates']);m=self.core.state.m;self.Dstar=m.state(nominal[self.ns:])['D'];self.target=self.Dstar@self.Dstar.T
        if point is None and seed=='libere_R340':point=json.loads((ROOT/'resultats/CONTINUATION_CONJOINTE_R340.json').read_text())['result']
        if point is None:base=nominal;mom=np.zeros(4)
        else:base=np.array(point['full_coordinates']);mom=np.array(point['momentum_chart_coordinates'])
        self.y0=np.r_[base[:self.ns],base[self.ns:].reshape(self.N,3)[:,0],np.zeros(3),mom]
        self.pose_slice=slice(self.ns+self.N,self.ns+self.N+3);self.mom_slice=slice(self.ns+self.N+3,self.ns+self.N+7)
        self.bounds=np.vstack([self.core.bounds[:self.ns],self.core.bounds[self.ns:self.nq:3],np.tile([-1.2,1.2],(3,1)),self.core.bounds[self.nq:]])
        self.scale=np.r_[np.full(50,.10),np.full(self.ns-50,.15),np.full(self.N,.12),np.full(3,.10),np.full(4,.10)]
        self.calls=0;self.last=None;self.last_full=None;self.started=time.time()
    def mapped(self,y):
        R=Rotation.from_rotvec(y[self.pose_slice]).as_matrix();D=self.Dstar@R.T;c=np.empty((self.N,3));c[:,0]=y[self.ns:self.ns+self.N];c[:,1:]=direction_chart(self.core.state.m,D)
        return np.r_[y[:self.ns],c.ravel(),y[self.mom_slice]]
    def evaluate(self,y):
        y=np.asarray(y);x=self.mapped(y);E,g=self.core.evaluate(x);s=self.core.state;X,J=centers_and_jacobian(s);Jl=left_jacobian(y[self.pose_slice]);gc=g[self.ns:self.nq].reshape(self.N,3);B=np.zeros((3*self.N,self.N+3));cart=[];condition=[]
        for i in range(self.N):
            cond=float(np.linalg.cond(J[i]));condition.append(cond)
            if cond>1e8:raise ValueError('Map of unskilled centres')
            cart.append(np.linalg.solve(J[i].T,gc[i]));B[3*i,i]=1.;B[3*i:3*i+3,self.N:]=np.linalg.solve(J[i],-skew(X[i+1])@Jl)
        gy=np.r_[g[:self.ns],gc[:,0],Jl.T@sum((np.cross(r,gr) for r,gr in zip(X[1:],cart,strict=True)),np.zeros(3)),g[self.nq:]]
        assert np.max(abs(B.T@gc.ravel()-gy[self.ns:self.ns+self.N+3]))<2e-8
        projection=B@np.linalg.solve(B.T@B,B.T@gc.ravel());normal=gc.ravel()-projection
        record={**self.core.last,'conditioned_coordinates':y.tolist(),'conditioned_gradient':gy.tolist(),'relative_pose_rotvec':y[self.pose_slice].tolist(),
            'prescribed_Gram_maximum_defect':float(np.max(abs(self.core.model.last['D']@self.core.model.last['D'].T-self.target))),
            'allowed_gradient_inf_native':float(max(abs(gy))),'allowed_field_gradient_inf_native':float(max(abs(gy[:self.ns]))),
            'allowed_radius_gradient_inf_native':float(max(abs(gy[self.ns:self.ns+self.N]))),'allowed_pose_gradient_inf_native':float(max(abs(gy[self.pose_slice]))),
            'constraint_normal_original_center_gradient':normal.tolist(),'constraint_normal_gradient_inf_native':float(max(abs(normal))),
            'constraint_normal_gradient_is_not_identified_physical_holding_source':True,'center_chart_condition_numbers':condition,
            'body_directions_held_up_to_relative_SO3_radii_free':True,'calibration_geometry_not_a_prediction':True}
        assert record['prescribed_Gram_maximum_defect']<2e-12
        self.calls+=1;self.last=record;self.last_full=x.copy();return E,gy
    def preconditioner(self,y):
        self.evaluate(y);H=np.eye(len(y));Jl=left_jacobian(y[self.pose_slice]);s=self.core.model.last
        for i,c in enumerate(self.core.model.reduction['contacts']):
            a=np.zeros(len(y));a[:self.ns]=-c['port_state_gradient'];a[self.ns+i]=s['radii'][i]
            a[self.pose_slice]=c['port_direction_gradient']@skew(s['D'][i])@Jl
            a*=self.scale;H+=c['kappa']*np.outer(a,a)
        return np.linalg.inv(H)
    def controls(self,y):
        E,g=self.evaluate(y);rows=[]
        for name,sl in [('fields',slice(0,self.ns)),('radii',slice(self.ns,self.ns+self.N)),('relative_pose',self.pose_slice),('momentum_orientation',self.mom_slice)]:
            v=np.zeros(len(y));v[sl]=np.sin(np.arange(len(v[sl]))*.31+.23);v/=np.linalg.norm(v);exact=float(g@v)
            for h in [2e-5,1e-5]:
                ep,_=self.evaluate(y+h*v);em,_=self.evaluate(y-h*v);fd=(ep-em)/(2*h);error=abs(fd-exact);assert error<8e-5*max(1.,abs(exact)),(name,h,exact,fd,error)
                rows.append({'coordinates':name,'step':h,'analytic':exact,'finite_difference':fd,'absolute_defect':error})
        self.evaluate(y);return rows
    def relax(self,maxiter=90,previous=None):
        if previous is None:
            y=self.y0.copy();inverse0=self.preconditioner(y);inverse=inverse0.copy();history=[];inherited=0
        else:
            y=np.array(previous['result']['conditioned_coordinates']);inverse=np.array(previous['inverse_BFGS']);inverse0=np.array(previous['initial_inverse_BFGS']);history=previous['history'].copy();inherited=len(history)
        E,g=self.evaluate(y);initial=self.last.copy() if previous is None else previous['initial'];z=y/self.scale;gz=g*self.scale;invalid=0;message='Budget achieved; conditional condition retained.'
        for it in range(maxiter):
            if max(abs(g))<1e-6 and self.last['Euler_torque_inf_native']<1e-7:message='Stationarity in only authorized variations, retained constraints.';break
            direction=-inverse@gz
            if direction@gz>=0:direction=-inverse0@gz;inverse=inverse0.copy()
            p=direction*self.scale;alpha=min(1.,.08/max(float(np.max(abs(self.core.state.Z@p[:50]))),1e-30),.06/max(float(max(abs(p[50:self.ns]))),1e-30),.08/max(float(max(abs(p[self.ns:]))),1e-30));accepted=False
            for attempt in range(24):
                trial=np.clip((z+alpha*direction)*self.scale,self.bounds[:,0],self.bounds[:,1]);zt=trial/self.scale
                try:and_field,gt=self.evaluate(trial)
                except ValueError:invalid+=1;alpha*=.5;continue
                step=zt-z
                if and_field<=E+1e-4*float(gz@step):accepted=True;break
                alpha*=.5
            if not accepted:message='Not acceptable not found; no physical domain eliminated.';break
            delta=gt*self.scale-gz;product=float(step@delta)
            if product>1e-12*np.linalg.norm(step)*np.linalg.norm(delta):
                rho=1/product;A=np.eye(len(y))-rho*np.outer(step,delta);inverse=A@inverse@A.T+rho*np.outer(step,step)
            z=zt;y=trial;E=and_field;g=gt;gz=gt*self.scale;history.append({'iteration':len(history)+1,**self.last})
            checkpoint={'revision':341,'seed':self.seed,'completed':False,'source_row':self.row,'initial':initial,'result':self.last,'history':history,
                'inverse_BFGS':inverse.tolist(),'initial_inverse_BFGS':inverse0.tolist(),'physical_status':'UNKNOWN','physical_region_excluded':False}
            dump(ROOT/'results'/f'{self.seed}_ETAT_CONDITIONNE_R341.json',checkpoint)
            print(self.seed,'inner',len(history),'E',round(E,8),'gfree',round(self.last['allowed_gradient_inf_native'],7),'gfield',round(self.last['allowed_field_gradient_inf_native'],7),'holding',round(self.last['constraint_normal_gradient_inf_native'],7),'s',round(time.time()-self.started,1),flush=True)
        self.evaluate(y);result={'revision':341,'seed':self.seed,'completed':True,'source_row':self.row,'initial':initial,'result':self.last,'history':history,
            'inverse_BFGS':inverse.tolist(),'initial_inverse_BFGS':inverse0.tolist(),'iterations':len(history),'inherited_iterations':inherited,
            'invalid_numerical_trials':invalid,'message':message,'active_numeric_bounds':[i for i,(v,(lo,hi)) in enumerate(zip(y,self.bounds,strict=True)) if min(v-lo,hi-v)<1e-7],
            'new_full_stationarity_or_physical_state_claimed':False,'physical_status':'UNKNOWN','physical_region_excluded':False,
            'native_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        dump(ROOT/'results'/f'{self.seed}_ETAT_CONDITIONNE_R341.json',result);return result

def refit(point,name,betas):
    # Recompute both native kinetic components at the conditioned fields.
    state=State('SF4');state.refresh(np.array(point['full_coordinates']))
    raw2,d2,raw4,d4=split_primitives(state,name);model=ReactiveContacts('SF4','current','mobile');frames=np.concatenate([np.eye(3)[None],model.frames]);rows=[]
    for beta in betas:
        install(model,beta);base,C,potential=force_components(model,state.x);raw=[a+beta*b for a,b in zip(raw2,raw4,strict=True)];der=[a+beta*b for a,b in zip(d2,d4,strict=True)]
        I,D,G,B,E=assemble(state,raw,der,frames);Ic=B.T@I@B;Dc=np.einsum('pi,apq,qj->aij',B,D,B);result=fit(Ic,Dc,base,C,model.nstate)
        row={'state':name,'alpha':1.,'beta':float(beta),'full_coordinates':state.x.tolist(),'potential':potential,'common_rotation_fit':result,
            'geometry_prescribed_during_refit':True,'physical_status':'UNKNOWN'};rows.append(row);tag=f'{name}_beta_{beta:.10g}'.replace('.','p')
        np.savez_compressed(ROOT/'results'/f'{tag}_OPERATEURS_R341.npz',I=Ic,D=Dc,base_gradient=base,coefficient_gradients=C,coordinates=state.x);row['operator_file']=f'{tag}_OPERATEURS_R341.npz'
        print(name,'refit',beta,'g',result['candidates'][0]['state_force_inf_native'],'gc',result['candidates'][0]['center_force_inf_native'],flush=True)
    record={'revision':341,'rows':rows,'completed':True,'holding_reactions_not_identified_as_sources':True,'physical_status':'UNKNOWN','physical_region_excluded':False}
    dump(ROOT/'results'/f'{name}_RECALAGE_R341.json',record);return record

def lift(row,iterations=20):
    m=UpdatedRouth(row);m.branch='R341';m.protocol='levee_constraints';checks=m.controls(m.x0);result=m.relax(iterations)
    result.update({'revision':341,'source_row':row,'initial_full_gradient_controls':checks,'geometry_constraints_removed':True,
        'no_angle_penalty_or_new_force_added':True,'no_temporal_stability_or_physical_BST_validation_claimed':True})
    dump(ROOT/'resultats/LEVEE_CONTRAINTES_R341.json',result);last=np.r_[result['result']['full_coordinates'],result['result']['momentum_chart_coordinates']];end=m.controls(last)
    dump(ROOT/'resultats/CONTROLES_LEVEE_R341.json',{'revision':341,'checks':end,'replayed_final':m.last,'physical_status':'UNKNOWN'});return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',choices=['fit','libere_R340'],default='fit');p.add_argument('--maxiter',type=int,default=90);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    row=json.loads((ROOT/'resultats/AFFINEMENT_DEPENDANCES_R340.json').read_text())['best_row'];m=Conditioned(row,a.seed);checks=m.controls(m.y0)
    dump(ROOT/'results'/f'{a.seed}_CONTROLES_CONDITIONNEMENT_R341.json',{'revision':341,'checks':checks,'free_coordinates':len(m.y0),'geometry_constraint_independent_rank':5})
    print(a.seed,'conditioned whole-gradient controls passed',len(checks),flush=True)
    if not a.check_only:
        result=m.relax(a.maxiter);end=m.controls(np.array(result['result']['conditioned_coordinates']));dump(ROOT/'results'/f'{a.seed}_CONTROLES_FIN_CONDITIONNE_R341.json',{'revision':341,'checks':end,'replayed_final':m.last,'physical_status':'UNKNOWN'})
