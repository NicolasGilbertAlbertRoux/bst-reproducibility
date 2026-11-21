'Conditional state-dependent extension of the received native curvature law.\n\nOne contact energy, complete state/direction derivatives, existing native action.\nNo experimental angle target. Physical current, frame preparation and clocks open.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.optimize import minimize,brentq
from scipy.spatial.transform import Rotation
from kinetic_preparations import ROOT,State
from angles_internal import move,pullback
from mix import dump

class RecordingAction:
    def __init__(self,action):self.action=action;self.hash=action.hash;self.record=False;self.rows=[]
    def __call__(self,n,h):
        E,g=self.action(n,h)
        if self.record:self.rows.append((E,g))
        return E,g

def align(a,b):
    'Declared zero-roll convention; not a physically selected frame.'
    a=np.asarray(a)/np.linalg.norm(a);b=np.asarray(b)/np.linalg.norm(b);axis=np.cross(a,b);s=np.linalg.norm(axis);c=float(a@b)
    if s<1e-12:
        if c>0:return np.eye(3)
        axis=np.cross(a,np.eye(3)[np.argmin(abs(a))]);axis/=np.linalg.norm(axis);return Rotation.from_rotvec(np.pi*axis).as_matrix()
    return Rotation.from_rotvec(np.arctan2(s,c)*axis/s).as_matrix()

class ReactiveContacts:
    def __init__(self,label,law='current',F_reader='held',rolls=None,stiffness_factor=1.,epsilon_factor=1.):
        self.state=State(label);self.m=self.state.m;self.nstate=self.state.system.nstate;self.label=label;self.law=law;self.F_reader=F_reader
        if law not in ['fixed','reference','current'] or F_reader not in ['held','mobile']:raise ValueError('Variante inconnue')
        self.action=RecordingAction(self.m.trial.action);self.m.trial.action=self.action
        panel=json.loads((ROOT/'sources/PANEL_ET_BLOCS_R207.json').read_text())['panel'];case=next(r for r in panel if r['label']==label)
        self.eps=float(case['generator_step'])*epsilon_factor
        self.pair_curvature=float(json.loads((ROOT/'sources/R321/entrees/ENTREES_CONTACTS_R319.json').read_text())['well_contract']['fit']['curvature_separation'])
        self.factor=float(stiffness_factor);self.rolls=np.zeros(self.m.N) if rolls is None else np.asarray(rolls,float)
        if self.rolls.shape!=(self.m.N,) or self.factor<=0:raise ValueError('Formula or invalid factor preparation')
        self.frames=np.array([Rotation.from_rotvec(self.rolls[i]*(-d)).as_matrix()@align(f,-d)
            for i,(f,d) in enumerate(zip(self.m.d['directions_f'],self.m.D0,strict=True))])
        self.reference_energies=[];self.reference_gradients=[]
        for i,n in enumerate(self.m.trial.fields):
            E,g=self.action(n,self.h(i));self.reference_energies.append(E);self.reference_gradients.append(g)
        self.calls=0;self.last=None
    def h(self,i):return float(self.m.d[('S' if i==0 else 'F')+'_h'])
    def reference(self,i):return self.m.d[('S' if i==0 else 'F')+'_b_reference']
    def port(self,field,i,direction):
        h=self.h(i);b=self.reference(i);p=np.maximum(self.m.unit@direction,0.);p4=p**4;density=h**3*np.sum(b*b,axis=1);normal=float(density@p**8)
        if normal<=0:raise ValueError('Port not defined')
        key='S' if i==0 else 'F';nodes=self.m.d[key+'_node_offset']+h**3*np.einsum('ij,ij->i',b,field);q=float(nodes@p4/normal)
        dd=4*self.m.unit.T@(nodes*p**3)/normal-q*(8*self.m.unit.T@(density*p**7))/normal
        return {'value':q,'gradient_field':h**3*b*(p4/normal)[:,None],'gradient_direction':dd,'p':p,'normal':normal}
    def curvature(self,field,i,p,baseline):
        'C_eps(n,P_n b p^4); include the moving tangent projection P_n.'
        b=self.reference(i);dot=np.einsum('ij,ij->i',field,b);projected=b-field*dot[:,None];p4=p**4;generator=projected*p4[:,None]
        value=-2*baseline[0];gn=-2*baseline[1].copy();gd=np.zeros(3);energies=[];maximum_turn=0.
        for sign in [1.,-1.]:
            e=sign*self.eps;w=e*generator;s=np.linalg.norm(w,axis=1);sinc=np.sinc(s/np.pi);c=np.empty_like(s);small=s<1e-3
            c[small]=-1/3+s[small]**2/30-s[small]**4/840;c[~small]=(s[~small]*np.cos(s[~small])-np.sin(s[~small]))/s[~small]**3
            moved=field*np.cos(s)[:,None]+w*sinc[:,None];E,g=self.action(moved,self.h(i));energies.append(E);value+=E
            maximum_turn=max(maximum_turn,float(max(s)))
            # E(n cos|w|+w sinc|w|), then w=e*p^4*(b-n(n.b)).
            pull_w=sinc[:,None]*g+w*(-sinc*np.einsum('ij,ij->i',field,g)+c*np.einsum('ij,ij->i',w,g))[:,None]
            pull_gen=e*pull_w
            gn+=np.cos(s)[:,None]*g-p4[:,None]*(b*np.einsum('ij,ij->i',field,pull_gen)[:,None]+dot[:,None]*pull_gen)
            gd+=4*self.m.unit.T@(p**3*np.einsum('ij,ij->i',pull_gen,projected))
        signed=value/self.eps**2
        if abs(signed)<1e-7:raise ValueError('Branch of unqualified absolute curvature in the vicinity of zero')
        branch=np.sign(signed)
        return {'signed':float(signed),'absolute':abs(float(signed)),'gradient_field':branch*gn/self.eps**2,
            'gradient_direction':branch*gd/self.eps**2,'maximum_virtual_turn':maximum_turn,'perturbed_energies':energies}
    def evaluate(self,x):
        x=np.asarray(x,float);self.action.rows=[];self.action.record=True
        try:self.state.refresh(x)
        finally:self.action.record=False
        own=list(self.action.rows);assert len(own)==len(self.state.fields);m=self.m;c=x[self.nstate:].reshape(m.N,3)
        r=m.rbase*np.exp(c[:,0]);w=np.einsum('ia,iac->ic',c[:,1:],m.tangents);D=move(m.D0,w)
        raw=[g.copy() for E,g in own];state_grad=np.zeros(self.state.system.size);gd=np.zeros_like(D);radial=np.zeros(m.N);rows=[];cached=[];contact_E=0.;well_E=0.
        if self.law=='reference':curvature_fields=m.trial.fields;curvature_own=list(zip(self.reference_energies,self.reference_gradients,strict=True))
        else:curvature_fields=self.state.fields;curvature_own=own
        for j,d in enumerate(D):
            i=j+1;df=m.d['directions_f'][j] if self.F_reader=='held' else -(self.frames[j].T@d)
            S=self.port(self.state.fields[0],0,d);F=self.port(self.state.fields[i],i,df)
            if self.law=='fixed':kap=self.factor*float(m.d['stiffness'][j]);CS=CF=None;dkd=np.zeros(3)
            else:
                CS=self.curvature(curvature_fields[0],0,S['p'],curvature_own[0]);CF=self.curvature(curvature_fields[i],i,F['p'],curvature_own[i])
                kap=self.factor*(CS['absolute']+CF['absolute']+self.pair_curvature)
                dkd=self.factor*CS['gradient_direction']
                if self.F_reader=='mobile':dkd-=self.factor*self.frames[j]@CF['gradient_direction']
            delta=float(r[j]-float(m.d['r0'])-S['value']-F['value']);rho=-kap*delta;W,Wd=m.trial.well.separation_energy_derivative(r[j])
            contact_E+=.5*kap*delta**2;well_E+=W;radial[j]=Wd+kap*delta
            raw[0]+=rho*S['gradient_field'];raw[i]+=rho*F['gradient_field']
            if self.law=='current':
                raw[0]+=.5*delta**2*self.factor*CS['gradient_field'];raw[i]+=.5*delta**2*self.factor*CF['gradient_field']
            gport=S['gradient_direction'].copy()
            if self.F_reader=='mobile':gport-=self.frames[j]@F['gradient_direction']
            gd[j]=rho*gport+.5*delta**2*dkd
            f_source=rho*F['gradient_direction']
            if self.law!='fixed':f_source+=.5*delta**2*self.factor*CF['gradient_direction']
            torque=np.cross(df,f_source)
            rows.append({'kappa':kap,'delta':delta,'radial_force':float(radial[j]),'pair_radial_derivative':float(Wd),
                'sulfur_port':S['value'],'F_port':F['value'],'signed_curvatures':None if CS is None else [CS['signed'],CF['signed']],
                'F_reader_local':df.tolist(),'F_frame_conjugate_torque_native':torque.tolist()})
            gq=np.zeros(self.nstate);gq[:52]=self.state.pull(0,S['gradient_field']);gq[52+2*j:54+2*j]=self.state.pull(i,F['gradient_field'])
            gk=np.zeros(self.nstate)
            if self.law=='current':
                gk[:52]=self.factor*self.state.pull(0,CS['gradient_field']);gk[52+2*j:54+2*j]=self.factor*self.state.pull(i,CF['gradient_field'])
            cached.append({'port_sum':S['value']+F['value'],'port_state_gradient':gq,'kappa_state_gradient':gk,'port_direction_gradient':gport,'kappa_direction_gradient':dkd,'kappa':kap})
        diff=D@D.T-m.target;gram=.5*m.weight*float(np.sum(diff*diff));gd+=2*m.weight*diff@D
        tangent=gd-D*np.sum(D*gd,axis=1)[:,None];gR=radial[:,None]*D+tangent/r[:,None];back=pullback(m.D0,w,r[:,None]*gR)
        gc=np.zeros_like(c);gc[:,0]=r*np.sum(gR*D,axis=1);gc[:,1:]=np.einsum('iac,ic->ia',m.tangents,back)
        state_grad[:self.nstate]=np.concatenate([self.state.pull(i,g) for i,g in enumerate(raw)]);state_grad[self.nstate:]=gc.ravel()
        energy=float(sum(E for E,g in own)+contact_E+well_E+gram);R=r[:,None]*D;self.calls+=1
        self.last={'R':R,'D':D,'radii':r,'contacts':rows,'contact_energy':contact_E,'well_energy':well_E,'Gram_energy':gram,
            'atomic_self_energies':[float(E) for E,g in own],'energy_total':energy,'gradient':state_grad.copy()}
        self.reduction={'contacts':cached,'own_state_gradient':np.concatenate([self.state.pull(i,g) for i,(E,g) in enumerate(own)]),
            'radii':r.copy(),'directions':D.copy(),'Gram_gradient_direction':2*m.weight*diff@D,'field_coordinates':x[:self.nstate].copy()}
        return energy,state_grad
    def result(self,x):
        E,g=self.evaluate(x);s=self.last;m=self.m
        return {'revision':338,'label':self.label,'law':self.law,'F_reader':self.F_reader,'stiffness_factor':self.factor,'generator_step':self.eps,
            'geometry':m.geometry(s),'energy_native':E,'state_gradient_inf_native':float(max(abs(g[:self.nstate]))),
            'center_gradient_inf_native':float(max(abs(g[self.nstate:]))),'state_gradient_native':g[:self.nstate].tolist(),
            'center_gradient_native':g[self.nstate:].tolist(),'contacts':s['contacts'],'full_coordinates':x.tolist(),
            'minimum_F_F_distance_native':min(float(np.linalg.norm(s['R'][i]-s['R'][j])) for i in range(m.N) for j in range(i)),
            'one_contact_energy_per_bond':True,'projection_of_reference_generator_on_current_field_is_conditional_extension':self.law=='current',
            'F_zero_roll_frame_convention_not_physical_selection':self.F_reader=='mobile','F_frame_rolls_rad':self.rolls.tolist(),
            'full_F_frame_equations_or_dynamics_solved':False,'physical_alignment_status':'UNKNOWN','physical_region_excluded':False}
    def check(self,x):
        E,g=self.evaluate(x);rows=[]
        for name,indices in [('S_shape',range(50)),('S_inside_outside',range(50,52)),('F_inside_outside',range(52,self.nstate)),('centers',range(self.nstate,len(x)))]:
            indices=list(indices);v=np.zeros(len(x));v[indices]=np.sin(np.arange(len(indices))*.31+.2);v/=np.linalg.norm(v);exact=float(g@v)
            for h in [2e-5,1e-5]:
                ep,_=self.evaluate(x+h*v);em,_=self.evaluate(x-h*v);fd=(ep-em)/(2*h);defect=abs(exact-fd)
                assert defect<8e-5*max(1.,abs(exact)),(name,h,exact,fd,defect)
                rows.append({'direction':name,'step':h,'analytic':exact,'finite_difference':fd,'absolute_defect':defect})
        self.evaluate(x);return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='SF4');p.add_argument('--law',choices=['fixed','reference','current'],default='current');p.add_argument('--F-reader',choices=['held','mobile'],default='held');p.add_argument('--check',action='store_true');a=p.parse_args()
    t=time.time();model=ReactiveContacts(a.label,a.law,a.F_reader);x=model.state.x.copy();checks=model.check(x) if a.check else [];point=model.result(x)
    result={'revision':338,'result':point,'analytic_checks':checks,'scope':'state/direction reactive native curvature, independent inside/outside preparations, held or declared co-oriented F reader protocol',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.time()-t,'no_physical_validation':True}
    dest=ROOT/'results'/f'{a.label}_{a.law}_{a.F_reader}_RAIDEUR_R338.json';dump(dest,result);print(json.dumps({'label':a.label,'law':a.law,'F_reader':a.F_reader,'state_gradient':point['state_gradient_inf_native'],'center_gradient':point['center_gradient_inf_native'],'kappas':[r['kappa'] for r in point['contacts']],'checks':len(checks),'seconds':result['seconds']}),flush=True)
if __name__=='__main__':main()
