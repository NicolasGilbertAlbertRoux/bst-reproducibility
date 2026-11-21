'Joint native states, not new fitted energies: S50 + S inner/outer + 2/F + centres.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from engine_local import NativeConsole,ROOT
from mix import dump,diagnostic

class JointStates:
    def __init__(self,label,weight,windows=None):
        self.engine=NativeConsole(windows=windows);self.m,self.Z=self.engine.model(label)
        self.label=label;self.m.weight=float(weight);self.N=self.m.N
        self.nstate=52+2*self.N;self.size=self.nstate+3*self.N
        w=self.engine.windows
        self.bounds=np.array([(-w['shape_abs'],w['shape_abs'])]*50+[(-w['preparation_abs'],w['preparation_abs'])]*(2+2*self.N)+self.m.bounds)
        self.limit=3.;self.calls=0;self.last=None
    def initial(self,recipe):
        q=np.array(recipe.get('preparation',self.m.baseq));F=np.array(recipe.get('individual_F',np.tile(q[2:],(self.N,1))))
        return np.r_[recipe['shape'],q[:2],F.ravel(),recipe.get('center_start',np.zeros(3*self.N))]
    def evaluate(self,x,details=False):
        x=np.asarray(x);m=self.m;d=m.d
        if x.shape!=(self.size,) or not np.isfinite(x).all():raise ValueError('Unfinished joint state')
        if np.any(x<self.bounds[:,0]) or np.any(x>self.bounds[:,1]):raise ValueError('No-key number declared to be declared')
        a=x[:50];qS=x[50:52];qF=x[52:self.nstate].reshape(self.N,2);c=x[self.nstate:]
        turn=self.Z@a
        if np.max(abs(turn))>self.limit:raise ValueError('Digital off-customer nodal rotation')
        fields=[];raw=[];energies=[];pulls=[];border=0.;normdefect=0.
        for i,(original,template) in enumerate(zip(m.trial.fields,m.trial.templates,strict=True)):
            symbol,b,norm,parts=template;coords=qS if i==0 else qF[i-1]
            amp=parts@coords;theta=norm*amp
            base=original*np.cos(theta)[:,None]+b*(amp*np.sinc(theta/np.pi))[:,None]
            base[~m.trial.mask]=original[~m.trial.mask]
            derivative=-original*(norm*np.sin(theta))[:,None]+b*np.cos(theta)[:,None]
            derivative[~m.trial.mask]=0
            if i==0:
                radius=np.linalg.norm(base[:,1:],axis=1);active=m.trial.mask&(radius>1e-12)
                unit=np.divide(base[:,1:],radius[:,None],out=np.zeros_like(base[:,1:]),where=radius[:,None]>1e-12)
                polar=np.column_stack([-radius,base[:,0,None]*unit]);polar[~active]=0
                effective=turn.copy();effective[~active]=0
                field=base*np.cos(effective)[:,None]+polar*np.sin(effective)[:,None]
                tangent=-base*np.sin(effective)[:,None]+polar*np.cos(effective)[:,None];tangent[~active]=0
                shape_pull=self.Z.copy();shape_pull[~active]=0
                # Differentiate the moving polar frame as well as the base field.
                dr=np.einsum('ij,ij->i',unit,derivative[:,1:])
                of=np.divide(derivative[:,1:]-unit*dr[:,None],radius[:,None],out=np.zeros_like(unit),where=radius[:,None]>1e-12)
                depolar=np.column_stack([-dr,derivative[:,0,None]*unit+base[:,0,None]*of]);depolar[~active]=0
                derivative=derivative*np.cos(effective)[:,None]+depolar*np.sin(effective)[:,None]
            else:field=base
            field[~m.trial.mask]=original[~m.trial.mask]
            e,g=m.trial.action(field,float(d[symbol+'_h']))
            fields.append(field);energies.append(e);raw.append(g);pulls.append((derivative,parts))
            border=max(border,float(np.max(abs(field[~m.trial.mask]-original[~m.trial.mask]))))
            normdefect=max(normdefect,float(np.max(abs(np.sum(field*field,axis=1)-np.sum(original*original,axis=1)))))
        m.fields=fields;m.self_energies=energies
        m.nodes=d['S_node_offset']+m.h3*np.einsum('ij,ij->i',d['S_b_reference'],fields[0])
        F=[]
        for i,field in enumerate(fields[1:],1):
            nodes=d['F_node_offset']+float(d['F_h'])**3*np.einsum('ij,ij->i',d['F_b_reference'],field)
            F.append(float((nodes@d[f'P_{i}']/d[f'normal_{i}'])[0]))
        m.F=np.array(F);state=m.state(c);gradient=np.zeros(self.size)
        for i,(g,(derivative,parts)) in enumerate(zip(raw,pulls,strict=True)):
            if i==0:
                g=g+m.h3*d['S_b_reference']*(state['weights']@(state['rho']/state['normal']))[:,None]
                gradient[:50]=shape_pull.T@np.einsum('ij,ij->i',g,tangent)
                gradient[50:52]=np.einsum('ij,ij->i',g,derivative)@parts
            else:
                g=g+float(d['F_h'])**3*d['F_b_reference']*(d[f'P_{i}']@(state['rho'][i-1:i]/d[f'normal_{i}']))[:,None]
                gradient[52+2*(i-1):52+2*i]=np.einsum('ij,ij->i',g,derivative)@parts
        gradient[self.nstate:]=state['gradient'];self.calls+=1
        self.last={'state':state,'norm_defect':normdefect,'border_defect':border,'turn':turn}
        if normdefect>3e-12 or border!=0:raise ValueError('Digital variant to be used')
        return state['energy_total'],gradient
    def result(self,x,optimizer=None):
        energy,g=self.evaluate(x);m=self.m;s=self.last['state'];R=s['R'];qF=x[52:self.nstate].reshape(self.N,2)
        distance=min(float(np.linalg.norm(R[i]-R[j])) for i in range(self.N) for j in range(i+1,self.N))
        gs=float(max(abs(g[:50])));gp=float(max(abs(g[50:self.nstate])));gc=float(max(abs(g[self.nstate:])))
        result={'revision':336,'status':'FINITE_NATIVE_CALCULATION','label':self.label,
            'geometry':m.geometry(s),'energy_total_native':energy,'existing_kangle':m.weight,
            'returned_shape':x[:50].tolist(),'preparation':np.r_[x[50:52],qF[0]].tolist(),
            'individual_F_preparations':qF.tolist(),'center_coordinates':x[self.nstate:].tolist(),
            'internal_shape_gradient_native':g[:50].tolist(),'maximum_internal_shape_gradient_native':gs,
            'S_inner_outer_conjugate_gradients_native':g[50:52].tolist(),'individual_F_conjugate_gradients_native':g[52:self.nstate].reshape(self.N,2).tolist(),
            'maximum_preparation_gradient_native':gp,'center_gradient_inf':gc,'center_stationarity_qualified':gc<1e-7,
            'stationarity_in_all_declared_state_and_center_coordinates_qualified':gs<1e-6 and gp<1e-6 and gc<1e-7,
            'minimum_F_F_distance_native':distance,'field_norm_defect':self.last['norm_defect'],'frozen_border_defect':self.last['border_defect'],
            'maximum_nodal_turn_rad':float(max(abs(self.last['turn']))),'algorithmic_windows':self.engine.windows.copy(),
            'active_algorithmic_bounds':[i for i,(v,(lo,hi)) in enumerate(zip(x,self.bounds,strict=True)) if min(v-lo,hi-v)<1e-7],
            'optimizer':optimizer,'full_joint_coordinates':x.tolist(),
            'no_new_energy_term':True,'no_symmetry_between_inner_outer_or_different_F_states_imposed':True,
            'scope':'fixed native references and contact stiffness; S50, S2, individual F2, centre3/F; other field/phase/source/inertia dynamics open',
            'physical_BST_validation':False,'physical_alignment_status':'UNKNOWN'}
        result['diagnostic']=diagnostic(result);return result
    def check(self,x):
        energy,g=self.evaluate(x);checks=[]
        directions={}
        for name,indices in [('shape',range(50)),('S_inner_outer',range(50,52)),('F_individual',range(52,self.nstate)),('centers',range(self.nstate,self.size)),('mixed',range(self.size))]:
            v=np.zeros(self.size);indices=list(indices);v[indices]=np.sin(np.arange(len(indices))*.37+.21);v/=np.linalg.norm(v);directions[name]=v
        for name,v in directions.items():
            expected=float(g@v)
            for h in [2e-5,1e-5]:
                plus,_=self.evaluate(x+h*v);minus,_=self.evaluate(x-h*v);fd=(plus-minus)/(2*h);defect=abs(fd-expected)
                assert defect<3e-5*max(1.,abs(expected)),(name,h,expected,fd)
                checks.append({'direction':name,'step':h,'analytic':expected,'finite_difference':fd,'abs_defect':defect})
        return checks
    def minimize(self,x,maxiter=240):
        start=time.time();energy,g=self.evaluate(x);diagonal=np.ones(len(x));invalid=0
        for j in range(self.nstate):
            step=np.zeros(len(x));step[j]=2e-4
            try:
                _,plus=self.evaluate(x+step);_,minus=self.evaluate(x-step)
                diagonal[j]=max(abs((plus[j]-minus[j])/4e-4),1e-3)
            except ValueError:diagonal[j]=1.;invalid+=1
        diagonal[self.nstate:]=np.tile([30.,max(2*self.m.weight,.01),max(2*self.m.weight,.01)],self.N)
        scale=1/np.sqrt(diagonal);z=x/scale;gz=g*scale;inverse=np.eye(len(x));trace=[];success=False;message='Budget achieved; retained status.'
        for iteration in range(maxiter):
            if max(abs(g[:self.nstate]))<1e-6 and max(abs(g[self.nstate:]))<1e-7:success=True;message='Stationarity in all declared states.';break
            direction=-(inverse@gz)
            if direction@gz>=0:direction=-gz;inverse=np.eye(len(z))
            physical=direction*scale
            alpha=min(1.,.05/max(float(np.max(abs(self.Z@physical[:50]))),1e-30),.03/max(float(max(abs(physical[50:self.nstate]))),1e-30),.08/max(float(max(abs(physical[self.nstate:]))),1e-30))
            accepted=False
            for attempt in range(32):
                trial=np.clip((z+alpha*direction)*scale,self.bounds[:,0],self.bounds[:,1]);zt=trial/scale
                try:and_field,gt=self.evaluate(trial)
                except ValueError:invalid+=1;alpha*=.5;continue
                displacement=zt-z
                if and_field<=energy+1e-4*float(gz@displacement):accepted=True;break
                alpha*=.5
            if not accepted:message='Not eligible native not found; calculation to be refined.';break
            difference=gt*scale-gz;product=float(displacement@difference)
            if product>1e-12*np.linalg.norm(displacement)*np.linalg.norm(difference):
                rho=1/product;A=np.eye(len(z))-rho*np.outer(displacement,difference);inverse=A@inverse@A.T+rho*np.outer(displacement,displacement)
            z=zt;energy=and_field;g=gt;gz=gt*scale
            trace.append({'iteration':iteration+1,'energy':float(energy),'state_gradient_inf':float(max(abs(g[:self.nstate]))),'center_gradient_inf':float(max(abs(g[self.nstate:])))})
        x=z*scale
        return self.result(x,{'success':success,'message':message,'iterations':len(trace),'calls':self.calls,'invalid_numerical_trials_retained_in_count':invalid,'trace':trace,'seconds':time.time()-start,'new_energy_term':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='SF4');ap.add_argument('--weight',type=float);ap.add_argument('--input',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--maxiter',type=int,default=240);ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    if a.input:
        d=json.loads(a.input.read_text());recipe=d.get('entry',{}).get('recipe',d.get('recipe',{}));point=d.get('result',d)
        if 'returned_shape' in point:recipe={**recipe,'shape':point['returned_shape'],'preparation':point['preparation'],'individual_F':point['individual_F_preparations'],'center_start':point['center_coordinates'],'kangle':point['existing_kangle']}
    else:
        d=json.loads((ROOT/'resultats/API_CALAGE_R335.json').read_text());recipe={**d['recipe'],'center_start':d['center_coordinates']}
    label=recipe.get('label',a.label);weight=a.weight if a.weight is not None else recipe.get('kangle',1.)
    system=JointStates(label,weight);x=system.initial(recipe);checks=system.check(x)
    initial=system.result(x);result=initial if a.check_only else system.minimize(x,a.maxiter)
    record={'revision':336,'label':label,'weight':weight,'source_recipe':recipe,'analytic_gradient_checks':checks,'initial':initial,'result':result,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'no_physical_region_excluded':True,'no_physical_validation':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,record)
    print(json.dumps({'label':label,'checks':len(checks),'angle_RMS':result['diagnostic']['all_angle_RMS_to_retrospective_reference_deg'],'state_g':max(result['maximum_internal_shape_gradient_native'],result['maximum_preparation_gradient_native']),'center_g':result['center_gradient_inf'],'optimizer':{k:v for k,v in (result['optimizer'] or {}).items() if k!='trace'}},ensure_ascii=False))
