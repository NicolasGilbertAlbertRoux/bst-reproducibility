'Atomic preparation -> freely relaxed centre angles at fixed received laws.\n\nUses the actual R321 four-coordinate atomic reconstruction and the received\nR324/R326 fixed-contact-stiffness, direction-dependent port potential. This\nis a conditional diagnostic: F reader orientations and the soft Gram reference\nremain held. It is not the full R330 mobile-stiffness/field/reference dynamics.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from scipy.optimize import minimize,root

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources/R321'))
from bench_native import Trial

def dump(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def move(base,w):
    b=np.linalg.norm(w,axis=-1);return base*np.cos(b)[:,None]+w*np.sinc(b/np.pi)[:,None]

def pullback(base,w,g):
    b=np.linalg.norm(w,axis=-1);sinc=np.sinc(b/np.pi);c=np.empty_like(b);small=b<1e-3
    c[small]=-1/3+b[small]**2/30-b[small]**4/840
    z=b[~small];c[~small]=(z*np.cos(z)-np.sin(z))/z**3
    return sinc[:,None]*g+w*(-sinc*np.sum(base*g,axis=1)+c*np.sum(w*g,axis=1))[:,None]

class Angles:
    def __init__(self,parent,label):
        self.trial=Trial(parent/'inputs',label);self.d=self.trial.data;self.label=label
        self.baseq=np.array(json.loads((parent/f'{label}_BALAYAGE_ET_CANDIDAT_R321.json').read_text())['two_sided_trial']['coordinates'])
        self.trial.evaluate(self.baseq);self.rbase=self.trial.cache['radii'].copy()
        self.D0=self.d['directions_s'];self.N=len(self.D0)
        config=json.loads((ROOT/'sources/CONTRAT_MOLECULAIRE_R231.json').read_text())[label]
        self.target=np.array(config['target_gram']);self.weight=float(config['kangle']);assert self.weight==1.
        grid=np.indices(tuple(self.d['grid_shape'])).reshape(3,-1).T-(self.d['grid_shape']-1)/2
        self.unit=grid/np.linalg.norm(grid,axis=1)[:,None]
        self.h3=float(self.d['S_h'])**3
        self.density=self.h3*np.sum(self.d['S_b_reference']**2,axis=1)
        self.tangents=[]
        for d in self.D0:
            axis=np.eye(3)[np.argmin(abs(d))];t=np.cross(d,axis);t/=np.linalg.norm(t)
            self.tangents.append(np.stack([t,np.cross(d,t)]))
        self.tangents=np.array(self.tangents);self.bounds=[(-.5,.5),(-.8,.8),(-.8,.8)]*self.N
        self.set_preparation(self.baseq)

    def set_preparation(self,q):
        self.q=np.array(q);self.fields,self.derivatives=self.trial.reconstruct(self.q);self.self_energies=[];self.raw_gradients=[];shifts=[]
        for i,field in enumerate(self.fields):
            key='S' if i==0 else 'F';h=float(self.d[key+'_h']);energy,g=self.trial.action(field,h)
            self.self_energies.append(energy);self.raw_gradients.append(g)
            nodes=self.d[key+'_node_offset']+h**3*np.einsum('ij,ij->i',self.d[key+'_b_reference'],field)
            shifts.append(nodes@self.d[f'P_{i}']/self.d[f'normal_{i}'])
        self.nodes=self.d['S_node_offset']+self.h3*np.einsum('ij,ij->i',self.d['S_b_reference'],self.fields[0])
        self.F=np.array([v[0] for v in shifts[1:]])

    def state(self,c,gradient=True):
        c=np.asarray(c).reshape(self.N,3);r=self.rbase*np.exp(c[:,0]);w=np.einsum('ia,iac->ic',c[:,1:],self.tangents);D=move(self.D0,w)
        pos=np.maximum(self.unit@D.T,0.);p=pos**4;normal=self.density@p**2
        if min(normal)<=0:raise ValueError('Undefined directional port')
        qs=self.nodes@p/normal;delta=r-float(self.d['r0'])-qs-self.F;k=self.d['stiffness'];rho=-k*delta
        wells=np.array([self.trial.well.separation_energy_derivative(x) for x in r])
        diff=D@D.T-self.target;gram=.5*self.weight*float(np.sum(diff*diff))
        contact=.5*float(k@(delta*delta));energy=float(np.sum(wells[:,0]))+contact+gram
        result={'energy_geometry':energy,'energy_total':energy+sum(self.self_energies),'D':D,'R':r[:,None]*D,'radii':r,
            'sulfur_ports':qs,'rho':rho,'weights':p,'normal':normal,'Gram_energy':gram,'contact_energy':contact}
        if gradient:
            ds=(4*self.unit.T@(self.nodes[:,None]*pos**3)).T/normal[:,None]
            dn=(8*self.unit.T@(self.density[:,None]*pos**7)).T
            ds-=qs[:,None]*dn/normal[:,None]
            gd=rho[:,None]*ds+2*self.weight*diff@D
            tangent=gd-D*np.sum(D*gd,axis=1)[:,None]
            gR=(wells[:,1]+k*delta)[:,None]*D+tangent/r[:,None]
            back=pullback(self.D0,w,r[:,None]*gR)
            gc=np.zeros((self.N,3));gc[:,0]=r*np.sum(gR*D,axis=1);gc[:,1:]=np.einsum('iac,ic->ia',self.tangents,back)
            result['gradient']=gc.ravel()
        return result

    def gradient_check(self,c):
        v=np.sin(np.arange(3*self.N)*.37+.2);v/=np.linalg.norm(v);exact=float(self.state(c)['gradient']@v);rows=[]
        for h in [2e-5,1e-5]:
            fd=(self.state(c+h*v,False)['energy_geometry']-self.state(c-h*v,False)['energy_geometry'])/(2*h)
            defect=abs(fd-exact);assert defect<2e-7
            rows.append({'step':h,'analytic':exact,'finite_difference':fd,'absolute_defect':defect})
        return rows

    def geometry(self,state):
        D=state['D'];angles=np.degrees(np.arctan2(np.linalg.norm(np.cross(D[:,None],D[None,:]),axis=-1),D@D.T))
        result={'all_pair_angles_deg':angles.tolist(),'radii_native':state['radii'].tolist(),'centers_native':state['R'].tolist()}
        if self.label=='SF4':result.update(equatorial_angle_deg=float(angles[2,3]),axial_angle_deg=float(angles[0,1]))
        else:result['opposite_pair_angles_deg']=[float(angles[i,i+1]) for i in [0,2,4]]
        return result

    def solve(self,c0):
        count=0
        def objective(c):
            nonlocal count
            s=self.state(c);count+=1;return s['energy_geometry'],s['gradient']
        sol=minimize(objective,c0,jac=True,method='L-BFGS-B',bounds=self.bounds,
            options={'maxiter':450,'maxcor':30,'maxls':40,'gtol':2e-11,'ftol':1e-15})
        c=sol.x;state=self.state(c)
        # Refine the actual free-centre stationarity equations, then keep it
        # only when inside the same chart and not worse in residual.
        refined=root(lambda x:self.state(x)['gradient'],c,method='hybr',options={'xtol':1e-9,'maxfev':180})
        if np.isfinite(refined.x).all() and all(lo<x<hi for x,(lo,hi) in zip(refined.x,self.bounds)):
            candidate=self.state(refined.x)
            if max(abs(candidate['gradient']))<max(abs(state['gradient'])):c=refined.x;state=candidate
        active=[i for i,(x,(lo,hi)) in enumerate(zip(c,self.bounds)) if min(x-lo,hi-x)<1e-7]
        gmax=float(max(abs(state['gradient'])));normdef=0.;border=0.;changes=[]
        for n,base in zip(self.fields,self.trial.fields):
            normdef=max(normdef,float(max(abs(np.sum(n*n,axis=1)-np.sum(base*base,axis=1)))))
            border=max(border,float(np.max(abs(n[~self.trial.mask]-base[~self.trial.mask]))))
            changes.append(float(np.max(np.linalg.norm(n-base,axis=1))))
        assert normdef<3e-12 and border==0
        record={'preparation':self.q.tolist(),'center_coordinates':c.tolist(),'geometry':self.geometry(state),
            'center_gradient_inf':gmax,'center_stationarity_qualified':gmax<1e-7 and not active,
            'active_algorithmic_geometry_bounds':active,'optimizer_message':str(sol.message),'evaluations':count,
            'energy_geometry_native':state['energy_geometry'],'energy_total_native':state['energy_total'],
            'atomic_self_energies_native':self.self_energies,'field_change_from_received_reference_max_by_atom':changes,
            'field_norm_defect':normdef,'frozen_border_defect':border,
            'all_law_coefficients_and_Gram_reference_unchanged':bool(self.weight==1.),
            'Gram_reference_unchanged':True,'existing_angular_coefficient_changed_from_received':bool(self.weight!=1.),
            'internal_preparation_prescribed_not_fully_relaxed':True,'F_readout_orientations_held':True,
            'fixed_stiffness_family_not_full_R330_mobile_stiffness':True,'physical_BST_validation':False}
        return c,record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent-r321',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['SF4','SF6']);parser.add_argument('--pilot',action='store_true');a=parser.parse_args()
    start=time.time();results={}
    for label in a.labels:
        model=Angles(a.parent_r321,label);baseline,base=model.solve(np.zeros(3*model.N))
        checks=model.gradient_check(baseline+.001*np.sin(np.arange(3*model.N)))
        record={'label':label,'base_preparation':model.baseq.tolist(),'baseline':base,'gradient_checks':checks,
            'whole_preparation_box_hemisphere_check':model.trial.hemisphere_box_bound(.8),'axes':{}}
        results[label]=record
        axes=[0] if a.pilot else range(4);nodes=[-.8,0.,.8] if a.pilot else [-.8,-.4,0.,.4,.8]
        for axis in axes:
            rows=[]
            for value in nodes:
                q=model.baseq.copy();q[axis]=value;model.set_preparation(q)
                c,row=model.solve(baseline.copy());row['axis']=axis;row['axis_value']=value;rows.append(row)
                print(label,'axis',axis,'value',value,'geometry',row['geometry'].get('equatorial_angle_deg',row['geometry'].get('opposite_pair_angles_deg')),
                    'g',row['center_gradient_inf'],'elapsed',round(time.time()-start,2),flush=True)
                record['axes'][str(axis)]=rows
                dump(ROOT/'results'/('PILOTE_ANGLES_R335.json' if a.pilot else 'BANQUE_ANGLES_INTERNES_R335.json'),
                    {'revision':335,'full':False,'models':results,'scope':'conditional fixed-stiffness R324/R326 port family at prescribed R321 internal states',
                    'full_R330_field_and_reference_relaxation_complete':False})
    out=ROOT/'results'/('PILOTE_ANGLES_R335.json' if a.pilot else 'BANQUE_ANGLES_INTERNES_R335.json')
    final=json.loads(out.read_text());final['full']=True;dump(out,final)

if __name__=='__main__':main()
