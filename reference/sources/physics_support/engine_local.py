'Actual native engine for freely combined console controls, not a slider mockup.'
from pathlib import Path
import hashlib,json,time,threading,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.optimize import minimize
from angles_internal import Angles,ROOT

class NativeConsole:
    def __init__(self,parent=None,windows=None):
        self.windows={"shape_abs":3.2,"preparation_abs":2.4,"kangle_max":1000.,"turn_guard_max":6.0,**(windows or {})}
        if not all(np.isfinite(v) and v>0 for v in self.windows.values()):raise ValueError('Positive and finite digital windows required')
        self.parent=Path(parent) if parent else ROOT/'sources/R321'
        self.models={};self.lock=threading.Lock()
        with np.load(ROOT/'resultats/BASE_FORME_COMPLETE_R335.npz') as z:
            self.transform=z['raw_polynomial_to_shape_transform'];self.powers=z['polynomial_powers']
        self.basis_hash=hashlib.sha256((ROOT/'resultats/BASE_FORME_COMPLETE_R335.npz').read_bytes()).hexdigest()

    def model(self,label):
        if label not in self.models:
            m=Angles(self.parent,label);m.bounds=[(-3.,2.),(-np.pi,np.pi),(-np.pi,np.pi)]*m.N
            u=m.unit;angular=np.column_stack([u[:,0]**a*u[:,1]**b*u[:,2]**c for a,b,c in self.powers])
            raw=np.concatenate([angular*m.d['S_parts'][:,i,None] for i in range(2)],axis=1)
            # The same archived coordinate transform is transported, not
            # re-diagonalised for each molecule or preparation.
            Z=raw@self.transform
            self.models[label]=(m,Z)
        return self.models[label]

    def evaluate(self,request):
        started=time.time();recipe=dict(request);label=recipe.get('label','SF4')
        if label not in ['SF4','SF6']:raise ValueError('Unrecognized model')
        shape=np.asarray(recipe.get('shape',np.zeros(50)),float)
        if shape.shape!=(50,) or not np.isfinite(shape).all() or max(abs(shape))>self.windows["shape_abs"]:raise ValueError('50 finned coordinates expected in the declared algorithmic window')
        weight=float(recipe.get('kangle',1.));limit=float(recipe.get('maximum_profile_turn',1.2))
        if not np.isfinite(weight) or weight<0 or weight>self.windows["kangle_max"]:raise ValueError('Angular weight out of the declared exploratory window')
        if not np.isfinite(limit) or not .01<=limit<=self.windows["turn_guard_max"]:raise ValueError('Guard of rotation out of declared radians')
        with self.lock:
            m,Z=self.model(label);q=np.asarray(recipe.get('preparation',m.baseq),float)
            if q.shape!=(4,) or not np.isfinite(q).all() or max(abs(q))>self.windows["preparation_abs"]:raise ValueError('Internal preparation outside the reported exploratory box')
            m.set_preparation(q)
            individual=np.asarray(recipe.get('individual_F',[q[2:].tolist() for _ in range(m.N)]),float)
            if individual.shape!=(m.N,2) or not np.isfinite(individual).all() or np.max(abs(individual))>self.windows["preparation_abs"]:
                raise ValueError('Two internal coordinates completed by F expected in declared')
            Fderivatives=[]
            for i,coords in enumerate(individual,1):
                original=m.trial.fields[i];symbol,b,norm,parts=m.trial.templates[i]
                amp=parts@coords;theta=norm*amp
                fi=original*np.cos(theta)[:,None]+b*(amp*np.sinc(theta/np.pi))[:,None]
                derivative=-original*(norm*np.sin(theta))[:,None]+b*np.cos(theta)[:,None]
                fi[~m.trial.mask]=original[~m.trial.mask];derivative[~m.trial.mask]=0
                m.fields[i]=fi;ef,gf=m.trial.action(fi,float(m.d['F_h']));m.self_energies[i]=ef
                nodes=m.d['F_node_offset']+float(m.d['F_h'])**3*np.einsum('ij,ij->i',m.d['F_b_reference'],fi)
                m.F[i-1]=float((nodes@m.d[f'P_{i}']/m.d[f'normal_{i}'])[0])
                Fderivatives.append((derivative,gf,parts))
            base=m.fields[0].copy();baseenergies=m.self_energies.copy()
            radius=np.linalg.norm(base[:,1:],axis=1);unit=np.divide(base[:,1:],radius[:,None],out=np.zeros_like(base[:,1:]),where=radius[:,None]>1e-12)
            polar=np.column_stack([-radius,base[:,0,None]*unit]);active=m.trial.mask&(radius>1e-12);polar[~active]=0
            Z=Z.copy();Z[~active]=0;m.weight=weight
            if np.max(abs(Z@shape))>limit:
                return {'status':'UNKNOWN','reason':'The combination exceeds the declared numerical rotation guard.','recipe':recipe,
                    'retained':True,'physical_rejection':False,'physical_BST_validation':False}
            def set_shape(coefficients):
                turn=Z@coefficients
                field=base*np.cos(turn)[:,None]+polar*np.sin(turn)[:,None];field[~m.trial.mask]=base[~m.trial.mask]
                tangent=-base*np.sin(turn)[:,None]+polar*np.cos(turn)[:,None];tangent[~active]=0
                energy,g=m.trial.action(field,float(m.d['S_h']))
                m.fields[0]=field;m.nodes=m.d['S_node_offset']+m.h3*np.einsum('ij,ij->i',m.d['S_b_reference'],field)
                m.self_energies=baseenergies.copy();m.self_energies[0]=energy
                return turn,field,tangent,g
            relaxed=bool(recipe.get('relax_shape',False));optimizer=None
            if relaxed:
                # Correlated guard is numerical, not an extra material law.
                center_start=np.asarray(recipe.get('center_start',np.zeros(3*m.N)),float)
                if center_start.shape!=(3*m.N,) or not np.isfinite(center_start).all():raise ValueError('Initial point of the invalid centres')
                initial=np.r_[shape,center_start]
                calls=0
                def objective(x):
                    nonlocal calls
                    coeff=x[:50];c=x[50:];turn,field,tangent,g=set_shape(coeff);s=m.state(c)
                    g=g+m.h3*m.d['S_b_reference']*(s['weights']@(s['rho']/s['normal']))[:,None]
                    gy=Z.T@np.einsum('ij,ij->i',g,tangent);calls+=1
                    return s['energy_total'],np.r_[gy,s['gradient']]
                # Native-domain backtracking: an invalid chart is a rejected
                # numerical step, not an extra physical potential or a veto on BST.
                energy,gradient=objective(initial);diagonal=np.ones(len(initial))
                for j in range(50):
                    h=2e-4;step=np.zeros(len(initial));step[j]=h
                    _,plus=objective(initial+step);_,minus=objective(initial-step)
                    diagonal[j]=max(abs((plus[j]-minus[j])/(2*h)),1e-3)
                diagonal[50:]=np.tile([30.,max(2*weight,.01),max(2*weight,.01)],m.N)
                scale=1/np.sqrt(diagonal);z=initial/scale;gz=gradient*scale;inverse=np.eye(len(z))
                bounds=np.array([(-self.windows["shape_abs"],self.windows["shape_abs"])]*50+m.bounds);trace=[];rejected=0;message='Budget of programmes achieved.';success=False
                maxiter=int(min(max(recipe.get('maxiter',120),10),300))
                for iteration in range(maxiter):
                    if max(abs(gradient[:50]))<1e-6 and max(abs(gradient[50:]))<1e-7:
                        success=True;message='Qualified stationarity in the coordinates evaluated.';break
                    direction=-(inverse@gz)
                    if direction@gz>=0:direction=-gz;inverse=np.eye(len(z))
                    physical=direction*scale;profile_step=float(np.max(abs(Z@physical[:50])))
                    center_step=float(max(abs(physical[50:])))
                    alpha=min(1.,.05/max(profile_step,1e-30),.08/max(center_step,1e-30))
                    accepted=False
                    for attempt in range(30):
                        trial=np.clip((z+alpha*direction)*scale,bounds[:,0],bounds[:,1]);zt=trial/scale
                        if float(np.max(abs(Z@trial[:50])))>limit:
                            rejected+=1;alpha*=.5;continue
                        try:and_field,gt=objective(trial)
                        except ValueError:rejected+=1;alpha*=.5;continue
                        if and_field<=energy+1e-4*float(gz@(zt-z)):
                            accepted=True;break
                        alpha*=.5
                    if not accepted:message='Not eligible native not found; accuracy or domain to be taken up.';break
                    displacement=zt-z;difference=gt*scale-gz;product=float(displacement@difference)
                    if product>1e-12*np.linalg.norm(displacement)*np.linalg.norm(difference):
                        rho=1/product;A=np.eye(len(z))-rho*np.outer(displacement,difference)
                        inverse=A@inverse@A.T+rho*np.outer(displacement,displacement)
                    z=zt;energy=and_field;gradient=gt;gz=gt*scale
                    trace.append({'iteration':iteration+1,'energy_native':float(energy),
                        'shape_gradient_inf':float(max(abs(gradient[:50]))),'center_gradient_inf':float(max(abs(gradient[50:])))})
                final=z*scale;shape=final[:50];c=final[50:];turn,field,tangent,g=set_shape(shape);state=m.state(c)
                geometry=m.geometry(state)
                optimizer={'reported_success':success,'message':message,'iterations':len(trace),'evaluations':calls,
                    'correlated_guard_margin':limit-float(np.max(abs(Z@shape))),
                    'all_50_sulfur_shape_coordinates_relaxed':True,'native_steps_rejected_numerically':rejected,'trace':trace,
                    'no_penalty_energy_added':True}
            else:
                turn,field,tangent,g=set_shape(shape)
                initial=np.asarray(recipe.get('center_start',np.zeros(3*m.N)),float)
                if initial.shape!=(3*m.N,) or not np.isfinite(initial).all():raise ValueError('Initial point of the invalid centres')
                c,record=m.solve(initial);state=m.state(c);geometry=m.geometry(state)
                optimizer={'reported_success':record['center_stationarity_qualified'],'message':record['optimizer_message'],
                    'evaluations':record['evaluations'],'all_50_sulfur_shape_coordinates_relaxed':False}
            # Re-evaluate conjugate internal forces at the returned geometry.
            turn,field,tangent,g=set_shape(shape);state=m.state(c)
            g+=m.h3*m.d['S_b_reference']*(state['weights']@(state['rho']/state['normal']))[:,None]
            gshape=Z.T@np.einsum('ij,ij->i',g,tangent)
            Fgradients=[]
            for i,(derivative,gf,parts) in enumerate(Fderivatives,1):
                augmented=gf+float(m.d['F_h'])**3*m.d['F_b_reference']*(m.d[f'P_{i}']@(state['rho'][i-1:i]/m.d[f'normal_{i}']))[:,None]
                Fgradients.append((np.einsum('ij,ij->i',augmented,derivative)@parts).tolist())
            point_gradient=float(max(abs(state['gradient'])));shape_gradient=float(max(abs(gshape)))
            pairdist=[float(np.linalg.norm(state['R'][i]-state['R'][j])) for i in range(m.N) for j in range(i+1,m.N)]
            result={'revision':336,'algorithmic_windows':self.windows.copy(),'status':'FINITE_NATIVE_CALCULATION','label':label,'recipe':recipe,
                'numerical_engine_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'geometry_engine_sha256':hashlib.sha256((ROOT/'scripts/angles_internes_R335.py').read_bytes()).hexdigest(),
                'native_input_npz_sha256':hashlib.sha256((self.parent/'inputs'/f'{label}_ENTREES_NATIVES_R321.npz').read_bytes()).hexdigest(),
                'returned_shape':shape.tolist(),'preparation':q.tolist(),'existing_kangle':weight,
                'center_coordinates':c.tolist(),'geometry':geometry,'energy_total_native':state['energy_total'],
                'isolated_S_energy_change_native':m.self_energies[0]-baseenergies[0],
                'internal_shape_gradient_native':gshape.tolist(),'maximum_internal_shape_gradient_native':shape_gradient,
                'individual_F_preparations':individual.tolist(),'individual_F_conjugate_gradients_native':Fgradients,
                'F_profile_coordinates_are_states_not_new_species_laws':True,
                'center_gradient_inf':point_gradient,'center_stationarity_qualified':point_gradient<1e-7,
                'stationarity_in_all_50_S_shape_and_centre_coordinates_qualified':shape_gradient<1e-6 and point_gradient<1e-7,
                'maximum_nodal_turn_rad':float(max(abs(turn))),
                'field_norm_defect':float(max(abs(np.sum(field*field,axis=1)-np.sum(base*base,axis=1)))),
                'frozen_border_defect':float(np.max(abs(field[~m.trial.mask]-base[~m.trial.mask]))),
                'minimum_F_F_distance_native':min(pairdist),'optimizer':optimizer,
                'geometry_algorithmic_bounds':m.bounds,
                'active_geometry_algorithmic_bounds':[i for i,(value,(lo,hi)) in enumerate(zip(c,m.bounds)) if min(value-lo,hi-value)<1e-7],
                'shape_basis_sha256':self.basis_hash,'native_action_sha256':m.trial.action.hash,
                'scope':'native R324/R326 held-reference, fixed-stiffness family with transported full R327 polar shape span',
                'native_inertia_current_sources_and_F_frames_not_completed':True,
                'physical_BST_validation':False,'physical_alignment_status':'UNKNOWN','seconds':time.time()-started}
            if result['field_norm_defect']>3e-12 or result['maximum_nodal_turn_rad']>limit+1e-6:
                result.update(status='UNKNOWN',reason='Digital guard to be resumed.',retained=True,physical_rejection=False)
            return result
