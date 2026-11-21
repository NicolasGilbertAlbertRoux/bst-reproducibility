'Conditional native dynamics of the extra R322 profile direction.\n\nFields: exact pointwise S3 geodesic through the R321 candidate. Centres follow\nthe same radial potential roots as a declared holonomic trial constraint;\ntheir induced velocities are included in the R235 field kinetic metric.\nAll local spatial charts are parallel, as in R309. Independent contact-reader\ndirections are not reinterpreted as spatial frame rotations. Zero total spatial\nmomentum is reduced only after assembling all cross terms. No physical species,\npreparation, electric charge, SI time or full molecular stability is selected.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse
import ctypes
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline,CubicSpline


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Metric:
    def __init__(self,source):
        source=source.resolve();key=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        suffix='.dylib' if sys.platform=='darwin' else '.so'
        target=source.with_name('metrique_R323_'+key+suffix)
        if not target.exists():
            cc=shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
            if not cc:raise RuntimeError('C compiler required')
            mode=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
            subprocess.run([cc,'-O3',*mode,str(source),'-lm','-o',str(target)],check=True)
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f=ctypes.CDLL(str(target)).bst_curve_gram
        f.argtypes=[ctypes.c_int,ctypes.c_double,array,array,array,array];f.restype=ctypes.c_int;self.f=f

    def __call__(self,n,t,h,u):
        g=np.zeros((4,4));n=np.ascontiguousarray(n);t=np.ascontiguousarray(t);u=np.ascontiguousarray(u,dtype=float)
        if self.f(round((n.size/4)**(1/3)),h,n,t,u,g):raise ValueError('Outside gnomonic chart')
        return (g+g.T)/2


def direct_gram(n,t,h,u,jets):
    'Independent original R235 NumPy jet contraction, not the C routine.'
    result=np.zeros((4,4));shape=(*([round(len(n)**(1/3))]*3),4)
    for d,v,w in jets(n.reshape(shape),h,t.reshape((1,*shape))):
        mode0=v[:,0]-np.einsum('i,pic->pc',u,d)
        modes=np.concatenate([mode0[:,None,:],-d],axis=1)
        tr=np.sum(d*d,axis=(1,2));pd=np.einsum('pac,pic->pai',modes,d)
        result+=w*(np.einsum('pac,pbc,p->ab',modes,modes,1+tr)-np.einsum('pai,pbi->ab',pd,pd))
    return (result+result.T)/2


def curvature(well,r):
    'Exact second derivative of the received piecewise energy, away from joins.'
    x=r/well.radius_sum
    def hermite_second(x,left,right,y0,y1,d0,d1):
        width=right-left;t=(x-left)/width
        return ((12*t-6)*y0+(6*t-4)*width*d0+(-12*t+6)*y1+(6*t-2)*width*d1)/width**2
    if x<=well.x[0]:v=well.left_tail_curvature;segment='left_tail'
    elif x>=well.x[-1]:v=well.right_tail_curvature;segment='right_tail'
    elif well.local_lower<=x<=well.local_upper:
        v=np.polyval(np.polyder(well.coefficients,2),x-well.origin);segment='local_polynomial'
    elif well.left_anchor<=x<well.local_lower:
        y1,d1=well.local(well.local_lower);i=well.left_anchor_index
        v=hermite_second(x,well.left_anchor,well.local_lower,well.y[i],y1,well.slopes[i],d1);segment='left_bridge'
    elif well.local_upper<x<well.right_anchor:
        y0,d0=well.local(well.local_upper);i=well.right_anchor_index
        v=hermite_second(x,well.local_upper,well.right_anchor,y0,well.y[i],d0,well.slopes[i]);segment='right_bridge'
    else:
        i=max(0,min(len(well.x)-2,int(np.searchsorted(well.x,x)-1)))
        v=hermite_second(x,well.x[i],well.x[i+1],well.y[i],well.y[i+1],well.slopes[i],well.slopes[i+1]);segment='pchip_'+str(i)
    return float(v/well.radius_sum**2),segment


class Curve:
    def __init__(self,parent,r322,label,metric):
        sys.path.insert(0,str(parent/'scripts'));sys.path.insert(0,str(r322))
        from bench_native import Trial
        from diagnose_field_R322 import energy_gradient,split,hemisphere_margin
        self.energy_gradient=energy_gradient;self.hemisphere_margin=hemisphere_margin
        self.model=Trial(parent/'inputs',label);self.metric=metric;self.label=label
        received=json.loads((parent/f'{label}_BALAYAGE_ET_CANDIDAT_R321.json').read_text())
        self.base,self.derivatives=self.model.reconstruct(np.asarray(received['two_sided_trial']['coordinates']))
        U,g,_=energy_gradient(self.model,self.base)
        sectors=[split(n,gg,self.model.mask) for n,gg in zip(self.base,g,strict=True)]
        gram=np.zeros((4,4));rhs=np.zeros(4);shapes=[]
        for i,(sector,t,template) in enumerate(zip(sectors,self.derivatives,self.model.templates,strict=True)):
            idx=[0,1] if i==0 else [2,3]
            Z=np.einsum('ij,ij->i',t,sector['polar_basis'])[:,None]*template[-1]
            shapes.append(Z);gram[np.ix_(idx,idx)]+=Z.T@Z;rhs[idx]+=Z.T@sector['scalar']
        coefficient=np.linalg.solve(gram,rhs);v=[]
        for i,(sector,Z) in enumerate(zip(sectors,shapes,strict=True)):
            idx=[0,1] if i==0 else [2,3]
            v.append(-sector['polar_basis']*(sector['scalar']-Z@coefficient[idx])[:,None])
        maximum=max(np.max(np.linalg.norm(t,axis=1)) for t in v)
        self.v=[t/maximum for t in v];self.speed=[np.linalg.norm(t,axis=1) for t in self.v]
        self.initial_slope=sum(float(np.sum(gg*t)) for gg,t in zip(g,self.v,strict=True));self.initial_U=U
        assert max(float(np.max(abs(np.sum(n*t,axis=1)))) for n,t in zip(self.base,self.v,strict=True))<1e-12
        self.calls=0

    def fields(self,s):
        fields=[];tangents=[]
        for n,v,w in zip(self.base,self.v,self.speed,strict=True):
            theta=s*w
            fields.append(n*np.cos(theta)[:,None]+v*(s*np.sinc(theta/np.pi))[:,None])
            tangents.append(-n*(w*np.sin(theta))[:,None]+v*np.cos(theta)[:,None])
        return fields,tangents

    def evaluate(self,s,check_direct=None):
        fields,tangents=self.fields(s);d=self.model.data
        U,g,details=self.energy_gradient(self.model,fields)
        of=sum(float(np.sum(gg*t)) for gg,t in zip(g,tangents,strict=True))
        shifts=[]
        for i,t in enumerate(tangents):
            symbol='S' if i==0 else 'F';h=float(d[symbol+'_h'])
            shifts.append((h**3*np.einsum('ij,ij->i',d[symbol+'_b_reference'],t))@d[f'P_{i}']/d[f'normal_{i}'])
        da=shifts[0]+np.asarray([x[0] for x in shifts[1:]])
        curv_and_segments=[curvature(self.model.well,r) for r in details['radii']]
        denominator=d['stiffness']+np.asarray([x[0] for x in curv_and_segments]);assert min(denominator)>0
        dr=d['stiffness']*da/denominator
        center_slopes=[np.zeros(3)]+[a*b for a,b in zip(dr,d['directions_s'],strict=True)]
        G=np.zeros((4,4));fixed=np.zeros((4,4));direct_defect=None
        for i,(n,t,u) in enumerate(zip(fields,tangents,center_slopes,strict=True)):
            symbol='S' if i==0 else 'F';h=float(d[symbol+'_h'])
            local=self.metric(n,t,h,u);G+=local
            if s==0.:fixed+=self.metric(n,t,h,np.zeros(3))
            if check_direct is not None and i in [0,1]:
                independent=direct_gram(n,t,h,u,check_direct)
                direct_defect=max(direct_defect or 0.,float(np.linalg.norm(independent-local)/np.linalg.norm(independent)))
                assert direct_defect<3e-11
        A=G[1:,1:];B=np.linalg.solve(A,G[1:,0]);m=float(G[0,0]-G[0,1:]@B)
        assert min(np.linalg.eigvalsh(G))>0 and m>0
        norm=max(float(np.max(abs(np.sum(n*n,axis=1)-1))) for n in fields)
        border=max(float(np.max(abs(n[~self.model.mask]-b[~self.model.mask]))) for n,b in zip(fields,self.base,strict=True))
        self.calls+=1
        return {'s':float(s),'U':U,'dU':of,'G':G.tolist(),'m_zero_momentum':m,'translation_connection':B.tolist(),
          'radii':details['radii'].tolist(),'radii_slope':dr.tolist(),'root_denominator_min':float(min(denominator)),
          'well_segments':[x[1] for x in curv_and_segments],
          'field_norm_defect':norm,'frozen_border_defect':border,
          'independent_metric_relative_defect':direct_defect,
          'm_without_center_motion_diagnostic':float(fixed[0,0]-fixed[0,1:]@np.linalg.solve(fixed[1:,1:],fixed[1:,0])) if s==0. else None}


class Reduced:
    def __init__(self,rows,indices):
        selected=[rows[i] for i in indices];self.nodes=np.array([r['s'] for r in selected]);self.offset=min(rows,key=lambda r:abs(r['s']))['U']
        self.U=CubicHermiteSpline(self.nodes,[r['U']-self.offset for r in selected],[r['dU'] for r in selected],extrapolate=False)
        self.m=CubicSpline(self.nodes,[r['m_zero_momentum'] for r in selected],extrapolate=False)
        self.B=CubicSpline(self.nodes,np.array([r['translation_connection'] for r in selected]),axis=0,extrapolate=False)
        assert self.m(np.linspace(self.nodes[0],self.nodes[-1],201)).min()>0

    def rhs(self,t,y,omit_connection=False):
        s,v=y[:2];mass=float(self.m(s));dm=float(self.m(s,1))
        if not np.isfinite(mass):raise ValueError('Trajectory left declared interpolation interval')
        acceleration=(-float(self.U(s,1))-(0. if omit_connection else .5*dm*v*v))/mass
        return np.r_[v,acceleration,-self.B(s)*v]

    def solve(self,horizon,rtol,atol,omit_connection=False):
        sol=solve_ivp(lambda t,y:self.rhs(t,y,omit_connection),(0,horizon),np.zeros(5),method='DOP853',
                      rtol=rtol,atol=atol,dense_output=True,max_step=horizon/16)
        assert sol.success
        times=np.linspace(0,horizon,161);y=sol.sol(times);s,v=y[:2]
        T=.5*self.m(s)*v*v;U=self.U(s);E=T+U
        return {'times':times,'state':y,'T':T,'U':U,'E':E,'nfev':sol.nfev}


def run(parent,r322,r249,out,label,metric,reuse_curves=False):
    started=time.monotonic();sys.path.insert(0,str(r249/'sources'))
    from kinetic_native import jets
    curve=Curve(parent,r322,label,metric);grid=np.linspace(-.005,.015,33);rows=[]
    assert abs(grid[8])<1e-15
    if reuse_curves:
        rows=json.loads((out/f'{label}_COURBE_R323.json').read_text())['rows']
        assert len(rows)==33 and np.max(abs(np.array([r['s'] for r in rows])-grid))<1e-15
        assert abs(rows[8]['U']-curve.initial_U)<2e-8
        print(label,'reusing recorded native curve after initial-state reproduction',flush=True)
    else:
        for i,s in enumerate(grid):
            if i==8:s=0.
            row=curve.evaluate(float(s),jets if i in [8,24] else None);rows.append(row)
            if i%4==0:print(label,'native curve',i+1,'/ 33',flush=True)
            dump(out/f'{label}_COURBE_R323.json',{'revision':323,'rows':rows})
    initial=rows[8];assert abs(initial['dU']-curve.initial_slope)<2e-8
    # Independent finite differences of the native potential, root and metric.
    derivative_checks=[]
    for h in [1e-4,5e-5]:
        p=curve.evaluate(h);m=curve.evaluate(-h)
        slope=(p['U']-m['U'])/(2*h);rp=(np.array(p['radii'])-np.array(m['radii']))/(2*h)
        finite_curvature=[]
        for r in initial['radii']:
            finite=(curve.model.well.separation_energy_derivative(r+h)[1]-curve.model.well.separation_energy_derivative(r-h)[1])/(2*h)
            exact=curvature(curve.model.well,r)[0];finite_curvature.append(abs(finite-exact))
        derivative_checks.append({'step':h,'potential_slope_fd':slope,'potential_slope_defect':abs(slope-initial['dU']),
            'radii_slope_max_defect':float(np.max(abs(rp-initial['radii_slope']))),
            'well_curvature_fd_max_defect':max(finite_curvature),
            'm_slope_fd':(p['m_zero_momentum']-m['m_zero_momentum'])/(2*h)})
    # The inherited R5 potential is C1, not globally C2. Root-following may
    # therefore have slope jumps. Do not smooth a mass across an untraversed
    # contact-piece join on the negative side of the exploratory curve.
    dynamic_rows=rows[8:]
    assert all(r['well_segments']==initial['well_segments'] for r in dynamic_rows)
    fine=Reduced(dynamic_rows,range(25));coarse=Reduced(dynamic_rows,range(0,25,2))
    mfd=derivative_checks[-1]['m_slope_fd'];mder=float(fine.m(0.,1))
    assert derivative_checks[-1]['potential_slope_defect']<1e-4
    assert derivative_checks[-1]['radii_slope_max_defect']<1e-6
    assert abs(mfd-mder)<max(1e-5,abs(mfd)*3e-5)
    offgrid=[]
    for s in [.002137,.006219,.010373]:
        native=curve.evaluate(s,jets if s==.006219 else None)
        offgrid.append({'s':s,'U_abs_interpolation_defect':abs(float(fine.U(s))+fine.offset-native['U']),
          'dU_abs_interpolation_defect':abs(float(fine.U(s,1))-native['dU']),
          'mass_relative_interpolation_defect':abs(float(fine.m(s))-native['m_zero_momentum'])/native['m_zero_momentum'],
          'independent_S_and_moving_F_metric_relative_defect':native['independent_metric_relative_defect']})
    assert max(x['U_abs_interpolation_defect'] for x in offgrid)<1e-6
    assert max(x['mass_relative_interpolation_defect'] for x in offgrid)<1e-5
    horizon=float(np.sqrt(2*.01*initial['m_zero_momentum']/(-initial['dU'])))
    primary=fine.solve(horizon,2e-11,2e-13);loose=fine.solve(horizon,2e-9,2e-11)
    sparse=coarse.solve(horizon,2e-11,2e-13);wrong=fine.solve(horizon,2e-11,2e-13,True)
    scale=max(float(np.max(primary['T'])),1e-12)
    balance=float(np.max(abs(primary['E']-primary['E'][0]))/scale)
    wrong_balance=float(np.max(abs(wrong['E']-wrong['E'][0]))/scale)
    solver_defect=float(np.max(abs(primary['state']-loose['state'])))
    interpolation_defect=float(np.max(abs(primary['state']-sparse['state'])))
    assert balance<2e-8 and solver_defect<2e-8 and interpolation_defect<2e-6
    s_end=float(primary['state'][0,-1]);native_end=curve.evaluate(s_end)
    v_end=float(primary['state'][1,-1]);native_T=.5*native_end['m_zero_momentum']*v_end*v_end
    native_balance=abs(native_end['U']-initial['U']+native_T)/scale
    assert native_balance<2e-6
    # Momentum reconstructed using the complete cross blocks at every sample.
    pmax=0.;kinetic_defect=0.
    # Interpolate full G only for this independent reduced/full algebra check.
    Gfit=CubicSpline(np.array([row['s'] for row in dynamic_rows]),np.asarray([row['G'] for row in dynamic_rows]),axis=0)
    for s,v in primary['state'][:2].T:
        G=Gfit(s);vel=np.r_[v,-fine.B(s)*v]
        pmax=max(pmax,float(np.linalg.norm((G@vel)[1:])))
        kinetic_defect=max(kinetic_defect,abs(.5*float(vel@G@vel)-.5*float(fine.m(s))*v*v))
    native_final_velocity=np.r_[v_end,-fine.B(s_end)*v_end]
    native_final_momentum=float(np.linalg.norm((np.asarray(native_end['G'])@native_final_velocity)[1:]))
    assert pmax<1e-9 and native_final_momentum<1e-9 and kinetic_defect<1e-8
    margin=curve.hemisphere_margin(curve.base,curve.v,max(abs(grid)),tuple(curve.model.data['grid_shape']))
    result={'revision':323,'label':label,'construction':'R235 intrinsic kinetic pullback on the R322 extra profile curve with R321 radial-root holonomic trial constraint',
      'coordinate_s_is_not_physical_time':True,'native_time_not_SI':True,
      'no_physical_species_preparation_or_conserved_internal_charge_selected':True,
      'spatial_charts_parallel_as_R309_not_inferred_from_contact_reader_directions':True,
      'directions_and_four_original_gauge_amplitudes_fixed':True,
      'radial_root_following_is_declared_trial_constraint_not_physical_adiabatic_law_proved':True,
      'moving_centers_induced_kinetic_and_profile_translation_cross_terms_included':True,
      'zero_total_spatial_momentum_reduced_after_assembly':True,
      'initial_native_U':initial['U'],'initial_native_profile_force':-initial['dU'],
      'initial_Gram_profile_and_translation':initial['G'],'initial_effective_profile_mass':initial['m_zero_momentum'],
      'initial_effective_mass_if_center_velocities_omitted':initial['m_without_center_motion_diagnostic'],
      'initial_profile_acceleration_from_rest':-initial['dU']/initial['m_zero_momentum'],
      'exploratory_native_curve_interval':[float(grid[0]),float(grid[-1])],'exploratory_native_curve_nodes':33,
      'dynamic_interpolation_interval':[0.,float(grid[-1])],'fine_curve_nodes':25,'coarse_curve_nodes':13,
      'minimum_mass_over_native_nodes':min(x['m_zero_momentum'] for x in rows),
      'independent_R235_numpy_vs_C_max_relative_defect':max([x['independent_metric_relative_defect'] or 0. for x in rows]+[x['independent_S_and_moving_F_metric_relative_defect'] or 0. for x in offgrid]),
      'directional_native_derivative_checks':derivative_checks,'mass_derivative_interpolation_vs_fd_abs_defect':abs(mfd-mder),
      'offgrid_native_checks':offgrid,'time_horizon_native':horizon,
      'trajectory_final_profile_coordinate':s_end,'trajectory_profile_velocity_final':v_end,
      'trajectory_final_center_recoil':primary['state'][2:,-1].tolist(),
      'trajectory_potential_change_native':native_end['U']-initial['U'],'trajectory_final_kinetic_native_direct':native_T,
      'interpolated_energy_balance_over_exchanged_energy':balance,
      'direct_native_final_energy_balance_over_exchanged_energy':native_balance,
      'time_tolerance_refinement_max_abs_state_defect':solver_defect,
      'curve_13_vs_25_node_max_abs_state_defect':interpolation_defect,
      'reconstructed_zero_momentum_abs_defect':pmax,
      'direct_native_final_momentum_abs_defect':native_final_momentum,
      'full_Gram_vs_interpolated_reduced_kinetic_max_abs_defect':kinetic_defect,
      'countertest_omit_metric_connection_energy_balance_defect':wrong_balance,
      'field_norm_defect_native_nodes':max(x['field_norm_defect'] for x in rows),
      'fixed_border_defect_native_nodes':max(x['frozen_border_defect'] for x in rows),
      'curve_interval_homotopy_hemisphere_margin_rad':margin,
      'well_segments_at_start':initial['well_segments'],
      'no_segment_change_on_sampled_native_nodes':all(x['well_segments']==initial['well_segments'] for x in rows),
      'no_segment_change_on_actual_dynamic_interpolation_nodes':True,
      'interpolation_does_not_cross_untraversed_negative_piece_join':True,
      'global_R5_C1_join_not_repaired_or_declared_C2':True,
      'new_whole_validation':False,'new_physical_domain_validation':False,
      'not_full_field_dynamics_or_long_time_stability_proof':True,'native_evaluations':curve.calls,'seconds':time.monotonic()-started}
    dump(out/f'{label}_DYNAMIQUE_R323.json',result)
    np.savez_compressed(out/f'{label}_TRAJECTOIRE_R323.npz',time_native=primary['times'],state_profile_velocity_recoil=primary['state'],
                        kinetic_native=primary['T'],potential_delta_native=primary['U'],energy_delta_native=primary['E'])
    print(label,'dynamic pullback PASS','mass',initial['m_zero_momentum'],'balance',native_balance,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['parent','r322','r249','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reuse-curves',action='store_true',help='Reuse same-run native curve checkpoints, after reproducing the initial state')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    metric=Metric(Path(__file__).with_name('metrique_courbe_R323.c'));results=[]
    for label in ['SF4','SF6']:
        results.append(run(a.parent.resolve(),a.r322.resolve(),a.r249.resolve(),a.output.resolve(),label,metric,a.reuse_curves))
        dump(a.output/'BILAN_DYNAMIQUE_PROFIL_R323.json',{'revision':323,'results':results,'new_physical_validation':False})
