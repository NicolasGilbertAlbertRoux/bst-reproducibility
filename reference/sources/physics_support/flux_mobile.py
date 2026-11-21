'Integrate an outward field pulse across two material cuts in mobile SF6.'

import argparse
from pathlib import Path

from fields_and_references import dump
from dynamics_size import CORE,OUTPUT,ROOT,Molecule,core_projection,field_inner
from trials_nodal import digest
from trajectories_moving import hashes,timeline
import numpy as np
from scipy.integrate import simpson


SOURCE = Path(__file__).resolve()
DURATION = .60
CUTS = (48,50)
CASES = {
    'HALO4_DT060':{'padding':4,'dt':.06,'pulse':True},
    'HALO4_DT030':{'padding':4,'dt':.03,'pulse':True},
    'HALO6_DT060':{'padding':6,'dt':.06,'pulse':True},
    'HALO4_TEMOIN_DT060':{'padding':4,'dt':.06,'pulse':False},
}
PULSE_SITE = 1
PULSE_AMPLITUDE = .025


def prepare_pulse(model):
    s = model.unpack(model.rest())
    n,t = s.n.copy(),s.t.copy()
    x,y,z = (np.indices((model.count,)*3,dtype=float)-.5*(model.count-1)).reshape(3,-1)
    xi = (x-(.5*(CORE-1)-1.))/1.5
    envelope = np.maximum(1.-xi**2,0.)
    transverse = np.maximum(1.-(y*y+z*z)/5.**2,0.)**3
    n[PULSE_SITE,:,2] += PULSE_AMPLITUDE*envelope**3*transverse
    n[PULSE_SITE] /= np.linalg.norm(n[PULSE_SITE],axis=1)[:,None]
    # Continuum outgoing preparation; the discrete dispersion can leave an inward part.
    t[PULSE_SITE,:,2] = PULSE_AMPLITUDE*6.*xi*envelope**2*transverse/(1.5*model.h[PULSE_SITE])
    t[PULSE_SITE] = core_projection(n[PULSE_SITE],t[PULSE_SITE])
    y,_ = model.project(model.pack(n,t,s.x,s.q,s.v))
    return y


def outside(model,y,derivative,d,cut):
    s,a = model.unpack(y),model.unpack(derivative)
    kernel = model.exterior[cut]
    w,wd = d['size_multipliers'],d['size_multiplier_rates']
    def site(row):
        n,t,acc,h,v,dv,w,wd = row
        V,dV = np.r_[np.zeros(3),v],np.r_[np.zeros(3),dv]
        Q,gn,gt,gv = kernel.quadratic(n,h,t,V)
        U,gu = kernel.potential(n,h)
        field_power = .5*w*(field_inner(gn,t)+field_inner(gt,acc))+field_inner(gu,t)
        return [.5*w*Q+U,field_power,.5*w*(gv@dV),.5*wd*Q,.5*w*Q,U]
    columns = np.array(list(model.pool.map(site,zip(s.n,s.t,a.t,model.h,s.v,a.v,w,wd,strict=True))))
    return {'energy_native':columns[:,0].sum(),'power_native':columns[:,1:4].sum(),
            'field_power_native':columns[:,1].sum(),'frame_power_native':columns[:,2].sum(),
            'size_multiplier_explicit_power_native':columns[:,3].sum(),
            'energy_by_site':columns[:,0],'power_by_site':columns[:,1:4].sum(axis=1),
            'energy_split_T_and_U_by_site':columns[:,4:]}


def outside_energy(model,y,cut):
    s = model.unpack(model.project(y)[0])
    ex = model.exterior[cut]
    def site(q):
        n,t,h,v = q
        r = model.radius(n,h)[0]
        Q,*_ = ex.quadratic(n,h,t,np.r_[np.zeros(3),v])
        U,_ = ex.potential(n,h)
        return .5*(r/model.ref)**model.p*Q+U
    return sum(model.pool.map(site,zip(s.n,s.t,model.h,s.v,strict=True)))


def interior_free_energy(model,s,d,cut):
    fields,rates = (model.crop(s.n,cut),model.crop(s.t,cut)) if cut<model.count else (s.n,s.t)
    action = model.model.m.action.action
    def site(row):
        n,t,h,v,w = row
        Q,*_ = model.kernel.quadratic(n,h,t,np.r_[np.zeros(3),v])
        U,_ = action(n,h)
        return [.5*w*Q,float(U)]
    return np.array(list(model.pool.map(site,zip(fields,rates,model.h,s.v,d['size_multipliers'],strict=True))))


def independent_checks(model,y,dy,d):
    s = model.unpack(y)
    full = interior_free_energy(model,s,d,model.count)
    inside = {i:interior_free_energy(model,s,d,i) for i in CUTS}
    by_cut = d['cuts']
    partition = {str(i):np.max(abs(full-inside[i]-by_cut[str(i)]['energy_split_T_and_U_by_site'])) for i in CUTS}
    direct_annulus = float(np.sum(inside[CUTS[1]]-inside[CUTS[0]]))
    ext_diff = by_cut[str(CUTS[0])]['energy_native']-by_cut[str(CUTS[1])]['energy_native']
    fd = {}
    for i in CUTS:
        flow = by_cut[str(i)]['power_native']
        checks = []
        for eps in (2e-4,1e-4):
            estimate = (outside_energy(model,y+eps*dy,i)-outside_energy(model,y-eps*dy,i))/(2.*eps)
            checks.append({'step':eps,'directional_FD_power_native':estimate,'absolute_defect':abs(estimate-flow)})
        fd[str(i)] = checks
    if max(partition.values())>3e-8 or abs(direct_annulus-ext_diff)>3e-8:
        raise ArithmeticError('The independent native energy cut failed')
    if max(z['absolute_defect'] for rows in fd.values() for z in rows)>2e-7:
        raise ArithmeticError('The flux did not match the directional derivative of exterior energy')
    return {'independent_T_and_U_partition_error_per_site_max':partition,
            'annular_energy_independent_native':direct_annulus,
            'annulus_partition_defect':abs(direct_annulus-ext_diff),'directional_power_FD':fd}


def ledger(rows):
    t = np.array([q['time_native'] for q in rows])
    out = {}
    for i in map(str,CUTS):
        a = [q['cuts'][i] for q in rows]
        H = np.array([q['energy_native'] for q in a])
        P = np.array([q['power_native'] for q in a])
        part = np.array([[q['field_power_native'],q['frame_power_native'],
                          q['size_multiplier_explicit_power_native']] for q in a])
        area = simpson(P,x=t)
        area_parts = simpson(part,x=t,axis=0)
        out[i] = {'energy_increase_native':H[-1]-H[0],'integrated_power_native':area,
                  'integrated_components_field_frame_explicit_size_native':area_parts,
                  'flux_energy_absolute_defect':abs(H[-1]-H[0]-area),
                  'power_components_integral_defect':abs(area-area_parts.sum()),
                  'min_max_power_native':[min(P),max(P)],
                  'energy_change_by_site':a[-1]['energy_by_site']-a[0]['energy_by_site']}
    return out


def run(name):
    case = CASES[name]
    model = Molecule('SF6',case['padding'])
    path = OUTPUT/f'{name}_R357.json'
    try:
        for i in CUTS:
            if i not in model.exterior:
                model.exterior[i] = model.exterior_kernel(i)
        s0 = model.rest()
        y = prepare_pulse(model) if case['pulse'] else s0
        H0 = model.energy(s0)
        def observed(y,dy,d,k):
            d['cuts'] = {str(i):outside(model,y,dy,d,i) for i in CUTS}
            if k in (0,round(DURATION/case['dt'])):
                d['independent_balance'] = independent_checks(model,y,dy,d)
        end,rows = timeline(model,y,DURATION,case['dt'],observed,log=name)
        first = rows[0]
        record = {'revision':357,'full':True,'physical_validation':False,'case_name':name,'case':case,
                  'duration_native':DURATION,'cut_node_counts':CUTS,'outer_node_count':model.count,
                  'grid_spacing_native_by_site':model.h,
                  'cut_half_width_native_by_site':{str(i):(.5*(i-1)*model.h) for i in CUTS},
                  'active_field_dofs':int(model.nsites*model.mask.sum()*3),
                  'scope':'evolved molecular site fields and frames, fixed outer vacuum, material cuts at same h',
                  'initial_release_also_tested_without_pulse':True,
                  'pulse':None if not case['pulse'] else {'site':PULSE_SITE,'amplitude':PULSE_AMPLITUDE,
                      'center_x_in_grid_steps':.5*(CORE-1)-1.,'longitudinal_half_width_grid_steps':1.5,
                      'transverse_radius_grid_steps':5.},
                  'initial_padded_rest_energy_native':H0,
                  'excitation_energy_native':first['H_native']-H0,
                  'max_total_H_drift_native':max(abs(d['H_native']-first['H_native']) for d in rows),
                  'max_P_drift_world':max(np.linalg.norm(d['P_world']-first['P_world']) for d in rows),
                  'max_J_drift_world':max(np.linalg.norm(d['J_world']-first['J_world']) for d in rows),
                  'ledger':ledger(rows),'history':rows,
                  'input_sha256':hashes(model,SOURCE,(
                       ROOT/'scripts/trajectoires_mobiles_R357.py',ROOT/'scripts/flux_nodal_R346.py'))}
        if name=='HALO4_DT030':
            snapshot = OUTPUT/'ETAT_FINAL_HALO4_R357.npz'
            np.savez_compressed(snapshot,state=end)
            record.update(final_state_file=snapshot.name,final_state_sha256=digest(snapshot))
        dump(path,record)
        print(name,'balances',record['ledger'],'max H drift',record['max_total_H_drift_native'],flush=True)
    finally:
        model.close()


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('case',choices=CASES)
    run(p.parse_args().case)
