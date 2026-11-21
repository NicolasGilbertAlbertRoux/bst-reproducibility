'Check whether a disjoint exterior fluctuation alters core inertia at rest.'

from pathlib import Path

from fields_and_references import dump
from contact_opposition_sf4 import FINAL
from dynamics_full import NodalKinetic
from dynamics_nodal_scripts import VACUUM,pad_cube
from dynamics_size import CORE,OUTPUT,ROOT,SIZES,core_projection
from trials_nodal import digest
from solve_acceleration_nodal import constrained_inverse
from response_of_frames import STATE
from relax_nodes import NodalSF6
from size_dynamics import Radius
from sizes_substrate import nodes
from sweep_sf4 import read
import numpy as np


SOURCE = Path(__file__).resolve()
PADDING = 8
GRID = CORE+2*PADDING
CUT_COUNTS = (16,20,24,28,32,36,40,44,48)
PERTURBED_SITE = 1
AMPLITUDES = (.05,.10,.20)
CENTERS_X = (27.5,29.0)
BUMPS_WIDTH_GRID = 2.35


def crop(n,m):
    N = round(len(n)**(1/3))
    p = (N-m)//2
    s = slice(p,p+m)
    return np.ascontiguousarray(n.reshape(N,N,N,4)[s,s,s]).reshape(-1,4)


def halo_patch(n,center,amp):
    coordinates = nodes(GRID,1.).reshape(-1,3)-[center,0.,0.]
    envelope = np.maximum(1.-np.sum(coordinates**2,axis=1)/BUMPS_WIDTH_GRID**2,0.)**3
    support = envelope>0.
    if not np.array_equal(n[support],np.broadcast_to(VACUUM,n[support].shape)):
        raise ArithmeticError('The distant perturbation does not begin in the exact vacuum')
    f = coordinates[support]*envelope[support,None]
    z = amp*f/np.max(np.linalg.norm(f,axis=1))
    out = n.copy()
    out[support] = np.column_stack((np.sqrt(1.-np.sum(z*z,axis=1)),z))
    if not np.array_equal(crop(out,CORE),crop(n,CORE)):
        raise ArithmeticError('The perturbation intersects the native contact domain')
    return out,support


def plateau(path,radius,p):
    with np.load(path) as s:
        n,h = s['fields'],s['h_native']
    result = []
    for site,(field,step) in enumerate(zip(n,h,strict=True)):
        R = [radius(crop(field,c),step)[0] for c in CUT_COUNTS]
        factors = (np.array(R)/R[-1])**p
        result.append({'site':site,'spacing_native':float(step),'cut_RMS_radii_native':R,
                       'size_factors_relative_to_N48':factors})
    return result


def main():
    p = read(SIZES)['calibration_of_size_law']['power']
    reference = read(SIZES)['calibration_of_size_law']['reference_F_support_radius_native']
    radius = Radius()
    m = NodalSF6(workers=5)
    try:
        with np.load(STATE) as s:
            n,h,r,d,O = (s[k] for k in ('fields','h_native','radii_native','directions','frames'))
            m.model.geometry.d['r0'] = float(s['contact_intercept_native'])
        n = n.copy()
        i = PERTURBED_SITE
        spacing = float(h[i])
        core_xyz = nodes(CORE,spacing).reshape(-1,3)
        bump = np.maximum(1.-np.sum((core_xyz-[.32,-.19,.28])**2,axis=1)/1.2**2,0.)**3
        n[i,:,2] += .025*bump
        n[i] /= np.linalg.norm(n[i],axis=1)[:,None]
        m.mask = np.ones(CORE**3,dtype=bool)
        _,g = m.evaluate(n,True,r=r,D=d,frames=O,pose=True)
        atom = m.m.action.action
        _,g_self_core = atom(n[i],spacing)
        contact = g[i]-core_projection(n[i],g_self_core)
        kin = NodalKinetic(m.model.common['beta'])
        base = pad_cube(n[i],CORE,PADDING,VACUUM)
        core_mask = np.linalg.norm(core_xyz,axis=1)<1.45
        # Exact matching between a core node and the corresponding padded node.
        inside = (slice(PADDING,PADDING+CORE),)*3
        indices = np.arange(GRID**3).reshape((GRID,)*3)[inside].ravel()[core_mask]
        R0 = radius(base,spacing)[0]
        core_size = radius(n[i],spacing)[0]
        A0,_,_ = kin.mass(base,spacing)
        U0,G0 = atom(base,spacing)
        F0 = core_projection(base[indices],G0[indices])+contact[core_mask]
        K0 = A0[indices]
        field0 = base[indices]
        inertia0 = (R0/reference)**p
        a0 = -np.einsum('nij,nj->ni',constrained_inverse(K0,field0),F0)/inertia0
        if max(np.linalg.norm(a0,axis=1))<1e-5:
            raise ArithmeticError('The chosen core probe was effectively at rest')
        del A0,G0
        rows = []
        for center in CENTERS_X:
            for amp in AMPLITUDES:
                field,support = halo_patch(base,center,amp)
                R = radius(field,spacing)[0]
                U,G = atom(field,spacing)
                A,_,_ = kin.mass(field,spacing)
                F = core_projection(field[indices],G[indices])+contact[core_mask]
                a = -np.einsum('nij,nj->ni',constrained_inverse(A[indices],field[indices]),F)/(R/reference)**p
                core_R = radius(crop(field,CORE),spacing)[0]
                coefficient = float(np.sum(a*a0)/np.sum(a0*a0))
                ratio = (R0/R)**p
                error = np.max(abs(a-ratio*a0))
                row = {'x_center_native':center*spacing,'support_nodes':int(support.sum()),'amplitude':amp,
                       'min_pulse_field_scalar':float(min(field[support,0])),
                       'outside_self_energy_increase_native':float(U-U0),
                       'R_full_native':R,'R_core_native':core_R,
                       'core_bare_mass_matrix_max_defect':np.max(abs(A[indices]-K0)),
                       'core_force_max_defect':np.max(abs(F-F0)),
                       'core_RMS_difference_native':abs(core_R-core_size),
                       'instantaneous_core_acceleration_ratio':coefficient,
                       'ratio_from_size_law_alone':ratio,'acceleration_scaling_defect_inf':error}
                if (row['core_force_max_defect']>2e-12 or row['core_bare_mass_matrix_max_defect']>2e-12
                        or row['core_RMS_difference_native']>2e-12 or error>2e-12):
                    raise ArithmeticError(f'A local stencil or core was changed by the exterior patch: {row}')
                rows.append(row)
                print('locality',center,amp,'R',R,'core acceleration ratio',coefficient,flush=True)
        summary = {'revision':357,'full':True,'physical_validation':False,'site':i,'grid':GRID,
                   'scope':'instantaneous zero-velocity core field acceleration, rigid frame and sites held, finite gnomonic stencil',
                   'native_frame_force_and_contact_potential_kept':True,
                   'undeformed_halo_baseline_R_native':R0,'reference_core_R_native':core_size,
                   'baseline_core_max_acceleration_native':max(np.linalg.norm(a0,axis=1)),
                   'fixed_cut_variant_has_invariant_radius_on_the_tested_exterior_perturbations':True,
                   'fixed_cut_variant_not_used_to_tune_or_reinterpret_the_dynamical_runs':True,
                   'core_acceleration_sample_node_count':int(core_mask.sum()),'rows':rows,
                   'cut_counts':CUT_COUNTS,
                   'size_vs_cut_by_site':{'SF4':plateau(FINAL,radius,p),'SF6':plateau(STATE,radius,p)},
                   'input_sha256':{str(q.relative_to(ROOT)):digest(q) for q in (SOURCE,FINAL,STATE,SIZES,
                         ROOT/'scripts'/'derivee_taille_R357.c',ROOT/'scripts'/'taille_dynamique_R357.py')}}
        dump(OUTPUT/'PORTEE_SPATIALE_TAILLE_R357.json',summary)
    finally:
        m.close()


if __name__=='__main__':
    main()
