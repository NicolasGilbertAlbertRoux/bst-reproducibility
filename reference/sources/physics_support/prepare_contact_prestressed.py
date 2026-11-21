'Exclude an entire zero-extension stationary family and replay a prestressed seed.'

from pathlib import Path

from sweep_sf4 import AMPLITUDES, LENGTH_TARGETS, LENGTH_UNC, OUTPUT as R348, SCALES, RadialSlice, read
from fields_and_references import dump
from test_fit import AdjustedFrames
from trials_nodal import digest
from clock_inertia import OUTPUT, ROOT, all_orientations_bound
from free_rotations import local_matrices, pose_coordinate_map
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq


def hermite(lo,hi,u,v,p,q):
    t = Polynomial([0.,1./(hi-lo)])
    return (u*(2*t**3-3*t**2+1)+(hi-lo)*p*(t**3-2*t**2+t)
            +v*(-2*t**3+3*t**2)+(hi-lo)*q*(t**3-t**2))


def segments(well):
    x,y,s = well.x,well.y,well.slopes
    for i in range(len(x)-1):
        if x[i+1]<=well.left_anchor+1e-14 or x[i]>=well.right_anchor-1e-14:
            yield x[i],x[i+1],hermite(x[i],x[i+1],y[i],y[i+1],s[i],s[i+1])
    i = well.left_anchor_index
    v,q = well.local(well.local_lower)
    yield well.left_anchor,well.local_lower,hermite(well.left_anchor,well.local_lower,y[i],v,s[i],q)
    yield well.local_lower,well.local_upper,Polynomial(well.coefficients[::-1])(
        Polynomial([well.local_lower-well.origin,1.]))
    i = well.right_anchor_index
    u,p = well.local(well.local_upper)
    yield well.local_upper,well.right_anchor,hermite(well.local_upper,well.right_anchor,u,y[i],p,s[i])


def derivative_envelope(well,lower,upper):
    scale = well.radius_sum
    records = []
    for a,b,U in sorted(segments(well),key=lambda s:s[0]):
        l,r = max(a,lower/scale),min(b,upper/scale)
        if l>=r:
            continue
        derivative = U.deriv()/scale
        tests = [l-a,r-a,*[float(z.real) for z in derivative.deriv().roots()
                            if abs(z.imag)<1e-12 and l-a<z.real<r-a]]
        values = derivative(np.array(tests))
        sample = np.linspace(l,r,5)
        mismatch = max(abs(derivative(f-a)-well.separation_energy_derivative(f*scale)[1]) for f in sample)
        if mismatch>2e-12:
            raise ArithmeticError(f'Native well derivative replay defect {mismatch}')
        records.append({'radius':[l*scale,r*scale],'derivative_min':float(min(values)),
                       'derivative_max':float(max(values)),'value_replay_defect':float(mismatch)})
    endpoints = [records[0]['radius'][0],records[-1]['radius'][-1]]
    gaps = [abs(a['radius'][1]-b['radius'][0]) for a,b in zip(records[:-1],records[1:],strict=True)]
    if np.max(abs(np.array(endpoints)-[lower,upper]))>2e-12 or max(gaps,default=0.)>2e-12:
        raise ArithmeticError('Derivative envelope missed a portion of the requested radius range')
    return records


def bare_curvature(well,r):
    eps = 5e-5
    return (well.separation_energy_derivative(r+eps)[1]-well.separation_energy_derivative(r-eps)[1])/(2*eps)


def preparation(slice_,hit,clock):
    r = np.array([b['radius'] for b in hit['selected_brackets']])
    q,force = [],[]
    beta,gamma = hit['beta'],hit['gamma']
    for j,row in enumerate(hit['selected_brackets']):
        S = slice_.sulphur(j)
        def f(a):
            return slice_.force(j,r[j],beta,gamma,S,slice_.fluor(j,a))[0]
        index = row['interval_index']
        root = brentq(f,*AMPLITUDES[index:index+2],xtol=3e-13)
        q.append(root)
        force.append(f(root))
    scale = hit['scale_native_per_angstrom']
    candidate = {'label':'precontraint_convexe_R350','beta':beta,'gamma':gamma,
        'kangle':float(slice_.configuration.kangle),'F_preparations':slice_.qF+np.array(q)[:,None]*slice_.fpaths,
        'center_chart':slice_.center_chart(r),'scale_native_per_angstrom':scale}
    model = AdjustedFrames(48,candidate)
    try:
        n,_,x,_ = model.unpack(model.adjusted_seed)
        _,_,gradient,_ = model.potential(n,x)
        contacts = model.received.c.model.last['contacts']
        extensions = np.array([a['delta'] for a in contacts])
        W = slice_.model.m.trial.well
        own_bare_curvatures = np.array([bare_curvature(W,x) for x in r])
        radial_native = np.array([a['radial_force'] for a in contacts])
        geo,G,_ = local_matrices(model,model.adjusted_seed,False)
        orientation = pose_coordinate_map(geo)[6:,6:].T@gradient
        length_lower = float(max((LENGTH_TARGETS-LENGTH_UNC)/r))*1e-10
        bound = all_orientations_bound({'count':48,'centers':geo[0],'native_blocks':G},
                                      length_lower*(1.-1e-10),clock['formula_isotopic_mass_kg'])
        diagnostic = {'candidate':candidate,'sampled_cell_indices':hit['indices'],'F_chart_amplitudes':q,
            'radii_native':r,'lengths_angstrom':r/scale,'radial_solver_residuals':force,
            'native_radial_force_derivative':radial_native,'contact_extensions_native':extensions,
            'bare_W_radial_curvatures_native':own_bare_curvatures,
            'orientation_force_covectors_native':orientation,'all_orientations_bound':bound,
            'full_equilibrium_claimed':False}
        if max(abs(radial_native))>2e-7 or min(own_bare_curvatures)<=0:
            raise ArithmeticError('Expected a positive-curvature radial stationary seed')
        return diagnostic
    finally:
        model.close()


def main():
    scan = R348/'BALAYAGE_RECUE_R348.json'
    s = RadialSlice('recue')
    W = s.model.m.trial.well
    lower,upper = float(SCALES[0]*(LENGTH_TARGETS[2]-LENGTH_UNC)),float(SCALES[-1]*(LENGTH_TARGETS[2]+LENGTH_UNC))
    bands = derivative_envelope(W,lower,upper)
    upper_force = max(p['derivative_max'] for p in bands)
    if upper_force>=0:
        raise ArithmeticError('Equatorial zero-extension stationarity was not excluded')
    acceptable = []
    for hit in read(scan)['bracket_cells']:
        if hit['indices'][:2]==[2,3] and hit['samples_at_nominal_measurement']==4:
            curvature = min(bare_curvature(W,row['radius']) for row in hit['selected_brackets'])
            if curvature>0:
                acceptable.append((curvature,hit))
    chosen = max(acceptable,key=lambda a:a[0])[1]
    clock_path = OUTPUT/'HORLOGE_ET_INERTIE_ABSOLUE_R350.json'
    seed = preparation(s,chosen,read(clock_path))
    report = {'revision':350,'full':True,'R348_scale_window':list(map(float,SCALES[[0,-1]])),
              'equatorial_radius_range_native':[lower,upper],'equatorial_W_derivative_bounds':bands,
              'global_upper_bound_on_equatorial_W_derivative':upper_force,
              'zero_extension_radial_stationarity_excluded_throughout_window':True,
              'received_laws_bracket_cells_with_positive_bare_curvature_and_four_central_lengths':len(acceptable),
              'prestressed_seed':seed,'physical_validation':False,
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (scan,clock_path,Path(__file__).resolve(),
                  ROOT/'sources'/'R321'/'SOURCE_R5.py',ROOT/'sources'/'R321'/'PUITS_SF_R319.npz')}}
    dump(OUTPUT/'CONTACT_PRECONTRAINT_R350.json',report)
    print('zero-extension global upper derivative',upper_force,'positive curvature cells',len(acceptable),
          'selected scale',seed['candidate']['scale_native_per_angstrom'],
          'orientation residual',max(abs(seed['orientation_force_covectors_native'])),
          'ABC lower deficits',seed['all_orientations_bound']['absolute_minimum_relative_shortfall'],flush=True)


if __name__=='__main__':
    main()
