'Independently test a nodal descent outside the relaxed finite SF6 chart.'

from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from trials_nodal import digest
from modes_radial import CORRECTED
from match_contact import TransferredContact
from relax_sf6 import OUTPUT,ROOT,SF6
import numpy as np


def field_energy(model,fields,r,d):
    m = model.m
    own = [m.action.action(field,m.h(i)) for i,field in enumerate(fields)]
    diff = d@d.T-model.geometry.target
    U = sum(v[0] for v in own)+.5*model.geometry.weight*np.sum(diff**2)
    for j,D in enumerate(d,1):
        df = -m.frames[j-1].T@D
        p = m.port(fields[0],0,D)
        q = m.port(fields[j],j,df)
        CS = m.curvature(fields[0],0,p['p'],own[0])
        CF = m.curvature(fields[j],j,q['p'],own[j])
        k = m.factor*(CS['absolute']+CF['absolute']+m.pair_curvature)
        delta = r[j-1]-float(model.geometry.d['r0'])-p['value']-q['value']
        U += model.well.separation_energy_derivative(r[j-1])[0]+.5*k*delta**2
    return U


def analysis(model,case,path):
    m = model.m
    z = read(path)['final']
    x,r,d = [np.asarray(z[k]) for k in ('coordinates','radii_native','directions')]
    port,curvature = m.port,m.curvature
    ports,curvatures = [],[]

    def capture_port(*args):
        out = port(*args)
        ports.append(out['gradient_field'])
        return out

    def capture_curvature(*args):
        out = curvature(*args)
        curvatures.append(out['gradient_field'])
        return out

    m.port,m.curvature = capture_port,capture_curvature
    try:
        E,g = m.evaluate(x)
    finally:
        m.port,m.curvature = port,curvature
    if len(ports)!=12 or len(curvatures)!=12:
        raise ValueError('Unexpected contact derivatives for six bonds')
    raw = np.array([v[1] for v in m.action.rows])
    for j,s in enumerate(m.last['contacts']):
        rho = -s['kappa']*s['delta']
        sigma = .5*s['delta']**2*m.factor
        for i,k in ((0,2*j),(j+1,2*j+1)):
            raw[i] += rho*ports[k]+sigma*curvatures[k]
    pull = np.concatenate([m.state.pull(i,gradient) for i,gradient in enumerate(raw)])
    defect = float(np.max(abs(pull-g[:model.n])))
    if defect>2e-9:
        raise ArithmeticError(f'The nodal gradient did not contract to the received chart: {defect}')
    n = np.array(m.state.fields)
    norms = np.sum(n*n,axis=2)
    tangent = raw-n*np.sum(n*raw,axis=2)[:,:,None]/norms[:,:,None]
    tangent[:,~model.geometry.trial.mask] = 0.
    size = float(np.linalg.norm(tangent))
    v = -tangent/size
    expected = float(np.sum(raw*v))
    baseline = field_energy(model,n,r,d)
    energy_defect = float(abs(baseline-E))
    if energy_defect>2e-9:
        raise ArithmeticError('Independent field energy did not replay')
    magnitude = np.linalg.norm(v,axis=2)
    results = []
    for eps in (2e-5,1e-5):
        n1 = n*np.cos(eps*magnitude)[:,:,None]
        shift = eps*v*np.sinc(eps*magnitude/np.pi)[:,:,None]
        a = field_energy(model,n1+shift,r,d)
        b = field_energy(model,n1-shift,r,d)
        actual = float((a-b)/(2*eps))
        results.append({'step':eps,'derivative':actual,'defect':abs(actual-expected),
                        'energy_change_in_descent_direction':float(a-baseline)})
    if results[-1]['defect']>1e-5*max(1.,abs(expected)) or results[-1]['energy_change_in_descent_direction']>=0.:
        raise ArithmeticError('The nodal descent did not confirm')
    print(case,'nodal tangent norm',size,'chart gradient',float(max(abs(g[:model.n]))),flush=True)
    return {'finite_chart_field_gradient_inf':float(max(abs(g[:model.n]))),
            'full_nodal_field_gradient_l2':size,'full_nodal_field_gradient_inf':float(np.max(abs(tangent))),
            'full_nodal_field_gradient_l2_per_site':np.linalg.norm(tangent.reshape(7,-1),axis=1),
            'field_projection_to_chart_gradient_defect':defect,'independent_energy_replay_defect':energy_defect,
            'descent_directional_derivative':expected,'finite_differences':results}


def main():
    held = OUTPUT/'CHAMPS_ET_ANGLES_LIBRES_R353.json'
    result = {'frozen_R352':analysis(SF6(),'frozen_R352',held),
              'recalibrated':analysis(TransferredContact(),'recalibrated',CORRECTED)}
    files = (Path(__file__).resolve(),held,CORRECTED,ROOT/'scripts'/'contacts_reactifs_R338.py',
             ROOT/'scripts'/'relaxer_SF6_R353.py',ROOT/'scripts'/'raccorder_contact_R353.py')
    dump(OUTPUT/'FORCES_HORS_CARTE_R353.json',{'revision':353,'full':True,'physical_validation':False,
        'scope':'all interior nodal directions tangent to the seven S3 fields with radii, angular geometry and frames held',
        'cases':result,'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in files}})


if __name__=='__main__':
    main()
