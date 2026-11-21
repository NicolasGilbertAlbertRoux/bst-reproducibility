'Differentiate the same continuous transported textures before quadrature.'

from pathlib import Path

from sweep_sf4 import read
from sweep_beta import base_fields
from fit_topological import choose_weight
from fields_and_references import dump
from dynamics_nodal_scripts import NodalDynamics
from trials_nodal import digest
from inertia_and_entrainment import BANK, REF, SF6_REF, Rotor, search, spin_free_mass
from refit_dependencies import State
from free_rotations import CALIBRATION, ROOT
from sizes_substrate import OUTPUT, SHAPES
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import map_coordinates
from scipy.stats import qmc


def samples(gauss,order):
    x,w = leggauss(gauss)
    points = (np.arange(47)[:,None]+.5*(x+1)).ravel()
    weights = np.tile(.5*w,47)
    eye = np.eye(48)

    def interpolation(nodes):
        return np.column_stack([map_coordinates(c,[nodes],order=order,mode='nearest') for c in eye])

    def derivative(eps):
        return (interpolation(points-2*eps)-8*interpolation(points-eps)
                +8*interpolation(points+eps)-interpolation(points+2*eps))/(12*eps)

    values = interpolation(points)
    slope = derivative(.001)
    defect = float(max(np.max(abs(slope-derivative(.0005))),np.max(abs(slope.sum(axis=1))),
                       np.max(abs(values.sum(axis=1)-1))))
    if defect>2e-9:
        raise ArithmeticError(f'Interpolation derivative not qualified: {defect}')
    return points,weights,values,slope,defect


def along(field,matrix,axis):
    plane = np.moveaxis(field,axis,0)
    shape = (matrix.shape[0],)+plane.shape[1:]
    return np.moveaxis((matrix@plane.reshape(plane.shape[0],-1)).reshape(shape),0,axis)


def weighted_matrices(field,h,beta,scheme):
    points,weights,mat,grad,defect = scheme
    coords = h*(points-23.5)
    raw = field.reshape(48,48,48,4)
    z = along(raw,mat,2)
    yz = along(z,mat,1)
    dy = along(z,grad,1)
    dz = along(along(raw,grad,2),mat,1)
    out = np.zeros((2,6,6))
    minimum_norm = float('inf')
    for k in range(0,len(points),7):
        sl = slice(k,k+7)
        value = along(yz,mat[sl],0).reshape(-1,4)
        norm = np.linalg.norm(value,axis=1)
        minimum_norm = min(minimum_norm,float(min(norm)))
        if minimum_norm<.2:
            raise ArithmeticError('Continuous interpolation is outside the normalized chart')
        field = value/norm[:,None]
        ds = np.stack([x.reshape(-1,4) for x in (along(yz,grad[sl],0),along(dy,mat[sl],0),along(dz,mat[sl],0))],axis=1)
        ds -= field[:,None,:]*np.einsum('ni,nai->na',field,ds)[:,:,None]
        ds /= (h*norm)[:,None,None]
        pos = np.stack(np.meshgrid(coords[sl],coords,coords,indexing='ij'),axis=-1).reshape(-1,3)
        vol = (h**3*(weights[sl,None,None]*weights[None,:,None]*weights[None,None,:])).ravel()
        rho = np.linalg.det(np.concatenate((field[:,None,:],ds),axis=1))
        v = np.empty((len(ds),6,4))
        v[:,3:] = -ds
        for a,e in enumerate(np.eye(3)):
            v[:,a] = -np.einsum('ni,nic->nc',np.cross(e,pos),ds)
        dv = np.einsum('nac,nic->nai',v,ds)
        g = np.sum(ds*ds,axis=(1,2))
        for j,mass_weight in enumerate((vol,vol*rho**2)):
            left = (v*np.sqrt(mass_weight*(1.+beta*g))[:,None,None]).transpose(1,0,2).reshape(6,-1)
            right = (dv*np.sqrt(mass_weight*beta)[:,None,None]).transpose(1,0,2).reshape(6,-1)
            out[j] += left@left.T-right@right.T
    return out,minimum_norm


def integrate(fields,spacing,beta,scheme):
    weighted,norm = zip(*[weighted_matrices(a,h,beta,scheme) for a,h in zip(fields,spacing,strict=True)],strict=True)
    a = np.array(weighted)
    return a[:,0],a[:,1],float(min(norm))


def main():
    rows = read(BANK)['rows']
    r = next(row for row in rows if row['preparation']==SHAPES[0] and row['count']==48)
    six = next(row for row in rows if row['species']=='SF6')
    c = read(CALIBRATION)['candidate']
    source = read(OUTPUT/'RESOLUTION_TOPOLOGIQUE_R351.json')
    initial_angles = np.asarray(source['N80_twists'])
    model = NodalDynamics(48)
    fields = base_fields(model,c,model.unpack(model.project(model.initial())[0])[0])
    common = None
    results = {}
    rules = [('cubic_Gauss2',2,3),('cubic_Gauss3',3,3),('linear_Gauss2',2,1)]
    for name,gauss,order in rules:
        scheme = samples(gauss,order)
        g0,extra,minimum = integrate(fields,r['h_native'],c['beta'],scheme)
        if common is None:
            common = choose_weight(g0,extra)
            points = (2*qmc.Sobol(4,scramble=True,seed=3351).random_base2(11)-1)*np.pi
            model0 = Rotor(r,g0+common*extra,r['length_scale_native_per_angstrom'],REF)
            angles,_ = search(model0,initial_angles,points)
        D = g0+common*extra
        rotor = Rotor(r,D,r['length_scale_native_per_angstrom'],REF)
        measurement = rotor.read(angles)
        mass_ratio = spin_free_mass(D[0])/spin_free_mass(D[1])
        results[name] = {'gauss_per_axis_per_parent_cell':gauss,'field_spline_order':order,
                         'analytic_cubic_derivative_crosscheck_defect':scheme[-1],'minimum_interpolant_norm':minimum,
                         'S_to_F1_mass_ratio':mass_ratio,'rotational_reading':measurement,
                         'native_metric':D,'lambda_from_refitting_only_as_diagnostic':choose_weight(g0,extra)}
        print(name,'common lambda',common,'mass ratio',mass_ratio,'ABC max',
              max(abs(measurement['relative_errors_if_isotropic_translation'])),flush=True)
    state6 = State('SF6')
    scheme = samples(2,3)
    g0,extra,_ = integrate(state6.fields,six['h_native'],c['beta'],scheme)
    rad = np.linalg.norm(np.asarray(six['centers'])[1])
    reserved = Rotor(six,g0+common*extra,rad/1.5607,np.full(3,SF6_REF)).read(np.zeros(6))
    dump(OUTPUT/'QUADRATURE_DENSITE_R351.json',{'revision':351,'full':True,
         'lambda_fitted_only_on_cubic_Gauss2_by_isotope_mass':common,'torsions_fitted_only_on_cubic_Gauss2':angles,
         'results':results,'SF6_reserved_at_measured_length':reserved,'physical_validation':False,
         'scope':'analytic derivative of normalized tensor-product interpolants of the same saved shapes; no new equilibrium or discretized field evolution',
         'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (BANK,CALIBRATION,
                         OUTPUT/'RESOLUTION_TOPOLOGIQUE_R351.json',Path(__file__).resolve())}})
    print('conditional SF6 without retuning',reserved['ABC_cm_inverse_if_isotropic_translation'],flush=True)


if __name__=='__main__':
    main()
