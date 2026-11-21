'Hot equations of the triad and acoustic projection with fixed parameters.'
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, least_squares
from scipy.stats import chi2 as chi2_distribution

from tensor_R449 import tensor_diagnostics


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def densities(reference):
    photon = reference["omega_gamma_h2"]
    one_neutrino = (7/8)*(4/11)**(4/3)*photon
    radiation = photon + reference["N_eff"]*one_neutrino
    return radiation, one_neutrino


def triad(p, radiation, tolerance=2e-11, amin=1e-8):
    'Evolve the comobile field and ja2K E and integrate energy.'
    omega_m, g, alpha = (p[k] for k in ("omega_m", "g", "alpha"))
    phi0 = p["phi0"]
    initial_j2 = 2*(1-omega_m-radiation) - g*g*phi0**4
    if initial_j2 <= 0:
        raise ValueError('Initial non-positive electrical energy.')
    j0 = p["sgn"]*np.sqrt(initial_j2)
    # Integrate U=j/a² when the electric energy remains well inside the quartic
    # regime. In that family U remains O(1) even toward the radiation era.
    use_electric_variable = 'density_magnetic_prescribed' in p

    def rhs(n, y):
        phi, momentum, _, _ = y
        a = np.exp(n)
        j = momentum*a*a if use_electric_variable else momentum
        quartic = alpha*(phi/a)**4
        response = 1 + quartic
        a4h2 = radiation + omega_m*a + .5*(j*j + g*g*phi**4)
        conformal_step = a/np.sqrt(a4h2)
        field_step = conformal_step/np.sqrt(response)
        gauge_transfer = 2*(quartic/response)*j
        magnetic_force = 2*g*g*phi**3*field_step
        derivative = -(2*momentum/response + magnetic_force/a**2) if use_electric_variable else gauge_transfer-magnetic_force
        return (j*field_step, derivative,
                -conformal_step, gauge_transfer*j)

    solution = solve_ivp(rhs, (0., np.log(amin)), [phi0,j0,0.,0.], method="DOP853",
                         atol=tolerance*.02, rtol=tolerance, max_step=.09, dense_output=True)
    if not solution.success:
        raise ArithmeticError(solution.message)
    grid = np.linspace(0., np.log(amin), 800)
    phi, momentum, _, energy_flow = solution.sol(grid)
    j = momentum*np.exp(2*grid) if use_electric_variable else momentum
    energy = .5*(j*j+g*g*phi**4)
    defect = float(np.max(np.abs(energy-energy_flow-energy[0])))

    def state(a):
        a = max(a,amin)
        phi,momentum,distance,_ = solution.sol(np.log(a))
        j = momentum*a*a if use_electric_variable else momentum
        energy = .5*(j*j+g*g*phi**4)
        return float(phi), float(j), float(energy), float(distance)

    def inverse(a):
        _, _, energy, _ = state(a)
        return 1/np.sqrt(radiation+omega_m*a+energy)

    return inverse, lambda z: state(1/(1+z))[-1], state, defect


def standard(matter, radiation):
    constant = 1-matter-radiation
    inverse = lambda a: 1/np.sqrt(radiation+matter*a+constant*a**4)
    distance = lambda z: quad(inverse,1/(1+z),1.,epsabs=5e-12,epsrel=5e-12)[0]
    return inverse,distance


def horizon(inverse, z, baryon_loading):
    return quad(lambda a: inverse(a)/np.sqrt(3*(1+baryon_loading*a)),
                0.,1/(1+z),epsabs=2e-11,epsrel=1e-9,limit=180)[0]


def acoustic(inverse, distance, ref):
    baryon_loading = .75*ref["omega_b_h2"]/ref["omega_gamma_h2"]
    star = ref["z_star"]
    rs = horizon(inverse,star,baryon_loading)
    rd = horizon(inverse,ref["z_drag"],baryon_loading)
    causal = quad(inverse,0.,1/(1+star),epsabs=2e-11,epsrel=1e-9,limit=180)[0]
    dm_star = distance(star)
    theta = rs/dm_star
    return dict(rstar_H0_over_c=rs,rd_H0_over_c=rd,Dstar_H0_over_c=dm_star,
                theta100=100*theta,theta100_si_c=100*causal/dm_star,
                ell_acoustic=np.pi/theta,ell_acoustic_min_si_speed_c=np.pi*dm_star/causal,
                ratio_rd_rstar=rd/rs)


def read_bao(mean_path, covariance_path):
    data = np.genfromtxt(mean_path, dtype=None, encoding="utf-8")
    covariance = np.loadtxt(covariance_path)
    assert len(data) == 13 and covariance.shape == (13,13)
    np.testing.assert_array_equal(covariance,covariance.T)
    np.linalg.cholesky(covariance)
    return data,covariance


def bao_vector(data, inverse, distance):
    pred = []
    for z, _, mode in data:
        a = 1/(1+z)
        radial, transverse = a*a*inverse(a), distance(z)
        pred.append(dict(DM_over_rs=transverse,DH_over_rs=radial,
                         DV_over_rs=(z*transverse*transverse*radial)**(1/3))[mode])
    return np.array(pred)


def whiten(covariance, vector):
    return solve_triangular(np.linalg.cholesky(covariance),vector,lower=True)


def chi2(covariance, vector):
    norm = whiten(covariance, vector)
    return float(norm@norm)


def bao_report(data, covariance, prediction, rd):
    observed = data["f1"]
    train = np.flatnonzero(data["f0"] < 1.)
    test = np.flatnonzero(data["f0"] >= 1.)
    assert len(train) == 7 and len(test) == 6
    assert not np.any(covariance[np.ix_(train,test)])
    tr_cov,te_cov = covariance[np.ix_(train,train)],covariance[np.ix_(test,test)]
    x,y = whiten(tr_cov,prediction[train]),whiten(tr_cov,observed[train])
    scale = float((x@y)/(x@x))
    native, effective = prediction/rd, prediction*scale

    def score(p):
        residual = p-observed
        return dict(chi2_total=chi2(covariance,residual),chi2_z_lower_1=chi2(tr_cov,residual[train]),
                    chi2_z_upper_or_equal_1=chi2(te_cov,residual[test]),readouts=p.tolist())

    return dict(H0_rd_over_c_native=rd,H0_rd_over_c_effective=1/scale,
                horizon_native_over_effective=rd*scale,with_horizon_native=score(native),
                with_horizon_fitted_over_z_lower_1=score(effective))


def peak_test(peaks, ell_acoustic):
    orders, ells, sigmas = np.asarray(peaks).T
    np.testing.assert_array_equal(np.diff(orders),1)
    gaps = np.diff(ells)
    offset = ells - ell_acoustic*orders
    # A constant phase or an integer change of the first peak leaves the range unchanged.
    return dict(indices=orders.astype(int).tolist(),positions=ells.tolist(),errors_marginal=sigmas.tolist(),
                deviations_successive=gaps.tolist(),deviation_par_mode_between_extremes=float(np.mean(gaps)),
                correction_minimax_in_multipoles=float(.5*np.ptp(offset)),
                excursion_phase_min_in_rad=float(np.pi*np.ptp(offset)/ell_acoustic))


def characteristics(electric, magnetic, kappa):
    "Eliminate the longitudinal electrical component via Gauss's law."
    x, y = kappa*magnetic**2, kappa*electric**2
    kinetic = (1+3*x)/(1+x)
    gradient = (1+x-2*y)/(1+x)
    mixing = 2*kappa*magnetic*electric/(1+x)
    return kinetic,gradient,mixing,gradient/kinetic


def check_characteristics():
    # Compare with a direct second difference of the local field Lagrangian.
    # Impose Gauss through the Schur complement of that numerical Hessian.
    worst=0.
    basis=np.eye(5)
    step=3e-4
    for e,b,k in ((.3,.4,0.),(.5,.7,.9),(.8,.1,40.),(-.6,.9,4.),(.14,.07,3e4)):
        def action(v):
            dx,dy,sx,sy,long=v/np.sqrt(2.)
            delta_e=np.array([[dx,dy,0.],[-dy,dx,0.],[0.,0.,np.sqrt(2.)*long]])
            delta_b=np.array([[sy,-sx,0.],[sx,sy,0.],[0.,0.,0.]])
            electric=e*np.eye(3)+delta_e
            magnetic=b*np.eye(3)+delta_b
            lagrangian=.5*(np.sum(electric**2-magnetic**2)+k*np.sum(electric*magnetic)**2)
            # At frozen background, this term is a total derivative.
            return lagrangian-k*(3*e*b)*float(np.sum(delta_e*delta_b))

        def entry(i,j):
            u,v=step*basis[i],step*basis[j]
            return (action(u+v)+action(-u-v)-action(u-v)-action(-u+v))/(4*step**2)

        full=np.array([[entry(i,j) for j in range(5)] for i in range(5)])
        reduced=full[:-1,:-1]-np.outer(full[:-1,-1],full[-1,:-1])/full[-1,-1]
        kinetic,grad,mixing,c2=characteristics(e,b,k)
        analytic=np.array([[kinetic,0.,0.,mixing],[0.,1.,0.,0.],[0.,0.,-1.,0.],[mixing,0.,0.,-grad]])
        np.testing.assert_allclose(reduced,analytic,rtol=4e-6,atol=5e-6)
        # Rebuild symbol in velocity v and test both independently calculated roots v².
        poly=[reduced[0,0]*reduced[1,1],reduced[0,0]*reduced[3,3]+reduced[1,1]*reduced[2,2]-reduced[0,3]**2,
              reduced[2,2]*reduced[3,3]]
        roots=np.sort(np.roots(poly).real)
        expected=np.sort([1.,c2])
        np.testing.assert_allclose(roots,expected,rtol=8e-6,atol=6e-6)
        worst=max(worst,float(np.max(np.abs(reduced-analytic))))
    return worst


def stability(state,p,z_star):
    local_kappa=p["alpha"]/(3*p["g"]**2)

    def speed_squared(z):
        a=1/(1+z)
        phi,j,_,_=state(a)
        magnetic=p["g"]*(phi/a)**2
        electric=j/(a*a*np.sqrt(1+p["alpha"]*(phi/a)**4))
        return characteristics(electric,magnetic,local_kappa)[-1]

    z_scan=np.expm1(np.linspace(0.,np.log1p(1e5),1400))
    values=np.asarray([speed_squared(z) for z in z_scan])
    flips=np.flatnonzero(np.signbit(values[1:])!=np.signbit(values[:-1]))
    roots=[brentq(speed_squared,z_scan[i],z_scan[i+1]) for i in flips]
    return dict(scope='Local main symbol of the homogeneous triad, frozen background and large waves. '
                      'The law of Gauss is imposed. The calculated propagating sector has two branches, one to c. '
                      'A negative velocity square gives gradient instability in this local action. '
                      'The growth rate at k finite and a possible cut-off of the actual model are separate issues.',
                c2_over_c2_present=speed_squared(0.),c2_over_c2_a_zstar=speed_squared(z_star),
                min_sampled=float(values.min()),max_sampled=float(values.max()),
                z_of_min_sampled=float(z_scan[values.argmin()]),changes_sign_z=roots,
                points={str(z):speed_squared(z) for z in (0,.5,1,2,5,10,100,1100,1e5)})


def evaluate(p, radiation, data, covariance, refs, sn, tolerance=2e-11, amin=1e-8):
    ref = refs['readout_baryon_photon']
    if p['name'].startswith("ΛCDM"):
        inverse, distance = standard(p["omega_m"], radiation)
        state = None
        defect = 0.
    else:
        inverse,distance,state,defect = triad(p,radiation,tolerance,amin)
    assert defect < 1e-8
    geometry = acoustic(inverse,distance,ref)
    peaks = refs['planck_2015_peaks']
    bao = bao_report(data,covariance,bao_vector(data,inverse,distance),geometry["rd_H0_sur_c"])
    of = refs["DES_Y6"]
    dm_rd = distance(of["z"])/geometry["rd_H0_sur_c"]
    dm_rd_free = distance(of["z"])/bao['H0_rd_sur_c_effective']
    sn_z = np.asarray(sn["redshifts"])
    inv_h = np.array([(1/(1+z))**2*inverse(1/(1+z)) for z in sn_z])
    native_energy = None
    local_stability = None
    if state is not None:
        _,_,energy_early,_ = state(0.)
        z_star = ref["z_star"]
        _,_,energy_at_star,_ = state(1/(1+z_star))
        _,one_neutrino = densities(ref)
        native_energy = dict(a4_omega_triade_asymptotic=energy_early,
                             a4_omega_triade_a_zstar=energy_at_star,
                             fraction_amplitude_h2_a_zstar=energy_at_star*inverse(1/(1+z_star))**2,
                             delta_N_eff_equivalent_background=energy_early*p["h"]**2/one_neutrino)
        local_stability = stability(state,p,z_star)
        local_stability['tensor_sans_reaction_metric']=tensor_diagnostics(state,inverse,p)
    limits = {}
    if p['name'].startswith("R445"):
        # An inverse problem, holding H fixed. These are redshifts required to match only the angle.
        theta_ref=ref["theta100"]/100
        loading=.75*ref["omega_b_h2"]/ref["omega_gamma_h2"]
        limits['z_star_exige_with_cs_standard'] = brentq(
            lambda z: horizon(inverse,z,loading)/distance(z)-theta_ref,10.,ref["z_star"])
        limits["z_star_exige_meme_a_c"] = brentq(
            lambda z: quad(inverse,0.,1/(1+z),epsabs=2e-11)[0]/distance(z)-theta_ref,10.,ref["z_star"])
    return dict(parameters_field=p,omega_r=radiation,acoustic=geometry,BAO_DESI_DR2=bao,
                amplitudes_early=native_energy,defect_summary_integre_a4rho_triade=defect,
                stability_main_local=local_stability,
                peaks={name:peak_test(peaks[name],geometry['ell_acoustic']) for name in ("TT","TE","EE")},
                peaks_a_the_bound_c={name:peak_test(peaks[name],geometry['ell_acoustic_min_si_speed_c'])
                                   for name in ("TT","TE","EE")},
                OF_independent=dict(native_field=dm_rd,effective=dm_rd_free,deviation_sigma_native=(dm_rd-of["DM_sur_rd"])/of["sigma"],
                                     deviation_sigma_effective=(dm_rd_free-of["DM_sur_rd"])/of["sigma"]),
                SN=dict(chi2=chi2(np.asarray(sn["covariance"]),inv_h-np.asarray(sn["E_inverse"])),
                        values_field=inv_h.tolist()),inversions=limits)


def certify_geometry_family(p, radiation, ref, epsilon):
    'Born the electrical/magnetic ratio for any 0a-1.'
    phi0,g,alpha=(p[k] for k in ("phi0","g","alpha"))
    # |dφ/d ln a| <= sqrt(2) a / sqrt(1 + α(φ/a)^4).
    # On φ >= φ0/2, the integral of 4√2 a³/(φ0²√α) bounds the excursion.
    limit=4*np.sqrt(2)/(3*phi0**2*np.sqrt(alpha))
    assert 0 < limit < phi0/2
    phi_low=phi0-limit
    # φ stays positive. Since j'=A j-S with S>0, j stays positive in reverse time.
    # Then . . . In reverse time, Uj/a2 obeys dU/d(-ln a) 2U/K
    # plus 2g²φ³/(a sqrt(a⁴ H²) sqrt(K)). Bound sqrt(a⁴H²) below by gφ²/sqrt(2).
    x_min=alpha*phi_low**4
    j0=np.sqrt(2*(1-p["omega_m"]-radiation-epsilon))
    u_upper=(j0+2*np.sqrt(2)*g*phi0/np.sqrt(x_min))*np.exp(1/(2*x_min))
    # E²/B² <= Umax²/[g²φ_min⁴(1+αφ_min⁴)]. Kinetic c²>= (1-2E²/B²)/3.
    ratio_upper=u_upper**2/(g*g*phi_low**4*(1+x_min))
    assert ratio_upper < .5
    # a⁴Ω_triade = (j² + g²φ⁴)/2. j=a² U bounds its early radiation coefficient.
    a_star=1/(1+ref["z_star"])
    upper_star=epsilon+.5*a_star**4*u_upper**2
    one_neutrino=densities(ref)[1]
    return dict(variation_phi_max=float(limit),phi_min=float(phi_low),phi_max=float(phi0),
                a4_omega_triade_at_cmb_bound_sup=float(upper_star),
                delta_N_eff_equivalent_bound_sup=float(epsilon*p["h"]**2/one_neutrino),
                ratio_e_over_b_at_square_bound_sup=float(ratio_upper),
                c2_over_c2_bound_inf=float((1-2*ratio_upper)/3),
                scope="Comparison walls on the entire past of this family's triad in this action. "
                      'The wave-squad sector is controlled. The transfer with finite wave numbers remains to be evaluated.')


def fit_geometry(epsilon, data, covariance, refs, sn):
    'Set two parameters on seven BAO values and the angular frame of reference, before reserves.'
    ref = refs['readout_baryon_photon']
    radiation_physical,_ = densities(ref)
    train = np.flatnonzero(data["f0"] < 1.)
    training_data = data[train]
    train_cov = covariance[np.ix_(train,train)]
    fields = dict(phi0=.4,sgn=1)

    def model(omega_m,h,radiation):
        name = f"Triade d’essai ε={epsilon:g}" if epsilon else 'CDM, whether or not wedging'
        model_params = dict(name_field=name,omega_m=float(omega_m),h=float(h))
        if epsilon:
            energy_electric = epsilon/4
            energy_quartic = 1-omega_m-radiation-epsilon-energy_electric
            phi0 = fields["phi0"]
            model_params.update(g=float(np.sqrt(2*epsilon)/phi0**2),
                                alpha=float(energy_quartic/(energy_electric*phi0**4)),
                                density_magnetic_prescribed=epsilon,**fields)
        return model_params

    def residual(v):
        omega_m,h = v
        radiation = radiation_physical/h**2
        parameters = model(omega_m,h,radiation)
        if epsilon:
            inverse,distance,*_ = triad(parameters,radiation,tolerance=8e-10,amin=1e-7)
        else:
            inverse,distance = standard(omega_m,radiation)
        geometry = acoustic(inverse,distance,ref)
        angles = (geometry["theta100"]-ref["theta100"])/ref["theta100_marge"]
        bao = bao_vector(training_data,inverse,distance)/geometry["rd_H0_sur_c"]
        return np.r_[whiten(train_cov,bao-training_data["f1"]),angles]

    limits = dict(omega_m=[.15,.45],h=[.45,.95])
    bounds = tuple(zip(limits["omega_m"],limits["h"]))
    trials = [least_squares(residual,start,bounds=bounds,diff_step=3e-5,ftol=3e-9,xtol=3e-9,gtol=3e-9)
              for start in ([.27,.64],[.34,.73])]
    assert all(fit.success for fit in trials)
    np.testing.assert_allclose([x.x for x in trials],np.broadcast_to(trials[0].x,(2,2)),rtol=5e-5)
    best = min(trials,key=lambda trial:trial.cost)
    omega_m,h = best.x
    radiation = float(radiation_physical/h**2)
    coefficients = model(omega_m,h,radiation)
    results = evaluate(coefficients,radiation,data,covariance,refs,sn)
    repeated = evaluate(coefficients,radiation,data,covariance,refs,sn,tolerance=7e-10,amin=3e-7)
    for key in ("theta100","rd_H0_sur_c","Dstar_H0_sur_c"):
        np.testing.assert_allclose(repeated['acoustic'][key],results['acoustic'][key],rtol=2e-6)
    theta_residual = (results['acoustic']["theta100"]-ref["theta100"])/ref["theta100_marge"]
    bao = results["BAO_DESI_DR2"]['with_horizon_native']
    np.testing.assert_allclose(theta_residual**2+bao['chi2_z_lower_1'],2*best.cost,rtol=2e-6)
    bound = float(chi2_distribution.ppf(.95,6))
    prescribed = dict(magnetic=epsilon,electric=epsilon/4) if epsilon else None
    results.update(epsilon=epsilon,
                   setting=dict(chi2_total=float(2*best.cost),deviation_theta_sigma=theta_residual,
                                domaine_omega_m_h=limits,densities_triade_prescribed=prescribed),
                   reserves=dict(chi2_95_pour_six=bound,
                                 DESI_z_upper_or_equal_1=bao['chi2_z_upper_or_equal_1']<bound,
                                 OF_Y6=abs(results['DES_independent']['deviation_sigma_native'])<1.96,
                                 Pantheon_MCT=results["SN"]["chi2"]<bound))
    if epsilon:
        certificate=certify_geometry_family(coefficients,radiation,ref,epsilon)
        assert results['stability_main_local']['min_sampled'] >= certificate['c2_over_c2_bound_inf']
        assert results['amplitudes_early']["a4_omega_triade_a_zstar"] < certificate['a4_omega_triade_at_cmb_bound_sup']
        results['bounds_family']=certificate
    return results


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--entrees",type=Path,default=ROOT/'inputs')
    p.add_argument("--sortie",type=Path,default=ROOT/'results')
    args=p.parse_args()
    source = args.inputs
    refs=read(source/"REPERES_R449.json")
    ref=refs['readout_baryon_photon']
    sn=read(source/"RESERVE_SUPERNOVAE_R448.json")
    parent=read(source/"PROJECTION_ACOUSTIQUE_R448.json")
    frozen=read(source/"FOND_MATIERE_R445.json")
    standard_radiation,_=densities(ref)
    data,covariance=read_bao(source/"DESI_DR2_mean.txt",source/"DESI_DR2_cov.txt")
    reference_case=min((row['choice'] for row in frozen['cases']),key=lambda case:case["chi2_reserve"])
    r445_runs=[]
    for i,prior in enumerate(parent['cases'][:2]):
        radiation=prior["omega_r"]
        omega_m=reference_case["omega_m"]
        phi0=(2*reference_case["q_initial"]*(1-omega_m-radiation))**.25
        model=dict(name_field=f"R445 {'with' if i else 'sans'} rayonnement standard",omega_m=omega_m,
                   h=ref['h_control'],g=1.,alpha=3*reference_case["kappa"],phi0=phi0,sgn=reference_case['sign_electric'])
        outcome=evaluate(model,radiation,data,covariance,refs,sn)
        for key in ("theta100","rd_H0_sur_c","Dstar_H0_sur_c"):
            np.testing.assert_allclose(outcome['acoustic'][key],prior[key],rtol=8e-8)
        r445_runs.append((model,radiation,outcome))
    nominal=r445_runs[0][2]
    np.testing.assert_allclose(nominal["SN"]["chi2"],sn['cases'][0]["chi2"],rtol=5e-8)
    np.testing.assert_allclose(1/nominal["BAO_DESI_DR2"]['H0_rd_sur_c_effective'],reference_case['scale'],rtol=5e-8)
    np.testing.assert_allclose(nominal["BAO_DESI_DR2"]['with_horizon_fitted_over_z_lower_1']["chi2_total"],
                               reference_case['chi2_full'],rtol=5e-8)

    comparator_runs=[]
    for case in refs["gaugessence"]['cases']:
        radiation=standard_radiation/case["h"]**2
        comparator_runs.append((case,radiation,evaluate(case,radiation,data,covariance,refs,sn)))
    runs=r445_runs+comparator_runs
    for case,radiation,actual in runs:
        repeat=evaluate(case,radiation,data,covariance,refs,sn,tolerance=7e-10,amin=3e-7)
        for key in ("theta100","rd_H0_sur_c","Dstar_H0_sur_c"):
            np.testing.assert_allclose(repeat['acoustic'][key],actual['acoustic'][key],rtol=2e-6)
        for key in ('c2_over_c2_present','c2_over_c2_a_zstar','changes_sign_z'):
            np.testing.assert_allclose(repeat['stability_main_local'][key],
                                       actual['stability_main_local'][key],rtol=2e-6,atol=1e-8)
        for tag in ("0","1","1090"):
            key='tensor_sans_reaction_metric'
            tight=actual['stability_main_local'][key]['bands'][tag]['boundary_upper_H0_sur_c']
            loose=repeat['stability_main_local'][key]['bands'][tag]['boundary_upper_H0_sur_c']
            np.testing.assert_allclose(loose,tight,rtol=2e-6,atol=1e-8)

    h=ref['h_control']
    lcdm=dict(name_field='CDM with R445 material',omega_m=reference_case["omega_m"],h=h)
    control=evaluate(lcdm,standard_radiation/h**2,data,covariance,refs,sn)
    candidates=[]
    for epsilon in (1e-5,3e-6,1e-6,1e-7,0.):
        candidate=fit_geometry(epsilon,data,covariance,refs,sn)
        candidates.append(candidate)
        print(candidate['parameters']['name'],candidate['setting'],candidate["reserves"],flush=True)
    result=dict(revision=449,cases=[row[2] for row in runs]+[control],
                family_calibrated_R449=candidates,
                fit_R449="Famille d’essai quartique, phi0=0,4, signe électrique positif. Ω_B=ε et Ω_E=ε/4 fixés. "
                            'For each .pi., only .pi. and h change for the seven DESI components z-1 and the angle Planck. '
                            'The six components DESI z-1, the DES Y6 point excluding recovery and the six components '
                            'Pantheon-MCT are kept as separate controls. The checks are retrospective.',
                check_Characteristics_defect_absolute=check_characteristics(),
                limits='Flat background projections with imposed sound velocity, baryonic density and redshifts. '
                        'Planck extrema give a spacing test, with the hypothesis of consecutive modes '
                        "et l’approximation de projection ℓ ≈ kχ. Leur covariance n’est pas fournie ici. "
                        'Gaugessence figures are evaluations at published centres, without readjustment. '
                        "Pas de recombinaison calculée, de spectre théorique TT/TE/EE ni de vraisemblance CMB.",
                checksums={name:sha(source/name) for name in (
                    "REPERES_R449.json","RESERVE_SUPERNOVAE_R448.json","PROJECTION_ACOUSTIQUE_R448.json",
                    "FOND_MATIERE_R445.json","DESI_DR2_mean.txt","DESI_DR2_cov.txt")})
    write(args.output/"PICS_ET_TRIADE_R449.json",result)
    for row in result['cases']:
        ba=row["BAO_DESI_DR2"]
        print(row['parameters']['name'], "100 theta",row['acoustic']["theta100"],
              "BAO natif",ba['with_horizon_native']["chi2_total"],"BAO effectif",ba['with_horizon_fitted_over_z_lower_1']["chi2_total"],
              "SN",row["SN"]["chi2"],flush=True)


if __name__=="__main__":
    main()
