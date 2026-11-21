'Extend the tensor drive R422 onto the FLRW triad, metric held.'
import numpy as np


def tensor_diagnostics(state, inverse, model):
    g, alpha, h = (model[key] for key in ("g","alpha","h"))
    H0_over_c=100*h/299792.458

    def coefficients(z):
        a=1/(1+z)
        phi,j,_,_=state(a)
        response=1+alpha*(phi/a)**4
        electric_comoving=j/np.sqrt(response)
        conformal_step=a*inverse(a)
        phi_n=electric_comoving*conformal_step
        # λ=κ a^-4 P0, P0=3 g φ² φ̇ with the dot in conformal time.
        # Use the background equation for dλ/dη to avoid subtracting large terms.
        lambda_derivative=alpha/(g*a**4*np.sqrt(response))*(
            2*j*phi*(phi_n-2*phi)/response-2*g*g*phi**5*conformal_step/np.sqrt(response)
        )/conformal_step
        return g*phi,lambda_derivative

    def band(z):
        mass,derivative=coefficients(z)
        parity=derivative-2*mass
        roots=np.sort(np.roots([1,-abs(parity),-derivative*mass]))
        assert not np.any(np.iscomplex(roots))
        low=max(float(roots[0]),0.)
        high=max(float(roots[1]),0.)
        return dict(helicite_plus_fragile=1 if parity < 0 else -1,
                    boundary_lower_H0_over_c=low,boundary_upper_H0_over_c=high,
                    boundary_upper_Mpc_inverse=high*H0_over_c,
                    width_Mpc_inverse=(high-low)*H0_over_c,
                    maximum_rate_instantane_over_aH=np.sqrt(max(parity**2/4+derivative*mass,0.))*inverse(1/(1+z))/(1+z),
                    g_phi=mass,derivative_lambda_eta=derivative)

    eta_defect=0.
    # Check the conformal derivative by integrating it over short symmetric
    # intervals and comparing with the endpoints of λ. This avoids taking a
    # derivative of the almost constant and large value of λ at small ε.
    for z in (.2,1.,20.):
        center=-np.log1p(z)

        def log_value(n):
            a=np.exp(n)
            phi,j,_,_=state(a)
            ratio=alpha*(phi/a)**4
            return np.log(abs(j)/a**2)-.5*np.log1p(1/ratio)

        step=6e-4
        n_low,n_high=center-step,center+step
        j_center=state(np.exp(center))[1]
        sign=np.sign(j_center)
        lambda_scale=np.sqrt(alpha)/g
        value_low=sign*lambda_scale*np.exp(log_value(n_low))
        delta_lambda=value_low*np.expm1(log_value(n_high)-log_value(n_low))
        estimate=delta_lambda/(2*step)/(np.exp(center)*inverse(np.exp(center)))
        analytic=coefficients(z)[1]
        eta_defect=max(eta_defect,float(abs(estimate-analytic)/max(abs(analytic),1.)))
    assert eta_defect < 8e-5, (model['name'],eta_defect)
    domain=np.expm1(np.linspace(0.,np.log1p(1500.),300))
    rows=[band(float(z)) for z in domain]
    max_upper=max(r['boundary_upper_Mpc_inverse'] for r in rows)
    return dict(scope='Instantaneous square frequency of a gauge tensor in an FLRW metric held. '
                      'The negative sign band signals a regime to be studied with the metric equation. '
                      'An instantaneous frequency does not predict the integrated amplification or the observed polarization.',
                defect_derivation_lambda=eta_defect,z_enumerated=len(rows),
                max_boundary_upper_Mpc_inverse=max_upper,
                bands={str(z):band(z) for z in (0,.5,1,10,100,1090,1500)})
