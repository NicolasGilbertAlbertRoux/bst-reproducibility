'Measuring the beat and current balances, reconstructing the 87Sr line.'
import argparse
import json
import math
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def maintenance():
    mass, stiffness, amplitude = 0.3, 2.0, 0.1
    omega = math.sqrt(stiffness / mass)
    period = math.tau / omega
    duration = 10 * period
    energy_initial = stiffness * amplitude**2 / 2
    cases = {}
    for name, gamma, contribution in (("conservatif", 0., False), ("amorti", .06, False),
                                ("entretenu", .06, True)):
        def derivative(t, y):
            b, speed = y[:2]
            force = gamma * amplitude * omega * math.cos(omega*t) if contribution else 0.
            return [speed, (force - gamma*speed - stiffness*b)/mass,
                    force*speed, gamma*speed**2]
        trajet = solve_ivp(derivative, (0, duration), (0., amplitude*omega, 0., 0.),
                           method="DOP853", rtol=1e-11, atol=1e-13,
                           t_eval=np.linspace(0, duration, 2001))
        if not trajet.success:
            raise RuntimeError(trajet.message)
        b, speed, work, loss = trajet.y
        energy = (mass*speed**2 + stiffness*b**2) / 2
        error = float(np.max(np.abs(energy - energy_initial - work + loss)))
        assert error < 6e-11
        cases[name] = {'fraction_energy_finale': float(energy[-1]/energy_initial),
                    'work_recu': float(work[-1]), 'energy_dissipee': float(loss[-1]),
                    'error_summary_max': error}
    mean_field = .06 * amplitude**2 * omega**2 / 2
    assert abs(cases["entretenu"]['energy_dissipee']/duration - mean_field) < 1e-11
    assert abs(cases["conservatif"]['fraction_energy_finale']-1) < 2e-9
    assert abs(cases["entretenu"]['fraction_energy_finale']-1) < 2e-9
    assert cases["amorti"]['fraction_energy_finale'] < .009
    return {"mu": mass, "kappa": stiffness, "amplitude": amplitude, 'gamma_with_losses': .06,
            'periods': 10, 'duration': duration, 'energy_initial': energy_initial,
            'puissance_entretenue_mean': mean_field, "preparations": cases}


def current_of_acts():
    path = ROOT/"cognition/sources/A0_A9/A9_sealed_vessel_source_attribution/bst_cog_a9_results.json"
    archive = read(path)["public_nonidentifiability"]
    trace = archive["public_trace"]
    transitions = Counter(zip(trace, trace[1:]))
    duration = len(trace)-1   # A time step by transition in this example.
    acts = sorted(set(trace))
    current = {(i, j): (transitions[i, j]-transitions[j, i])/duration
               for i in acts for j in acts}
    for i in acts:
        variation = (int(trace[-1] == i)-int(trace[0] == i))/duration
        assert math.isclose(sum(current[j, i] for j in acts), variation, abs_tol=1e-15)
    assert len(set(archive["public_sha256_by_source_class"].values())) == 1
    return {"trace_historique": trace, 'duration_in_step': duration,
            'classes_of_source': archive["candidate_source_classes"],
            'currents_positifs_par_step': [{'of': i, 'toward': j, "net": jnet}
                                          for (i, j), jnet in current.items() if jnet > 0]}


def current_of_field():
    # Free harmonic solution of m s'tt - k's s'xx .K s . . . 0 within the continuous limit.
    stiffness, mass, rappel, amplitude = 1., 1., 1., .1
    phases = np.arange(256)*math.tau/256
    flux = {}
    for k in (-.75, 0., .75):
        omega = math.sqrt((rappel+stiffness*k*k)/mass)
        ds_dt = amplitude*omega*np.sin(phases)
        ds_dx = -amplitude*k*np.sin(phases)
        mean_field = float(np.mean(-stiffness*ds_dt*ds_dx))
        theorique = stiffness*amplitude**2*omega*k/2
        assert abs(mean_field-theorique)<1e-14
        flux[str(k)] = mean_field
    return {"k_s": stiffness, "m_s": mass, "rappel": rappel, "amplitude": amplitude,
            'samples_of_phase': len(phases), 'flux_mean_par_count_wave': flux}


def clock_atomic():
    inputs = read(ROOT/"redaction/donnees/Entrees_strontium.json")
    c, b, measurement, units = (inputs[k] for k in ("courtillot", "bipm", "bothwell", "conventions"))
    nu = (c["transition_g_vers_P1_kHz"]-c["intervalle_P0_vers_P1_kHz"])*1000
    residual = Decimal(nu)-Decimal(b['frequency_recommandee_Hz'])
    energy_eV = Decimal(units["h_J_s"])*Decimal(nu)/Decimal(units["e_C"])
    base = ROOT/"references/R573/entrees/ATLAS_IONS_ISOTOPES_R557.sqlite"
    with sqlite3.connect(base.resolve().as_uri()+"?mode=ro", uri=True) as db:
        row = db.execute("SELECT electrons, ionization_eV FROM IONISATIONS WHERE Z=38 AND q=0").fetchone()
    assert row[0] == 38 and abs(row[1]-5.69486745) < 1e-8
    # Vertical axis of the authors. Normal gravity is only used to restore the order of magnitude.
    normal = -units["g_conventionnel_m_par_s2"]*1e-3/units["c_m_par_s"]**2
    slope = measurement["gradient_fractionnaire_par_mm"]
    uncertainty = measurement['uncertainty_gradient_par_mm']
    expected = measurement['gradient_expected_axis_article_par_mm']
    assert abs(normal/expected-1)<.002
    return {'frequency_indirecte_Hz': nu, 'incertitude_indirecte_published_Hz': c['incertitude_indirecte_published_kHz']*1000,
            'difference_indirect_moins_BIPM_Hz': str(residual), "energie_transition_eV": float(energy_eV),
            "electrons_Sr_neutre": row[0], "energie_ionisation_atlas_eV": row[1],
            "gradient_fractionnaire_normal_par_mm": normal, 'gradient_published_attendu_Hz_par_mm': nu*expected,
            'gradient_measurement_Hz_par_mm': nu*slope, "incertitude_gradient_Hz_par_mm": nu*uncertainty,
            "sensibilite_supplementaire_effective": slope/expected-1,
            'uncertainty_sensibilite': uncertainty/abs(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results = {"battement": maintenance(), 'current_field': current_of_field(),
                 'current_acts': current_of_acts(), "strontium": clock_atomic()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
