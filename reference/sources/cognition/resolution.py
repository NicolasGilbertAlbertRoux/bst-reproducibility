'Resolution of the reader and transfer of a pulse into the H528 model.'
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

# A small system of this type is effectively calculated on a BLAS wire.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

from compute import COGNITION, write_table, json_write

ETA_REF = 0.8
J_REF = 0.10
AMPLITUDES = (0.02, J_REF, 0.20)
NAME = "BST_Resolution_et_asymptote"


def first_passage(time, q):
    passages = np.flatnonzero((q[:-1] > 0) & (q[1:] <= 0))
    if not len(passages):
        return None
    i = passages[0]
    return float(time[i] - q[i] * (time[i+1] - time[i]) / (q[i+1] - q[i]))


def probe_transfer(h):
    settings = [(0.8, 0.2), (0.8, 2.0), (0.0, 2.0)]
    cases = [h.make_case(tag="m1", eta=eta, lam=0.5, mass=0.5, impulse=j, sign=1)
           for eta, j in settings]
    t_fin = 4.0
    # Two different integrators and two different sampling steps.
    step = 0.001
    rec, drift, _ = h.integrate(cases, dt=step, end=t_fin, sample=0.005)
    measurements = []
    for i, c in enumerate(cases):
        t2, etat2, derive2, _ = h.independent(c, end=t_fin)
        time, q = rec[:, i, 0], rec[:, i, 2]
        first = first_passage(time, q)
        control = first_passage(t2, etat2[1])
        deviation_q = np.max(np.abs(np.interp(t2, time, q) - etat2[1]))
        assert (first is None) == (control is None)
        if control is not None:
            assert abs(first - control) < 0.0001
        assert deviation_q < 0.0001 and drift[i] < 0.00003 and derive2 < 0.000001
        measurements.append(dict(eta=c["eta"], J=c["impulse"], energy=c["impulse"]**2/2,
                            minimum_q_sampled=float(q.min()), t_first_zero_Verlet=first,
                            t_first_zero_DOP853=control, deviation_q_max_aux_time_common=float(deviation_q),
                            drift_relative_Verlet=float(drift[i]), drift_relative_DOP853=float(derive2)))
    assert [m["t_premier_zero_DOP853"] is not None for m in measurements] == [False, True, False]
    return dict(bath="m1", lambda_barrier=0.5, mass_q=0.5, q_initial=1, p_q_initial=0,
                eta_J=[list(x) for x in settings], horizon=t_fin, step_Verlet=step,
                criterion='First passage of the sign of q under an pulse exclusively in x.',
                measurements=measurements)


def compute(data):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "R528"
        shutil.copytree(COGNITION / "sources/R528", directory)
        spec = importlib.util.spec_from_file_location('model_h528', directory/"tester_persistance_R528.py")
        h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)
        k0 = h.K0
        transfer = probe_transfer(h)
    with (data/"Canal_H528_controles.csv").open(encoding="utf-8") as f:
        points = list(csv.DictReader(f))
    depths_checked = sorted({ETA_REF/float(p["eta"]) for p in points if float(p["eta"]) > 0})
    depth = np.unique(np.r_[np.geomspace(1, 1000, 301), depths_checked])
    eta = ETA_REF/depth
    contrast = np.array([2*j*k0*np.sinh(eta) for j in AMPLITUDES])
    c_ref = 2*J_REF*k0*np.sinh(ETA_REF)
    power = c_ref/contrast
    np.testing.assert_allclose(contrast*power, c_ref)
    assert np.all(np.diff(contrast, axis=1) < 0) and np.all(np.diff(power, axis=1) > 0)
    for p in points:
        exact = 2*float(p["J"])*k0*np.sinh(float(p["eta"]))
        np.testing.assert_allclose(exact, float(p['contrast_exact']), rtol=1e-12, atol=1e-15)
    columns = ["d", "eta", "C_J002", "C_J010", "C_J020", "R_J002", "R_J010", "R_J020"]
    write_table(data/"Profondeur_et_resolution.csv", columns, zip(depth, eta, *contrast, *power))
    np.testing.assert_allclose(np.arcsinh(c_ref/(2*J_REF*k0*power[1])), eta)
    content = dict(eta_ref=ETA_REF, J_ref=J_REF, J=list(AMPLITUDES), k0=k0, C_ref=c_ref,
                   d_checks=depths_checked, checks_short=len(points),
                   domaine_trace=[1, 1000], decoupling="eta=0, d tend vers l'infini, C=0",
                   criterion='u is a common error terminal by reading. Guarantee if C is 2 u. R is C-ref / (2u).',
                   transfer=transfer)
    json_write(data/"Bilan_resolution.json", content)
    return content


def tracer(data, output, with_pgf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    from illustrate import BLUE, INK, GRIS, LAVENDER, ROW, TEAL, VERTE, VIOLET, save
    x = np.genfromtxt(data/"Profondeur_et_resolution.csv", names=True, delimiter=",")
    info = json.loads((data/"Bilan_resolution.json").read_text())
    d = x["d"]
    checks = np.array(info['d_checks'])
    last = max(checks)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.05, 5.45), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.96, top=0.79, bottom=0.15, hspace=0.53)
    fig.text(0.044, 0.963, "Lire le latent quand le couplage diminue", fontsize=17, weight="bold",
             va="top", color=INK)
    fig.text(0.045, 0.898, 'Each band covers pulses from 0.02 to 0.20 in model units.',
             fontsize=9.6, color=GRIS)
    fig.text(0.045, 0.85, 'Purple bottom beyond the last simulated positive coupling. The formula is exactly being extended.',
             fontsize=9.1, color=VIOLET)
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlim(1, 1000)
        ax.axvspan(last, 1000, color=LAVENDER, zorder=0)
        ax.axvline(last, color=VIOLET, linewidth=0.8, linestyle=":", zorder=1)
        ax.grid(axis="y", which="major", color=ROW, linewidth=0.65, zorder=1)
        ax.tick_params(labelsize=9)
    a.set_title('Contrast between the two signs of the trace', loc="left", fontsize=11, weight="bold", pad=5)
    a.set_ylabel("Contraste  C")
    a.fill_between(d, x["C_J002"], x["C_J020"], color=VERTE, zorder=2)
    a.plot(d, x["C_J010"], color=TEAL, linewidth=1.8, zorder=3)
    with (data/"Canal_H528_controles.csv").open(encoding="utf-8") as f:
        measurements = [p for p in csv.DictReader(f) if p['bath']=="m1" and float(p["eta"])>0 and float(p["J"])==J_REF]
    a.scatter([ETA_REF/float(p["eta"]) for p in measurements], [float(p["estimation_trajectoires"]) for p in measurements],
              marker="o", s=18, color=TEAL, clip_on=False, zorder=4)
    a.axhline(0, color=GRIS, linestyle="--", linewidth=0.9, zorder=4)
    a.set_ylim(-0.12, 4.3)
    a.set_yticks([0, 2, 4])
    a.annotate("Limite  C = 0", xy=(370, 0.0), xytext=(0.95, 0.51), textcoords="axes fraction",
               ha="right", color=TEAL, fontsize=9.3,
               arrowprops={"arrowstyle": "->", "lw": 0.8, "color": TEAL})
    b.set_title('Resolution power required to distinguish signs', loc="left", fontsize=11, weight="bold", pad=6)
    b.set_ylabel('Resolution Threshold')
    b.set_yscale("log")
    b.fill_between(d, x["R_J020"], x["R_J002"], color="#c9dae9", zorder=2)
    b.plot(d, x["R_J010"], color=BLUE, linewidth=1.8, zorder=3)
    b.set_ylim(0.35, 8000)
    b.set_yticks([1,10,100,1000], labels=["1","10","100","1 000"])
    b.yaxis.set_minor_locator(NullLocator())
    b.xaxis.set_major_locator(FixedLocator([1,10,100,1000]))
    b.set_xticklabels(["1","10","100","1 000"])
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlabel(r"Couplage inverse relatif  $d = 0{,}8/\eta$  (échelle logarithmique)", labelpad=5)
    fig.text(0.045, 0.035, "Traits centraux pour J = 0,10. Points en haut issus des trajectoires. L’axe d est sans unité.",
             fontsize=8.8, color=GRIS)
    save(fig, output, NAME, with_pgf)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('data', type=Path)
    cli.add_argument("figures", type=Path)
    cli.add_argument("--pgf", action="store_true", help='Additional LaTeX auction for editing.')
    args = cli.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    info = compute(args.data)
    tracer(args.data, args.figures, args.pgf)
    print(json.dumps(info['transfer']['measurements'], ensure_ascii=False, indent=2))
