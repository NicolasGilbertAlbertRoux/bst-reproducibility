'Calculations of the substrate R412 in reduced units. Equations (4) to (7), (15) and (16).'
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def area_squared(a, b):
    return np.dot(a, a) * np.dot(b, b) - np.dot(a, b) ** 2


def response_local():
    # Orthonormal base of the tangential plan of dimension three. Unit volume weight.
    gradients = np.array(((1., 0., 0.), (0., .5, 0.), (0., 0., 0.)))
    identite = np.eye(3)
    gradient_square = sum(np.dot(g, g) for g in gradients)
    area_spatial = sum(area_squared(g, other) for i, g in enumerate(gradients)
                        for other in gradients[i + 1:])
    inertia = identite + sum(np.dot(g, g) * identite - np.outer(g, g) for g in gradients)

    def energies(speed, s):
        beta = math.exp(2 * s)
        kinetic = (np.dot(speed, speed)
                     + beta * sum(area_squared(speed, g) for g in gradients)) / 2
        potential = (gradient_square + beta * area_spatial) / 2
        return kinetic, potential

    results = {}
    for name, speed in (('rest', np.zeros(3)), ('motion', identite[2])):
        t, v = energies(speed, 0)
        ds = 1e-5
        tp, vp = energies(speed, ds)
        tm, vm = energies(speed, -ds)
        derivative = ((tp - vp) - (tm - vm)) / (2 * ds)
        force = sum(area_squared(speed, g) for g in gradients) - area_spatial
        np.testing.assert_allclose((t, derivative), (speed @ inertia @ speed / 2, force),
                                   rtol=1e-8, atol=2e-10)
        results[name] = dict(T=float(t), V=float(v), force_scale=float(force),
                              force_difference_centered=float(derivative))
    return dict(inerties=np.linalg.eigvalsh(inertia).tolist(), preparations=results)


def instant_of_pic(readouts, dt):
    i = int(np.argmax(readouts))
    if i == 0 or i == len(readouts) - 1:
        raise ValueError('The maximum of the signal reaches the edge of the detection window.')
    a, b, c = readouts[i-1:i+2]
    return (i + 0.5 * (a-c) / (a-2*b+c)) * dt


def speed_of_signal(h, dt, speed, channel):
    # Ring of length 64. The rest profile emits two opposite pulses.
    x = np.arange(round(64/h)) * h
    distance = (x-16+32) % 64 - 32
    field = np.exp(-0.5 * (distance/1.25)**2)
    sensors = np.rint(np.array((24., 32.)) / h).astype(int)
    courant2 = (speed*dt/h)**2
    lap = np.roll(field, 1) + np.roll(field, -1) - 2*field
    precedent = field + 0.5 * courant2 * lap  # vitesse initiale nulle
    readouts = np.empty((round(28/dt)+1, 2))
    for i in range(len(readouts)):
        lap = np.roll(field, 1) + np.roll(field, -1) - 2*field
        next = 2*field-precedent+courant2*lap
        if channel == "orientation":
            readouts[i] = field[sensors]
        else:
            # Jauge temporelle, polarisation transverse A_y(x), E_y = -dA_y/dt.
            readouts[i] = (precedent[sensors]-next[sensors])/(2*dt)
        precedent, field = field, next
    t1, t2 = (instant_of_pic(s, dt) for s in readouts.T)
    return 8/(t2-t1), t1, t2


def trace_modal(k, c_s, h, dt):
    # For a Fourier mode, the two neighbors have the phases zkh and -kh.
    # The discrete equation is integrated from a unit amplitude without speed.
    # The rest clock a U'/m's 1 in all preparations.
    acceleration = 2*c_s**2 * (math.cos(k*h)-1)/h**2 - 1
    trace = np.empty(round(100/dt)+1)
    trace[0] = 1.
    trace[1] = 1. + 0.5*dt**2*acceleration
    for i in range(1, len(trace)-1):
        trace[i+1] = 2*trace[i] - trace[i-1] + dt**2*acceleration*trace[i]
    return trace


def read_cycles(trace, dt):
    indices = np.flatnonzero((trace[:-1] > 0) & (trace[1:] <= 0))
    passages = dt*(indices + trace[indices]/(trace[indices]-trace[indices+1]))
    if len(passages) < 3:
        raise ValueError('Too few periods to determine the pulse.')
    return math.tau*(len(passages)-1)/(passages[-1]-passages[0]), passages


def wave_over_trajectory(trace, omega, k, positions, dt):
    cosine = trace[1:-1]
    sinus = (trace[:-2]-trace[2:]) / (2*math.sin(omega*dt))
    return cosine*np.cos(k*positions) + sinus*np.sin(k*positions)


def compare_clocks():
    # Family, resolution speed, gauge speed, carrier wave number.
    cases = (('common', 1., 1., 3/4), ('control_scale', 3/4, 1., 16/9),
           ('control_gauge', 1., 3/4, 3/4))
    grids = [("raffinement", h, h/4, h/4) for h in (.25, .125, .0625, .03125)]
    h = grids[-1][1]
    grids += [('half_step_time', h, h/8, h/4), ('half_band', h, h/4, h/8)]
    rows = []
    for check, h, dt, deviation_k in grids:
        c_q, tq1, tq2 = speed_of_signal(h, dt, 1., "orientation")
        gauges = {c: speed_of_signal(h, dt, c, 'electric') for c in (1., 3/4)}
        trace_rest = trace_modal(0., 1., h, dt)
        rest, _ = read_cycles(trace_rest, dt)
        time = np.arange(1, len(trace_rest)-1) * dt
        for family, c_s, c_a, center in cases:
            c_a_lu, ta1, ta2 = gauges[c_a]
            for sign in (1, -1):
                k = sign * center
                k_minus, k_plus = k-deviation_k, k+deviation_k
                mode_minus = trace_modal(k_minus, c_s, h, dt)
                mode_plus = trace_modal(k_plus, c_s, h, dt)
                w_minus, _ = read_cycles(mode_minus, dt)
                w_plus, _ = read_cycles(mode_plus, dt)
                speed = (w_plus - w_minus) / (2*deviation_k)
                positions = speed * time
                field_lu = (wave_over_trajectory(mode_minus, w_minus, k_minus, positions, dt)
                            + wave_over_trajectory(mode_plus, w_plus, k_plus, positions, dt)) / 2
                omega_lu, tics = read_cycles(field_lu, dt)
                rate = omega_lu / rest
                rate_par_phase = ((w_plus + w_minus)/2 - k*speed) / rest
                if abs(rate-rate_par_phase) > 1e-5:
                    raise ValueError('The count of the cycles diverges from the phase of the carrier.')
                rate_echo = math.sqrt(1-(speed/c_a_lu)**2)
                rate_q = math.sqrt(1-(speed/c_q)**2)
                rows.append(dict(check=check, family=family, h=h, dt=dt,
                                   half_band_k=deviation_k, c_s=c_s, c_A=c_a, k_center=k,
                                   t_q_24=tq1, t_q_32=tq2, c_q_lu=c_q, t_A_24=ta1,
                                   t_A_32=ta2, c_A_lu=c_a_lu, omega_rest=rest,
                                   omega_minus=w_minus, omega_plus=w_plus, speed_envelope=speed,
                                   intervals_read=len(tics)-1, first_tic=tics[0]+dt, last_tic=tics[-1]+dt,
                                   rate_par_phase=rate_par_phase,
                                   rate_battement=rate, rate_echo=rate_echo, rate_echo_q=rate_q,
                                   deviation_battement_minus_echo=rate-rate_echo))
    return rows


def draw_clock(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speed, separation = 0.6, 1.
    half_period = separation / math.sqrt(1-speed**2)
    t = np.linspace(0, 2*half_period, 100)
    green, ochre = "#155e60", "#a44728"
    fig, (cone, trajet) = plt.subplots(1, 2, figsize=(8.6, 3.55),
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    cone.fill_betweenx(t, -t, t, color="#e1eeea")
    cone.plot(-t, t, color=green)
    cone.plot(t, t, color=green)
    cone.plot(speed*t, t, color=ochre, lw=2)
    cone.scatter([0, speed*t[-1]], [0, t[-1]], s=28, color=ochre, zorder=3)
    cone.annotate("compteur mobile", (0.87, 1.45), (-1.35, 1.96), color=ochre,
                  arrowprops={"arrowstyle": "-", "color": ochre}, fontsize=9)
    cone.set(xlabel="Position x", ylabel='Time - signal speed',
             xlim=(-2.65, 2.65), ylim=(-.12, 2.72), aspect="equal")
    cone.set_title('The counter inside the cone', loc="left", fontsize=10)
    xx = speed*half_period
    trajet.axhline(0, color="#7a8687", lw=1)
    trajet.axhline(separation, color="#7a8687", lw=1)
    trajet.plot([0, xx, 2*xx], [0, separation, 0], color=green, lw=2)
    trajet.scatter([0, xx, 2*xx], [0, separation, 0], s=33, color=green, zorder=3)
    for x, y, mot, dy in ((0, 0, "émission", -.16), (xx, 1, "réflexion", .10),
                           (2*xx, 0, 'return', -.16)):
        trajet.text(x, y+dy, mot, ha="center", fontsize=9)
    trajet.set(xlabel="Position x", ylabel="Position transverse y",
               xlim=(-.34, 1.87), ylim=(-.32, 1.37), aspect="equal")
    trajet.set_title('The round trip between mobile mirrors', loc="left", fontsize=10)
    for ax in (cone, trajet):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout(pad=1.2)
    fig.savefig(output/"Cone_et_horloge.png", dpi=165)
    plt.close(fig)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"Reponse_locale_R412.json").write_text(
        json.dumps(response_local(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = compare_clocks()
    with (output/"Horloges_R412.csv").open("w", encoding="utf-8", newline="") as f:
        table = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        table.writeheader()
        for row in rows:
            table.writerow({k: format(v, ".12g") if isinstance(v, float) else v
                            for k, v in row.items()})
    draw_clock(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path, help='folder for CSV clocks and local JSON')
    main(p.parse_args().output)
