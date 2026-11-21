#!/usr/bin/env python3
'R173. Structural inferences at observation windows; no cognitive fit.\n\nRun with Python 3, numpy and matplotlib. All inputs are frozen R170 graph\nmemory matrices; no native image, molecular trajectory or radiation source\nis generated here. Numerical witnesses supplement the written proofs.'
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 10})


def memory():
    spec = json.loads((ROOT/'entrees/CANDIDAT_GELE.json').read_text())
    p = np.asarray(spec['parameters'])
    norm = spec['normalization']
    sizes = np.asarray(spec['radii']) / spec['radii'][0]
    P0 = np.zeros((6, 6)); P0[:3, :3] = 1/3
    P2 = np.eye(6)-P0
    r = np.exp(p[3]); L = (r*P0+P2)/(r+5)
    root = (np.exp(p[7]/2)*P0+np.exp(p[8]/2)*P2)/np.sqrt(norm['alpha_reference'])
    curves = []; report = []
    # Same native time reference for every species; scaled only by frozen size law.
    tref = np.exp(p[9])/norm['frequency_reference']
    t = tref*np.geomspace(1e-4, 8, 900)
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.5), layout='constrained')
    for i, sp in enumerate(['Ar', 'Kr', 'Xe']):
        with np.load(ROOT/'inputs'/f'GRAPH_{sp}.npz') as z:
            lam, G = z['lam'], z['G']
        tau = tref*sizes[i]**2
        W = root@G@root
        weights = np.einsum('ab,jba->j', L, W)
        rates = lam**p[12]/tau
        # Positive contraction of PSD matrices: allow only floating point noise.
        assert weights.min() >= -1e-12*weights.max()
        positive = weights > 0
        weights, rates = weights[positive], rates[positive]
        q = np.exp(-t[:, None]*rates[None, :])*weights[None, :]
        K = q.sum(1)
        K0 = weights.sum()
        E = (q*rates).sum(1)/K
        variance = (q*(rates[None, :]-E[:, None])**2).sum(1)/K
        # Analytic derivative and central finite difference at independent times.
        probes = tref*np.array([.02, .08, .2, .5, 1.0])
        h = tref*1e-6
        def effective(tt):
            qq = np.exp(-tt[:, None]*rates)*weights
            return (qq*rates).sum(1)/qq.sum(1)
        qp = np.exp(-probes[:, None]*rates)*weights
        ep = (qp*rates).sum(1)/qp.sum(1)
        vp = (qp*(rates-ep[:, None])**2).sum(1)/qp.sum(1)
        fd = (effective(probes+h)-effective(probes-h))/(2*h)
        derivative_error = float(np.linalg.norm(fd+vp)/np.linalg.norm(vp))
        assert derivative_error < 1e-5
        assert np.max(np.diff(E)) < 1e-8*max(E)
        # Positive Laplace-axis scalar impedance identity Gamma = Sigma / s.
        s = np.geomspace(1e-3/tref, 1e3/tref, 23)
        gamma_modal = (weights/(s[:, None]+rates)).sum(1)
        u = s*tau
        spatial = np.einsum('sj,jab->sab', u[:, None]/(lam**p[12]+u[:, None]), G)
        sigma = np.einsum('ab,sbc,cd->sad', root, spatial, root)
        gamma_matrix = np.einsum('ab,sba->s', L, sigma)/s
        modal_defect = float(np.max(abs(gamma_modal/gamma_matrix-1)))
        assert modal_defect < 1e-10
        report.append(dict(species=sp, native_modes=len(lam),
                           positive_contraction_modes=len(weights),
                           tau_over_tref=float(tau/tref),
                           nu=float(p[12]), minimum_weight=float(weights.min()),
                           minimum_rate=float(rates.min()), maximum_rate=float(rates.max()),
                           mean_rate_t0_times_tref=float(np.dot(weights, rates)/K0*tref),
                           rate_derivative_fd_relative_error=derivative_error,
                           modal_vs_matrix_Laplace_relative_defect=modal_defect,
                           effective_rate_nonincreasing=True))
        curves.append(dict(species=sp, t_over_tref=t/tref, K_over_K0=K/K0,
                           effective_rate_times_tref=E*tref,
                           variance_times_tref2=variance*tref**2))
        axs[0].loglog(t/tref, K/K0, label=sp)
        axs[1].semilogx(t/tref, E*tref, label=sp)
    axs[0].set(title='A. Memorial of the fixed candidate', xlabel='t/t common ref.',
               ylabel='K(t) / K(0)', ylim=(.025, 1.1))
    axs[1].set(title='B. Taux de relaxation effectif', xlabel='t/t common ref.',
               ylabel='r effective - t ref.')
    for ax in axs: ax.grid(alpha=.2); ax.legend()
    fig.suptitle('BST R173 · Formes latentes contraintes par une mémoire positive\n'
                 'Ar/Kr/Xe : noyau interne R170, paramètres inchangés ; aucune mesure optique.', fontsize=12)
    for ext in ['png', 'svg']: fig.savefig(FIG/f'MEMOIRE_LATENTE.{ext}', dpi=180)
    plt.close(fig)
    np.savez_compressed(OUT/'MEMOIRE_LATENTE.npz', t_over_tref=t/tref,
                        K_over_K0=np.array([c['K_over_K0'] for c in curves]),
                        effective_rate_times_tref=np.array([c['effective_rate_times_tref'] for c in curves]))
    return dict(source=spec, time_reference_native=float(tref), species=report,
                interpretation='Frozen model prediction of the internal memory kernel, not a measured radiation curve',
                full_monotonicity_assumption='Positive sum of real exponential relaxation modes; not every passive bath')


def observability():
    rates = np.array([.5, 1.5, .12])
    A = -np.diag(rates)
    B = np.eye(3)
    C1 = np.diag([1., 1., 0.])
    C2 = np.array([[0., 0., 1.]])
    T = 10.
    # Exact integration for diagonal A, including all off-diagonal factors.
    def gram(C):
        pairs = rates[:, None]+rates[None, :]
        return (C.T@C)*(-np.expm1(-pairs*T))/pairs
    W1, W2 = gram(C1), gram(C2)
    eig1, eig12 = np.linalg.eigvalsh(W1), np.linalg.eigvalsh(W1+W2)
    assert eig1[0] == 0 and eig12[0] > 0
    # Numerically check a change of latent coordinates. The independent
    # dark-mode extension is established analytically in the manuscript.
    angle = .61
    U = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0],
                  [np.sin(angle), 0, np.cos(angle)]])
    AA, BB, CC = U@A@U.T, U@B, C1@U.T
    ss = .03+1j*np.geomspace(.01, 30, 31)
    dif = []
    for s in ss:
        h1 = C1@np.linalg.solve(s*np.eye(3)-A, B)
        h2 = CC@np.linalg.solve(s*np.eye(3)-AA, BB)
        dif.append(np.linalg.norm(h1-h2)/np.linalg.norm(h1))
    assert max(dif) < 1e-12
    t = np.linspace(0, T, 401)
    x0a = np.array([.7, -.2, 1.])
    x0b = np.array([.7, -.2, 0.])
    xa = np.exp(-t[:, None]*rates)*x0a
    xb = np.exp(-t[:, None]*rates)*x0b
    assert np.array_equal(xa@C1.T, xb@C1.T)
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.5), layout='constrained')
    axs[0].plot(t, xa[:, 0], label='Window 1, first component (two states)')
    axs[0].plot(t, xa[:, 2], label='Window 2, latent load retained')
    axs[0].plot(t, xb[:, 2], '--', label='Window 2, load erased')
    axs[0].set(title='A. Same initial reading, other discriminating reading',
               xlabel='Time of the illustrative model', ylabel='Amplitude')
    xx=np.arange(3)
    axs[1].bar(xx-.17, eig1, .34, label='Window 1')
    axs[1].bar(xx+.17, eig12, .34, label='Windows 1 and 2')
    axs[1].set(xticks=xx, xticklabels=['λ min.', '- med.', 'λ max.'],
               title='B. Cumulative dynamic information', ylabel='Valeurs propres du gramien, T = 10')
    for ax in axs: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('BST R173 · Rendre un état latent observable\n'
                 'Explicit linear indicator; windows are not identified by sound or light.', fontsize=12)
    for ext in ['png','svg']: fig.savefig(FIG/f'FENETRES_ET_PERSISTANCE.{ext}', dpi=180)
    plt.close(fig)
    return dict(rates=rates.tolist(), A=A.tolist(), C1=C1.tolist(), C2=C2.tolist(),
                horizon=T, gramian_1=W1.tolist(), gramian_2=W2.tolist(),
                eigenvalues_window1=eig1.tolist(), eigenvalues_joint=eig12.tolist(),
                exact_same_window1=True, window2_distinguishes_latent_payload=True,
                orthogonal_similarity_transfer_relative_defect=float(max(dif)),
                arbitrary_unforced_dark_modes_identifiable=False,
                joint_norm_bound_coefficient=float(1/np.sqrt(eig12[0])),
                physical_cognitive_identification=False)


def symmetry():
    # C2 exchanges full states L/R, whereas the coarse observation is constant.
    states = (0,1)
    pi = lambda _: 0
    swap = lambda x: 1-x
    selector = lambda x: x
    assert all(pi(swap(x)) == pi(x) for x in states)
    assert all(selector(swap(x)) == swap(selector(x)) for x in states)
    assert len({selector(x) for x in states if pi(x)==0}) == 2
    return dict(group='C2', full_states=['L','R'], coarse_state='undifferentiated',
                full_stabilizer_order=1, coarse_stabilizer_order=2,
                equivariant_full_selector_exists=True,
                equivariant_selector_of_coarse_state_exists=False,
                physical_state_completeness_proved=False)


def main():
    result = dict(symmetry=symmetry(), memory=memory(), observability=observability(),
                  scope='Structural and numerical witnesses, no new evidence of phenomenal consciousness or strong authorship')
    (OUT/'FENETRES_LATENTES.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status':'PASS', 'new_native_images':0,
                      'native_memory_modes':sum(s['native_modes'] for s in result['memory']['species']),
                      'joint_gramian_eigenvalues':result['observability']['eigenvalues_joint'],
                      'max_modal_defect':max(s['modal_vs_matrix_Laplace_relative_defect'] for s in result['memory']['species']),
                      'max_fd_defect':max(s['rate_derivative_fd_relative_error'] for s in result['memory']['species'])}, indent=2))


if __name__ == '__main__': main()
