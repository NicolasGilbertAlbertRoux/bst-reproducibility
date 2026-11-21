'Probe the frozen R460 Kr/Xe transport law at one common gas state.'
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.constants import Avogadro, Boltzmann, R, physical_constants
from scipy.fft import irfft, next_fast_len, rfft
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
PRESSURE = 3 * 101325.
TEMPERATURE = 298.15
WINDOWS = np.array([40., 80., 120., 160., 240., 320., 480., 640.])
DIFFUSION_WINDOWS = ((80, 160), (160, 320), (320, 640))
SEEDS = (46111, 46137, 46159)


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def compile_reader(parent, cache):
    # The additions only read the unwrapped NVE trajectory. The force,
    # initial state, time stepping and stress columns are those of R454.
    source = parent / 'transport_R454.cpp'
    text = source.read_text()
    before = 'if (i%stride==0) m.record(out,i*dt);'
    assert text.count(before) == 1
    text = text.replace('for (int i=0; i<=steps; ++i) {',
                        '''std::ofstream xyz(std::string(argv[1])+".pos",std::ios::binary);
        if (!xyz) throw std::runtime_error("Cannot open positions");
        const int stride_xyz=std::max(1,int(std::lround(1./dt)));
        for (int i=0; i<=steps; ++i) {''')
    text = text.replace(before, before + '''
            if (i%stride_xyz==0) for (const auto &x : m.unwrapped)
                xyz.write(reinterpret_cast<const char *>(x.data()),sizeof(double)*x.size());''')
    text = text.replace('if (!out) throw std::runtime_error("Cannot write the trajectory output");',
                        'if (!out || !xyz) throw std::runtime_error("Cannot write the trajectory output");')
    copy = cache / 'transport_positions_R461.cpp'
    copy.write_text(text)
    binary = cache / 'transport_R461'
    subprocess.run(['c++', '-O3', '-std=c++17', copy, '-o', binary], check=True)
    return binary, digest(source)


def gas_setup(gas, parent, cache, r460, calibration):
    inputs = parent / 'entrees/transport'
    constants = json.loads((inputs/'POTENTIELS_ET_ETATS_R442.json').read_text())['configs'][gas]
    if gas == 'Xe':
        previous = constants['mass_kg']
        constants['mass_kg'] = 128.9047808611 * physical_constants['atomic mass constant'][0]
        scale = np.sqrt(constants['mass_kg']/previous)
        constants['unit_time_s'] *= scale
        constants['unit_viscosite_uPa_s'] *= scale
    constants['T_reduced'] = TEMPERATURE * Boltzmann / constants['energie_J']
    length = calibration['factors_length'][gas]
    potential = inputs / f'POTENTIEL_{gas}_R444.dat'
    pair = np.loadtxt(potential)
    pair[0] *= length
    pair[1] *= calibration['amplitude_pair']
    pair[10] = json.loads((inputs/'DISPERSION_R453.json').read_text())['espèces'][gas]['C9']
    coeff = np.asarray(calibration['coefficients_three_corps'])
    short = .09*(coeff[:2]+np.log(constants['sigma_m']/3.63e-10)/.05*coeff[2:])
    params = cache / f'{gas}_coefficients_R461.txt'
    np.savetxt(params, np.r_[pair, 2.45, short][None])
    virials = calibration['viriels'][-1]['cases']
    if gas == 'Xe':
        rows = json.loads((r460/'resultats/DIFFUSION_Xe_LONG_R460.json').read_text())
        state = next(x for x in rows['extrapolations_diluées'] if x['T_K'] == TEMPERATURE)
        density = state['rho_molar']
    else:
        state = next(x for x in virials if x['gas'] == gas and x['T_K'] == TEMPERATURE)
        b2, b3 = state['B2']*1e-6, state['B3']*1e-12
        density = brentq(lambda n: n*R*TEMPERATURE*(1+b2*n+b3*n*n)-PRESSURE,
                         0., 2*PRESSURE/(R*TEMPERATURE))
    rho = density * Avogadro * constants['sigma_m']**3
    return constants, params, rho, density, short, digest(potential)


def msd(positions, largest_lag, batch=24):
    'Mean squared displacement over every available time origin.'
    n = len(positions)
    lag = np.arange(largest_lag+1)
    squared = np.einsum('tij,tij->t', positions, positions, dtype=np.float64)
    prefix = np.r_[0., squared.cumsum()]
    ends = prefix[n-lag] + prefix[-1] - prefix[lag]
    correlations = np.zeros_like(ends)
    size = next_fast_len(2*n-1, real=True)
    for start in range(0, positions.shape[1], batch):
        xyz = np.asarray(positions[:, start:start+batch], dtype=np.float64)
        spectrum = rfft(xyz, n=size, axis=0)
        power = np.einsum('tij,tij->t', spectrum.real, spectrum.real)
        power += np.einsum('tij,tij->t', spectrum.imag, spectrum.imag)
        correlations += irfft(power, n=size)[:largest_lag+1]
    return (ends-2*correlations)/((n-lag)*positions.shape[1])


def stress_readout(curve, constants, rho, count):
    time = curve[:,0]
    stress = curve[:,5:10] - curve[:,5:10].mean(axis=0)
    dt = time[1] - time[0]
    size = next_fast_len(2*len(stress)-1, real=True)
    power = abs(rfft(stress,n=size,axis=0))**2
    correlation = irfft(power,n=size,axis=0)[:len(stress)].mean(axis=1)
    correlation /= np.arange(len(stress),0,-1)
    scale = constants['unit_viscosite_uPa_s']*rho/(count*curve[:,1].mean())
    green = cumulative_trapezoid(correlation,dx=dt,initial=0.)
    accumulated = cumulative_trapezoid(stress,time,axis=0,initial=0.)

    def variance(delay):
        i = round(delay/dt)
        return np.mean((accumulated[i:]-accumulated[:-i])**2)

    helfand = [scale*(variance(w+.5)-variance(w-.5))/2 for w in WINDOWS]
    return dict(G_K=(scale*green[np.round(WINDOWS/dt).astype(int)]).tolist(), Helfand=helfand,
                drift_energy=float(np.ptp(curve[:,4])/abs(curve[0,4])),
                T_mean_K=float(TEMPERATURE*curve[:,1].mean()/constants['T_reduced']),
                P_mean_MPa=float(curve[:,2].mean()*constants['energie_J']/constants['sigma_m']**3*1e-6))


def trial(job):
    md, binary, cache, gas, setup, seed, cells, step, duration = job
    constants, params, rho, _, short, _ = setup
    count = 4*cells**3
    key = f'{gas}_{seed}_n{count}_d{step:g}_T{duration:g}'
    identity = digest(params)[:12] + digest(binary)[:12]
    stress = cache / f'{key}_{identity}.csv'
    path = stress.with_suffix('.csv.pos')
    if not (stress.is_file() and path.is_file()):
        partial = stress.with_suffix('.part')
        partial_positions = partial.with_suffix('.part.pos')
        print('départ', key, flush=True)
        command = [binary, partial, cells, rho, constants['T_reduced'], params,
                   step, 4.5, seed, round(duration/step), round(40./step), 1]
        process = subprocess.run(list(map(str, command)), check=True, capture_output=True, text=True)
        expected = short @ np.exp(-((1.13-np.array([1.04,1.34]))/[.23,.41])**2)
        expected *= (1-(1.13/2.45)**18)**9
        np.testing.assert_allclose(float(process.stdout.split()[-1]), expected, rtol=1e-12, atol=1e-14)
        assert partial_positions.stat().st_size == (duration+1)*count*3*8
        partial_positions.replace(path)
        partial.replace(stress)
    xyz = np.memmap(path, mode='r', dtype=np.float64, shape=(duration+1,count,3))
    curve = np.loadtxt(stress, delimiter=',', skiprows=1)
    np.testing.assert_allclose(curve[-1,0], duration)
    original = np.mean(np.sum((xyz[::20]-xyz[0])**2,axis=-1),axis=1)
    np.testing.assert_allclose(original, curve[::1000,10], rtol=3e-11, atol=1e-12)
    segments = {'première_moitié': (0,duration//2), 'second_moitié': (duration//2,duration),
                'trajectory_entière': (0,duration)}
    rows = {}
    for name, (start,end) in segments.items():
        part = curve[(curve[:,0]>=start)&(curve[:,0]<=end)].copy()
        part[:,0] -= start
        reads = stress_readout(part, constants, rho, count)
        if name == 'trajectory_entière':
            baseline = md.read_curve(part, constants, rho, count)
            for field in ('G_K','Helfand'):
                np.testing.assert_allclose(reads[field],baseline[field],rtol=1e-10,atol=1e-9)
            reads['diffusion_origin_initial_m2_s'] = baseline['diffusion_m2s']
        time = np.arange(DIFFUSION_WINDOWS[-1][-1]+1)
        moved = msd(xyz[start:end+1], time[-1])
        assert abs(moved[0]) < 1e-7 and min(moved) > -1e-7
        # Check all origins independently at one lag away from the FFT origin.
        np.testing.assert_allclose(msd(xyz[start:end+1:17], 5)[5],
            np.mean(np.sum((xyz[start+85:end+1:17] - xyz[start:end+1-85:17])**2,axis=-1)), rtol=2e-10)
        conversion = constants['sigma_m']**2 / (6*constants['unit_time_s'])
        estimates = [float(np.polyfit(time[lo:hi+1], moved[lo:hi+1], 1)[0]*conversion)
                     for lo,hi in DIFFUSION_WINDOWS]
        reads['diffusion_origines_multiples_m2_s'] = estimates
        rows[name] = reads
    print('terminé',key,rows['trajectory_entière']['diffusion_origines_multiples_m2_s'],flush=True)
    return dict(gas_field=gas, seed=seed, N=count, step_field=step, duration_field=duration, rho_reduced=rho, cases=rows,
                checksums=dict(constraint=digest(stress), positions=digest(path)))


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ('r454','r460','cache'):
        parser.add_argument(key,type=Path)
    parser.add_argument('--processus',type=int,default=5)
    parser.add_argument('--durée',type=int,default=4800)
    parser.add_argument('--prolongé',action='store_true',help='Three new seeds per gas; sizes 256 and 500.')
    args = parser.parse_args()
    if args.duration_field < 2560 or args.duration_field % 40:
        parser.error('The duration must be a multiple of 40 and exceed 2,560.')
    parent, r460, cache = map(Path.resolve,(args.r454,args.r460,args.cache))
    sys.path.insert(0,str(parent))
    import transport_R454 as md

    md.WINDOWS = WINDOWS
    cache.mkdir(exist_ok=True,parents=True)
    calibration_path = r460 / 'resultats/TRANSPORT_COMMUN_R460.json'
    calibration = json.loads(calibration_path.read_text())
    binary, base_kernel = compile_reader(parent,cache)
    setups = {gas:gas_setup(gas,parent,cache,r460,calibration) for gas in ('Xe','Kr')}
    plans = [(4,.01,seed) for seed in SEEDS] + [(5,.01,SEEDS[0]), (4,.005,SEEDS[0])]
    if args.extended:
        plans = [(4,.01,seed) for seed in (46109,46183,46197)] + [(5,.01,46109)]
    jobs = [(md,binary,cache,gas,setup,seed,cells,step,args.duration_field)
            for cells,step,seed in plans for gas,setup in setups.items()]
    with ThreadPoolExecutor(max_workers=args.process) as pool:
        results = list(pool.map(trial,jobs))
    limits = {}
    for gas in setups:
        group = [row['cases']['trajectory_entière'] for row in results
                 if row['gas'] == gas and row['step'] == .01 and row['N'] == 256]
        limits[gas] = {name:np.stack((np.min([r[name] for r in group],axis=0),
                                     np.max([r[name] for r in group],axis=0)),axis=-1).tolist()
                       for name in ('G_K','Helfand','diffusion_origines_multiples_m2_s')}
    report = dict(revision=461, pressure_target_Pa=PRESSURE, temperature_target_K=TEMPERATURE,
                  windows_shear=WINDOWS.tolist(), windows_diffusion=DIFFUSION_WINDOWS,
                  densities_targets_mol_m3={gas:setup[3] for gas,setup in setups.items()},
                  cases=results, ranges_seeds_N256=limits,
                  references=dict(diffusion_Xe_m2_s=1.93e-6, source_Xe='https://doi.org/10.1006/jmre.1999.1850',
                                  frame_Xe_viscosité=[22.83,22.96], comparison_Kr='numerical prediction in the same nominal state'),
                  checksums=dict(fit=digest(calibration_path), kernel_R454=base_kernel,
                                  potentials={gas:setup[-1] for gas,setup in setups.items()}))
    filename = 'TRANSPORT_KR_XE_LONG_R461.json' if args.extended else 'TRANSPORT_KR_XE_R461.json'
    (ROOT/'results'/filename).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
