'Linked-window search with exact native replay of individual ligand brackets.'

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path

from fields_and_references import Configuration, dump
from dynamics_nodal_scripts import ROOT
from trials_nodal import digest
from geometry_conditional import direction_chart
from readout_inertial import REFERENCE, lengths_and_angles
from precision_sum import AccurateSumAction
import numpy as np
from scipy.optimize import brentq


OUTPUT = ROOT/'results'/'R348'
SOURCE = Path(__file__).resolve()
R347 = ROOT/'results'/'R347'
R341 = ROOT/'results'/'SELECTION_RECALAGE_R341.json'
BETAS = np.array([.25,.5,1.,2.,4.])
GAMMAS = np.array([.125,.25,.5,1.,2.,4.,8.])
SCALES = np.linspace(1.55,2.60,43)
AMPLITUDES = np.linspace(-.8,.8,9)
LENGTH_TARGETS = np.array([REFERENCE['geometry']['r_ax_angstrom']]*2+[REFERENCE['geometry']['r_eq_angstrom']]*2)
LENGTH_UNC = REFERENCE['geometry']['listed_radius_uncertainty_angstrom']
ANGLE_UNC = REFERENCE['geometry']['listed_angle_uncertainty_deg']
SITES = ('Ax1','Ax2','Eq1','Eq2')
POSES = {'recue':0.,'angles_low':-.4,'angles_hauts':.4}


def read(path):
    return json.loads(path.read_text())


def passive_readers():
    'Quadratic certificates on [0,1] for the recorded position decimals.'
    path = R347/'LECTURES_TERMINAL_R347.json'
    rows = []
    for row in read(path)['rows']:
        coord = []
        for reader in ('direct_centers','kinetic_centers'):
            points = [[Fraction(str(x)) for x in point] for point in row[reader]['coordinates_native']]
            coord.append([[p-q for p,q in zip(point,points[0],strict=True)] for point in points[1:]])
        def squared_polynomial(i):
            a = coord[0][i]
            b = [y-x for x,y in zip(a,coord[1][i],strict=True)]
            dot = lambda x,y:sum(u*v for u,v in zip(x,y,strict=True))
            return [dot(a,a),2*dot(a,b),dot(b,b)]
        ratio = Fraction('1.05')
        c,b,a = [y-ratio*ratio*x for x,y in zip(squared_polynomial(0),squared_polynomial(1),strict=True)]
        ends = [Fraction(0),Fraction(1)]
        if a>0 and 0 < -b/(2*a) < 1:
            ends.append(-b/(2*a))
        minimum = min(c+b*x+a*x*x for x in ends)
        measured_upper = Fraction('1.649')/Fraction('1.643')
        rows.append({'count':row['count'],'time_native':row['time_native'],'reader_lambda_interval':[0.,1.],
                     'ax2_ax1_bound':float(ratio),'reference_ax2_ax1_upper':float(measured_upper),
                     'polynomial_coefficients_ascending_exact':[str(q) for q in (c,b,a)],
                     'quadratic_minimum_native_squared_exact':str(minimum),
                     'quadratic_minimum_native_squared':float(minimum),
                     'incompatible_on_this_reader_segment':minimum>0 and ratio>measured_upper})
    dump(OUTPUT/'LECTEURS_INTERVALLE_R348.json',{'rows':rows,'input_sha256':{str(path.relative_to(ROOT)):digest(path)},
         'interpretation':'the linear interpolation of the two native positions is a diagnostic slice only',
         'physical_validation':False})
    print('Lecteurs, bornes quadratiques',[r['quadratic_minimum_native_squared'] for r in rows],flush=True)


def field_F(model, index, coordinates):
    trial = model.m.trial
    source = trial.fields[index]
    _, b, norm, parts = trial.templates[index]
    amplitude = parts@coordinates
    angle = norm*amplitude
    n = source*np.cos(angle)[:,None]+b*(amplitude*np.sinc(angle/np.pi))[:,None]
    n[~trial.mask] = source[~trial.mask]
    return np.ascontiguousarray(n)


def directions(base, offset):
    if offset == 0:
        return base.copy()
    bisector = base[0]+base[1]
    bisector /= np.linalg.norm(bisector)
    pair_axes = [base[0]-base[1],base[2]-base[3]]
    result = []
    for start, axis in zip((0,2),pair_axes,strict=True):
        theta = .5*(np.arccos(np.clip(base[start]@base[start+1],-1,1))+np.deg2rad(offset))
        s = axis/np.linalg.norm(axis)*np.sin(theta)
        m = bisector*np.cos(theta)
        result.extend((m+s,m-s))
    return np.array(result)


class RadialSlice:
    def __init__(self, pose):
        self.configuration = Configuration()
        c = self.configuration
        c.potential(c.q0)
        self.model = c.model
        self.action = AccurateSumAction(0.)
        self.base_fields = np.array(c.state.fields)
        self.base_radii = self.model.last['radii'].copy()
        self.D = directions(self.model.last['D'], POSES[pose])
        self.local_D = np.einsum('nji,nj->ni',c.frames[1:],-self.D)
        self.qF = c.q0[52:60].reshape(4,2).copy()
        self.radial_origin = float(self.model.m.d['r0'])
        self.fpaths = []
        self.F_base_ports = []
        for j in range(4):
            i = j+1
            port = self.model.port(self.base_fields[i],i,self.local_D[j])
            tangent = c.state.tangent[i]
            direction = np.einsum('ni,ni->n',port['gradient_field'],tangent)@c.state.parts[i]
            self.fpaths.append(direction/np.linalg.norm(direction))
            self.F_base_ports.append(port['value'])
            if np.max(abs(field_F(self.model,i,self.qF[j])-self.base_fields[i]))>2e-13:
                raise ValueError('Ligand chart failed to replay the initial field')

    def terms(self, field, index):
        g = np.zeros_like(field)
        values = np.zeros(3,dtype=np.longdouble)
        if self.action.f(round(len(field)**(1/3)),self.model.h(index),1.,0.,field,g,values):
            raise ValueError('Gnomonic action chart is not qualified')
        return values[1:]

    def port_terms(self, field, index, direction, own=None):
        if own is None:
            own = self.terms(field,index)
        port = self.model.port(field,index,direction)
        b = self.model.reference(index)
        tangent = b-field*np.einsum('ij,ij->i',field,b)[:,None]
        generator = tangent*port['p'][:,None]**4
        vector = self.model.eps*generator
        length = np.linalg.norm(vector,axis=1)
        c = field*np.cos(length)[:,None]
        s = vector*np.sinc(length/np.pi)[:,None]
        raw = -2*own+self.terms(c+s,index)+self.terms(c-s,index)
        return float(port['value']), np.asarray(raw/self.model.eps**2,dtype=float), np.asarray(own,dtype=float)

    def sulphur(self,j):
        return self.port_terms(self.base_fields[0],0,self.D[j])

    def fluor(self,j,a):
        q = self.qF[j]+a*self.fpaths[j]
        bounds = self.model.state.system.bounds[52+2*j:54+2*j]
        if np.any(q<bounds[:,0]) or np.any(q>bounds[:,1]):
            raise ValueError('Ligand preparation outside its numeric chart')
        return self.port_terms(field_F(self.model,j+1,q),j+1,self.local_D[j])

    def center_chart(self,radii):
        chart = self.configuration.q0[60:84].copy()
        rows = chart[:12].reshape(4,3)
        rows[:,0] = np.log(radii/self.model.m.rbase)
        rows[:,1:] = direction_chart(self.model.m,self.D)
        return chart

    def force(self, j, radius, beta, gamma, s, f):
        _, well = self.model.m.trial.well.separation_energy_derivative(radius)
        weights = np.array([1.,beta])
        raw = np.array([s[1]@weights,f[1]@weights])
        if min(abs(raw))<1e-7:
            raise ValueError('The curvature chart at this point is not qualified')
        kappa = gamma*(sum(abs(raw))+self.model.pair_curvature)
        delta = radius-self.radial_origin-s[0]-f[0]
        return float(well+kappa*delta),float(kappa)


def scan(pose):
    slice_ = RadialSlice(pose)
    c = slice_.configuration
    m = slice_.model
    beta_values, gamma_values = c.beta*BETAS,c.gamma*GAMMAS
    with ThreadPoolExecutor(max_workers=4) as pool:
        S = list(pool.map(slice_.sulphur,range(4)))
        F = list(pool.map(lambda p:slice_.fluor(*p),[(j,a) for j in range(4) for a in AMPLITUDES]))
    F = [F[j*len(AMPLITUDES):(j+1)*len(AMPLITUDES)] for j in range(4)]
    weights = np.column_stack((np.ones(len(beta_values)),beta_values))
    signed_S = weights@np.array([x[1] for x in S]).T
    signed_F = np.einsum('ba,jka->bjk',weights,np.array([[x[1] for x in row] for row in F]))
    kappa = abs(signed_S[:,:,None])+abs(signed_F)+m.pair_curvature
    rest = slice_.radial_origin + np.array([s[0] for s in S])[:,None]+np.array([[x[0] for x in row] for row in F])
    hold = len(AMPLITUDES)//2
    limits = [float(SCALES[0]*min(LENGTH_TARGETS-LENGTH_UNC)),float(SCALES[-1]*max(LENGTH_TARGETS+LENGTH_UNC))]
    rows = []
    brackets = []
    hits = []
    for b, beta in enumerate(beta_values):
        for g, gamma in enumerate(gamma_values):
            native_radii = []
            for j in range(4):
                def frozen_force(r):
                    return slice_.force(j,r,beta,gamma,S[j],F[j][hold])[0]
                try:
                    r = brentq(frozen_force,*limits,xtol=1e-12)
                except ValueError:
                    native_radii = []
                    break
                native_radii.append(r)
            valid = (abs(signed_S[b,:,None])>=1e-7)&(abs(signed_F[b])>=1e-7)
            index = [b,g]
            row = {'beta':float(beta),'gamma':float(gamma),'index':index,
                   'frozen_radial_root_radii':native_radii,'numeric_root_bracket':limits,
                   'frozen_radial_root_reading':(lengths_and_angles(np.vstack((np.zeros(3),np.array(native_radii)[:,None]*slice_.D))) if native_radii else None)}
            rows.append(row)
            for k,scale in enumerate(SCALES):
                rr = scale*(LENGTH_TARGETS[:,None]+LENGTH_UNC*np.array([0.,-1.,1.]))
                well = np.array([[m.m.trial.well.separation_energy_derivative(r)[1] for r in radii] for radii in rr])
                forces = well[:,:,None]+gamma*kappa[b,:,None,:]*(rr[:,:,None]-rest[:,None,:])
                crossing = ((forces[:,:,:-1]*forces[:,:,1:]<=0.)&valid[:,None,:-1]&valid[:,None,1:])
                each = []
                for j in range(4):
                    choices = []
                    for h,a in np.argwhere(crossing[j]):
                        y,z = forces[j,h,a:a+2]
                        estimate = AMPLITUDES[a] if y==z else AMPLITUDES[a]-(AMPLITUDES[a+1]-AMPLITUDES[a])*y/(z-y)
                        choices.append({'radius':float(rr[j,h]),'radius_choice':int(h),'interval_index':int(a),
                                        'estimate':float(estimate)})
                    each.append(choices)
                if not all(each):
                    continue
                chosen = [min(options,key=lambda r:(r['radius_choice']!=0,abs(r['estimate']))) for options in each]
                hit = {'pose':pose,'indices':[b,g,k],'beta':float(beta),'gamma':float(gamma),'scale_native_per_angstrom':float(scale),
                       'maximum_abs_estimated_preparation':max(abs(r['estimate']) for r in chosen),
                       'estimated_preparation_l2':float(np.linalg.norm([r['estimate'] for r in chosen])),
                       'samples_at_nominal_measurement':sum(r['radius_choice']==0 for r in chosen),
                       'selected_brackets':chosen,'all_per_ligand_brackets':each}
                hits.append(hit)
    order = lambda h:(-h['samples_at_nominal_measurement'],h['maximum_abs_estimated_preparation'],h['estimated_preparation_l2'])
    hits.sort(key=order)
    baseline = next((h for h in hits if h['indices'][:2]==[2,3]),None)
    result = {'revision':348,'pose':pose,'fields_and_axes_before_radial_search':'R345/R347, with declared angle offset',
              'axes':slice_.D,'F_paths':slice_.fpaths,'F_initial':slice_.qF,'beta_values':beta_values,'gamma_values':gamma_values,
              'scales_native_per_angstrom':SCALES,'amplitudes_native':AMPLITUDES,
              'measurement_radial_samples':'lower, central and upper per site, jointly linked by one scale',
              'cells':len(beta_values)*len(gamma_values)*len(SCALES),'S_terms':S,'F_terms':F,'frozen_rows':rows,
              'cells_with_sampled_sign_brackets':len(hits),'best_frozen_laws_bracket':baseline,'bracket_cells':hits,
              'unsampled_or_unbracketed_regions_retained':True,'physical_regions_excluded':0,
              'input_sha256':{str(R341.relative_to(ROOT)):digest(R341)},'source_sha256':digest(SOURCE)}
    dump(OUTPUT/f'BALAYAGE_{pose.upper()}_R348.json',result)
    fixed = [r['frozen_radial_root_reading']['minimum_possible_max_relative_length_error_over_common_scale']
             for r in rows if r['frozen_radial_root_reading'] is not None]
    print(pose,'cells',result['cells'],'bracketed',len(hits),'frozen best',min(fixed,default=None),
          'best preparation',None if not hits else hits[0]['maximum_abs_estimated_preparation'],flush=True)


def replay():
    scans = [read(OUTPUT/f'BALAYAGE_{pose.upper()}_R348.json') for pose in POSES]
    all_hits = [hit for scan_ in scans for hit in scan_['bracket_cells']]
    order = lambda h:(-h['samples_at_nominal_measurement'],h['maximum_abs_estimated_preparation'],h['estimated_preparation_l2'])
    proposals = [('lois_recues',scans[0]['best_frozen_laws_bracket']),
                 ('lois_balayees',min((hit for hit in all_hits if hit['indices'][:2]!=[2,3]),key=order,default=None))]
    controls = []
    for label, hit in proposals:
        if hit is None:
            continue
        slice_ = RadialSlice(hit['pose'])
        radii = np.array([row['radius'] for row in hit['selected_brackets']])
        roots, radial, kappa = [], [], []
        for j,row in enumerate(hit['selected_brackets']):
            s = slice_.sulphur(j)
            def force(a):
                return slice_.force(j,radii[j],hit['beta'],hit['gamma'],s,slice_.fluor(j,a))[0]
            i = row['interval_index']
            root = brentq(force,*AMPLITUDES[i:i+2],xtol=3e-13)
            f,k = slice_.force(j,radii[j],hit['beta'],hit['gamma'],s,slice_.fluor(j,root))
            roots.append(root)
            radial.append(f)
            kappa.append(k)
        center = np.vstack((np.zeros(3),radii[:,None]*slice_.D))
        reading = lengths_and_angles(center)
        controls.append({'label':label,'pose':hit['pose'],'beta':hit['beta'],'gamma':hit['gamma'],
                         'kangle':float(slice_.configuration.kangle),'scale_native_per_angstrom':hit['scale_native_per_angstrom'],
                         'F_preparations':slice_.qF+np.array(roots)[:,None]*slice_.fpaths,'F_paths':slice_.fpaths,
                         'amplitudes':roots,'center_chart':slice_.center_chart(radii),'reading':reading,
                         'radial_residuals_at_rest':radial,'kappa_native':kappa,
                         'all_six_angles_inside_published_intervals':bool(max(abs(reading['six_angle_errors_deg']))<=ANGLE_UNC),
                         'individual_lengths_inside_intervals':bool(np.all(abs(radii/hit['scale_native_per_angstrom']-LENGTH_TARGETS)<=LENGTH_UNC+1e-12)),
                         'physical_validation':False})
        print(label,'pose',hit['pose'],'a',roots,'F_rad',radial,flush=True)
    if not controls:
        raise RuntimeError('No bracket could be replayed')
    dump(OUTPUT/'CANDIDATS_R348.json',{'revision':348,'candidates':controls,'source_sha256':digest(SOURCE),
         'input_sha256':{str((OUTPUT/f'BALAYAGE_{pose.upper()}_R348.json').relative_to(ROOT)):digest(OUTPUT/f'BALAYAGE_{pose.upper()}_R348.json') for pose in POSES}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('lecteurs','replay',*POSES))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.action=='lecteurs':
        passive_readers()
    elif args.action=='replay':
        replay()
    else:
        scan(args.action)
