'Re-evaluate the same nodal action with the spatially gated inertia law.'

import argparse
from pathlib import Path

from sweep_sf4 import read
from fields_and_references import dump
from fit_core import Bank,GATES
from kinetic_local import LocalKinetic,LocalRadius,PARENT,SOURCE_K,SOURCE_R
from dynamics_size import (Molecule,OUTPUT as PREVIOUS,ROOT,Site,core_projection,errors,
                                    field_inner,prepare_motion,work_and_covariance)
from trials_nodal import digest
from solve_acceleration_nodal import constrained_inverse
from scope_spatial import GRID,PADDING,halo_patch
from dynamics_nodal_scripts import VACUUM,pad_cube
from trajectories_moving import history_bounds,timeline
import numpy as np
from scipy.spatial.transform import Rotation

OUTPUT = ROOT/'results'/'R358'
CALIBRATION = PREVIOUS/'COURBES_INERTIE_COEUR_R357.json'
GATE = (.50,.90)
DURATION = .12
CASES = {'SF4':(.04,.02),'SF6':(.04,.02,.01)}


class LocalMolecule(Molecule):
    def __init__(self,species):
        super().__init__(species)
        self.local_law = next(row for row in read(CALIBRATION)['gate_catalog_all_results']
                              if tuple(row['thresholds'])==GATE)
        self.p,self.ref = (self.local_law[k] for k in ('exponent','reference_radius_native'))
        self.kernel = LocalKinetic(self.beta,*GATE)
        self.radius = LocalRadius(*GATE)

    def amplitude(self,radius):
        return (radius/self.ref)**self.p-1.

    def site(self,task):
        n,t,h,v,gu,load = task
        R,cusp,edge,dr = self.radius(n,h,t)
        a = self.amplitude(R)
        dadr = self.p*(1.+a)/R
        adot = dadr*field_inner(dr,t)
        raw = np.r_[np.zeros(3),v]
        Q,gn,gt,gv,Qg = self.kernel.quadratic(n,h,t,raw,a)
        gn += Qg*dadr*dr
        speed2 = np.sum(t*t,axis=1)[:,None]
        angle = self.jet*np.sqrt(speed2)
        cos,sin = np.cos(angle),self.jet*np.sinc(angle/np.pi)
        np_,nm = n*cos+t*sin,n*cos-t*sin
        ap,am = (self.amplitude(self.radius(z,h)[0]) for z in (np_,nm))
        plus = self.kernel.quadratic(np_,h,t*cos-n*speed2*sin,raw,ap)
        minus = self.kernel.quadratic(nm,h,t*cos+n*speed2*sin,raw,am)
        pt,pv = .5*gt,.5*gv[3:]
        gt_dot = (plus[2]-minus[2])/(2.*self.jet)
        gv_dot = (plus[3][3:]-minus[3][3:])/(2.*self.jet)
        coadjoint = np.r_[np.cross(pv[:3],v[:3])+np.cross(pv[3:],v[3:]),np.cross(pv[3:],v[:3])]
        mask = self.mask
        ni = n[mask]
        rn = core_projection(ni,(.5*(gt_dot-gn)+gu)[mask])
        rb = .5*gv_dot-coadjoint-load
        A,C,G = self.kernel.mass(n,h,a)
        A = A[mask]
        C,G = np.ascontiguousarray(C[mask,:,3:]),G[3:,3:]
        I = constrained_inverse(A,ni)
        H = I@C
        Cf = C.reshape(-1,6)
        D = G-Cf.T@H.reshape(-1,6)
        D = .5*(D+D.T)
        minimum = np.linalg.eigvalsh(D)[0]
        if minimum<=0.:
            raise ArithmeticError('Loss of positive local site Schur block')
        inv_rn = np.einsum('nij,nj->ni',I,rn)
        dv = np.linalg.solve(D,-rb+Cf.T@inv_rn.ravel())
        tangential = -inv_rn-H@dv
        acc = np.zeros_like(n)
        acc[mask] = tangential-ni*speed2[mask]
        fn = core_projection(ni,rn+np.einsum('nij,nj->ni',A,tangential)+C@dv)
        fb = rb+G@dv+Cf.T@tangential.ravel()
        residual = np.array([np.max(abs(fn)),np.max(abs(fb))])
        hdot = field_inner(.5*gn+gu,t)+field_inner(pt,acc)+float(pv@dv)-float(load@v)
        return Site(.5*Q,pv,R,1.+a,adot,abs(.5*Qg*dadr)*cusp,edge,acc,dv,residual,hdot,minimum)

    def energy(self,y):
        y,_ = self.project(y)
        s = self.unpack(y)
        O = Rotation.from_quat(s.q).as_matrix()
        U,_,_,_,_ = self.potential(s.n,s.x,O)
        def kinetic(z):
            n,t,h,v = z
            a = self.amplitude(self.radius(n,h)[0])
            return .5*self.kernel.quadratic(n,h,t,np.r_[np.zeros(3),v],a)[0]
        return float(U+sum(self.pool.map(kinetic,zip(s.n,s.t,self.h,s.v,strict=True))))

    def direct_checks(self,y):
        s = self.unpack(y)
        bank = Bank(self.beta)
        j = 1+list(map(tuple,GATES)).index(GATE)
        checks = []
        for i in (0,1):
            n,t,h,v = s.n[i],s.t[i],self.h[i],np.r_[np.zeros(3),s.v[i]]
            R,cusp,_,dr = self.radius(n,h,t)
            a = self.amplitude(R)
            Q,gn,gt,gv,Qg = self.kernel.quadratic(n,h,t,v,a)
            A,C,G = self.kernel.mass(n,h,a)
            r,b = bank(n,h)
            expected = b[0]+a*b[j]
            bank_defects = [abs(R-r[j]),np.max(abs(G[3:,3:]-expected))]
            mass_energy = np.einsum('ni,nij,nj->',t,A,t)+2.*np.einsum('ni,nij,j->',t,C,v)+v@G@v
            slope = field_inner(gn,t)-field_inner(gt,n*np.sum(t*t,axis=1)[:,None])
            radius_work = Qg*self.p*(1.+a)*field_inner(dr,t)/R
            diffs = []
            for eps in (2e-4,1e-4):
                def point(sign):
                    z = n+sign*eps*t
                    z /= np.linalg.norm(z,axis=1)[:,None]
                    rate = core_projection(z,t.copy())
                    rr = self.radius(z,h)[0]
                    return rr,self.kernel.quadratic(z,h,rate,v,self.amplitude(rr))[0]
                pp,pm = point(1.),point(-1.)
                rd,qd = (np.array(pp)-pm)/(2.*eps)
                diffs.append({'step':eps,'radius_derivative_defect':abs(rd-field_inner(dr,t)),
                              'kinetic_chain_rule_defect':abs(qd-slope-radius_work),
                              'defect_without_radius_covector':abs(qd-slope)})
            if max(bank_defects)>4e-9 or abs(mass_energy-Q)>1e-8 or max(d['kinetic_chain_rule_defect'] for d in diffs)>2e-8:
                raise ArithmeticError(f'Gated action bank or chain rule mismatch {bank_defects} {diffs}')
            checks.append({'site':i,'bank_defects':bank_defects,'quadratic_mass_energy_defect':abs(mass_energy-Q),
                           'abs_density_cusp_bound':cusp,'kinetic_radius_work':radius_work,'FD':diffs})
        return checks

    def remote_controls(self):
        i = 1
        n = pad_cube(self.n0[i],self.count,PADDING,VACUUM)
        h = self.h[i]
        grid = np.indices((GRID,)*3).reshape(3,-1).T-.5*(GRID-1)
        mask = np.linalg.norm(grid,axis=1)<4.
        R,*_ = self.radius(n,h)
        a = self.amplitude(R)
        A,_,_ = self.kernel.mass(n,h,a)
        base = A[mask].copy()
        control = []
        for x in (27.5,29.):
            p,_ = halo_patch(n,x,.20)
            rr,*_ = self.radius(p,h)
            B,_,_ = self.kernel.mass(p,h,self.amplitude(rr))
            defects = [abs(rr-R),np.max(abs(B[mask]-base))]
            if max(defects)>1e-12:
                raise ArithmeticError(f'Gated core responds to a disjoint near-vacuum patch {defects}')
            control.append({'patch_x_grid':x,'gated_radius_defect':defects[0],'core_mass_defect':defects[1]})
        return control


def run(species):
    model = LocalMolecule(species)
    path = OUTPUT/f'MOUVEMENT_LOCAL_{species}_R358.json'
    try:
        y = prepare_motion(model)
        dy,d = model.evaluate(y)
        work,cov = work_and_covariance(model,y,dy,d)
        checks = model.direct_checks(y)
        remote = model.remote_controls() if species=='SF6' else None
        excitation = d['H_native']-model.energy(model.rest())
        if excitation<=0.:
            raise ArithmeticError('Nonpositive prepared excitation')
        inputs = (Path(__file__),CALIBRATION,model.path,PARENT,SOURCE_K,SOURCE_R)
        out = {'revision':358,'full':False,'physical_validation':False,'species':species,
               'scope':'R357 gated inertia in N48 mobile SE(3) action, fixed outer faces',
               'gate':GATE,'N48_parameters':{k:model.local_law[k] for k in ('exponent','reference_radius_native','mass_unit_kg')},
               'duration_native':DURATION,'initial_excitation_native':excitation,
               'same_native_time_windows_for_step_comparisons':True,
               'fixed_faces_are_not_a_relaxed_radiative_boundary':True,
               'kinetic_checks':checks,'independent_H_and_potential_FD':work,
               'rigid_motion_covariance':cov,'disjoint_vacuum_control_at_rest':remote,
               'input_sha256':{str(f.relative_to(ROOT)):digest(f) for f in inputs},'runs':[]}
        dump(path,out)
        ends = []
        for dt in CASES[species]:
            end,forward = timeline(model,y,DURATION,dt,log=f'{species} gated forward {dt}')
            back,reverse = timeline(model,model.reverse(end),DURATION,dt,log=f'{species} gated return {dt}')
            out['runs'].append({'dt':dt,'forward':forward,'return':reverse,
                'forward_bounds':history_bounds(forward,excitation),'return_bounds':history_bounds(reverse,excitation),
                'reverse_state_errors':errors(model,y,model.reverse(back))})
            ends.append(end)
            if len(ends)>1:
                out['runs'][-1]['forward_endpoint_change_from_previous_dt'] = errors(model,ends[-2],end)
            dump(path,out)
        out['full'] = True
        out['clock_qualification'] = 'native times only; changing mass law changes time conversion'
        dump(path,out)
    finally:
        model.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('species',choices=CASES)
    run(parser.parse_args().species)
