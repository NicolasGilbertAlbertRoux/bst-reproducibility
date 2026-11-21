'Four actual trial coordinates in the current nonlinear molecular potential.\nNo experimental angle/length target is used. Not a global BST feasibility solver.'
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from scipy.optimize import brentq


class NativeAction:
    def __init__(self,inputs):
        source=inputs/'action_native_R321.c'
        self.hash=hashlib.sha256(source.read_bytes()).hexdigest()
        suffix='.dylib' if sys.platform=='darwin' else '.so'
        target=inputs/('action_R321_'+self.hash[:12]+suffix)
        if not target.exists():
            compiler=shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
            if not compiler:raise RuntimeError('A C compiler is required for the unchanged native action')
            mode=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
            subprocess.run([compiler,'-O3',*mode,str(source),'-lm','-o',str(target)],check=True)
        f=ctypes.CDLL(str(target)).bst_action
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        f.argtypes=[ctypes.c_int,ctypes.c_double,array,array,ctypes.POINTER(ctypes.c_double)];f.restype=ctypes.c_int
        self.f=f

    def __call__(self,field,h):
        field=np.ascontiguousarray(field);gradient=np.zeros_like(field);out=ctypes.c_double()
        n=round((field.size/4)**(1/3))
        if self.f(n,h,field,gradient,ctypes.byref(out))!=0:raise ValueError('Outside native gnomonic chart')
        return out.value,gradient


class Trial:
    def __init__(self,inputs,label,action=None):
        self.inputs=inputs;self.label=label
        with np.load(inputs/f'{label}_ENTREES_NATIVES_R321.npz') as z:self.data={k:z[k] for k in z.files}
        d=self.data;self.fields=d['fields'];self.count=len(self.fields);self.mask=d['mask']
        self.action=action or NativeAction(inputs);sys.path.insert(0,str(inputs))
        from source import GlobalContactExtension
        contract=json.loads((inputs/'ENTREES_CONTACTS_R319.json').read_text())['well_contract']
        with np.load(inputs/'PUITS_SF_R319.npz') as z:self.well=GlobalContactExtension(z['fractions'],z['energies'],contract)
        self.templates=[]
        for i,field in enumerate(self.fields):
            symbol='S' if i==0 else 'F';b=d[symbol+'_b_reference']
            tangent=b-field*np.einsum('ij,ij->i',b,field)[:,None]
            tangent[~self.mask]=0.
            norm=np.linalg.norm(tangent,axis=1)
            self.templates.append((symbol,tangent,norm,d[symbol+'_parts']))
        self.calls=0;self.cache=None

    def reconstruct(self,q):
        result=[];derivatives=[]
        for i,(base,(symbol,b,norm,parts)) in enumerate(zip(self.fields,self.templates,strict=True)):
            coords=q[:2] if i==0 else q[2:]
            amp=parts@coords;theta=norm*amp
            field=base*np.cos(theta)[:,None]+b*(amp*np.sinc(theta/np.pi))[:,None]
            field[~self.mask]=base[~self.mask]
            derivative=-base*(norm*np.sin(theta))[:,None]+b*np.cos(theta)[:,None]
            derivative[~self.mask]=0.
            result.append(field);derivatives.append(derivative)
        return result,derivatives

    def evaluate(self,q):
        q=np.asarray(q,dtype=float)
        if self.cache is not None and np.array_equal(q,self.cache['q']):return self.cache['energy'],self.cache['gradient']
        fields,derivatives=self.reconstruct(q);d=self.data
        energy=float(d['Gram_energy']);raw=[];shifts=[];self_energies=[]
        for i,(field,(symbol,b,norm,parts)) in enumerate(zip(fields,self.templates,strict=True)):
            h=float(d[symbol+'_h']);e,g=self.action(field,h);self_energies.append(e);energy+=e;raw.append(g)
            nodes=d[symbol+'_node_offset']+h**3*np.einsum('ij,ij->i',d[symbol+'_b_reference'],field)
            shifts.append(nodes@d[f'P_{i}']/d[f'normal_{i}'])
        anchors=float(d['r0'])+shifts[0]+np.array([s[0] for s in shifts[1:]])
        radii=[];reaction=[];force_defect=0.
        for k,anchor in zip(d['stiffness'],anchors,strict=True):
            def force(r):return self.well.separation_energy_derivative(r)[1]+k*(r-anchor)
            lo=max(.05,anchor-.6);hi=anchor+.6
            if lo>=hi:raise ValueError('Undefined received radial root bracket')
            r=brentq(force,lo,hi,xtol=1e-13);delta=r-anchor
            energy+=self.well.separation_energy_derivative(r)[0]+.5*k*delta*delta
            radii.append(r);reaction.append(-k*delta);force_defect=max(force_defect,abs(force(r)))
        reaction=np.asarray(reaction);gradient=np.zeros(4);full_tangent_norm2=0.;phase_norm2=0.
        for i,(field,derivative,(symbol,b,norm,parts),g) in enumerate(zip(fields,derivatives,self.templates,raw,strict=True)):
            h=float(d[symbol+'_h']);local=reaction if i==0 else reaction[i-1:i]
            g=g+h**3*d[symbol+'_b_reference']*(d[f'P_{i}']@(local/d[f'normal_{i}']))[:,None]
            tangent=g-field*np.einsum('ij,ij->i',field,g)[:,None]
            tangent[~self.mask]=0.;full_tangent_norm2+=float(np.sum(tangent*tangent))
            nv=field[:,1:];gv=g[:,1:];nn=np.sum(nv*nv,axis=1)
            phase=gv-nv*np.divide(np.sum(nv*gv,axis=1),nn,out=np.zeros(len(nn)),where=nn>1e-24)[:,None]
            phase[~self.mask]=0.;phase_norm2+=float(np.sum(phase*phase))
            pull=np.einsum('ij,ij->i',g,derivative)@parts
            if i==0:gradient[:2]+=pull
            else:gradient[2:]+=pull
        self.calls+=1
        self.cache={'q':q.copy(),'energy':float(energy),'gradient':gradient,'radii':np.asarray(radii),
                    'fields':fields,'self_energies':self_energies,'radial_force_max':force_defect,
                    'full_unrestricted_S3_tangent_gradient_l2':float(np.sqrt(full_tangent_norm2)),
                    'fixed_scalar_phase_gradient_l2':float(np.sqrt(phase_norm2))}
        return float(energy),gradient

    def geometry(self,q):
        self.evaluate(q);r=self.cache['radii'];ds=self.data['directions_s']
        angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(ds[2],ds[3])),np.dot(ds[2],ds[3]))) if self.label=='SF4' else None
        result={'radii_native':r.tolist(),'bond_spread_relative':float(np.ptp(r)/np.mean(r)),'equatorial_angle_deg_held':angle}
        if self.label=='SF4':result['axial_equatorial_ratio']=float(np.mean(r[:2])/np.mean(r[2:]))
        return result

    def invariants(self,q):
        fields,_=self.reconstruct(np.asarray(q));scalar=0.;normdefect=0.;border=0.
        for field,base in zip(fields,self.fields,strict=True):
            normdefect=max(normdefect,float(np.max(abs(np.sum(field*field,axis=1)-np.sum(base*base,axis=1)))))
            border=max(border,float(np.max(abs(field[~self.mask]-base[~self.mask]))))
            scalar=max(scalar,float(np.max(abs(field[:,0]-base[:,0]))))
        return {'S3_norm_abs_defect':normdefect,'frozen_three_layers_abs_defect':border,
                'scalar_profile_change_max':scalar,'scalar_profile_change_is_explicit_trial_extension':True}

    def hemisphere_box_bound(self,limit):
        # A common open hemisphere in every cell keeps normalised interpolation
        # defined along the full homotopy, relative to the current prepared field.
        shape=tuple(self.data['grid_shape']);records=[]
        from itertools import product
        for base,(symbol,b,norm,parts) in zip(self.fields,self.templates,strict=True):
            n=base.reshape(*shape,4);anchor=n[:-1,:-1,:-1]
            max_rotation=(limit*np.sum(parts,axis=1)*norm).reshape(shape)
            maximum=0.
            for bits in product((0,1),repeat=3):
                corner=tuple(slice(1,None) if bit else slice(None,-1) for bit in bits)
                dots=np.einsum('...i,...i->...',anchor,n[corner])
                angle=np.arccos(np.clip(dots,-1,1))+max_rotation[corner]
                maximum=max(maximum,float(np.max(angle)))
            records.append({'species':symbol,'cube_path_angle_upper_rad':maximum,'hemisphere_margin_rad':float(np.pi/2-maximum),
                            'whole_declared_box_hemisphere_condition_pass':maximum<np.pi/2})
        return records
