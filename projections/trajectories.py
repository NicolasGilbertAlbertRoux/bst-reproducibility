"""Two illustrative molecular preparations and native atomic motions."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.constants import c, elementary_charge, h
from scipy.integrate import quad, solve_ivp
from scipy.linalg import null_space
from scipy.optimize import brentq
from scipy.spatial import ConvexHull, cKDTree

DATA = Path(__file__).resolve().parent / 'inputs'
FRAMES = 72
PHASE = np.arange(FRAMES) / FRAMES
azimuth, elevation = .59, .37
cam_xy = np.array([[np.cos(azimuth), -np.sin(azimuth), 0.],
                   [-np.sin(azimuth)*np.sin(elevation), -np.cos(azimuth)*np.sin(elevation), np.cos(elevation)]])
CAMERA = np.vstack((cam_xy, np.cross(*cam_xy)))
# All matrices here transform into a right-handed camera frame. Display y is flipped by the plotter.

@dataclass
class Pair:
    species: str
    rest: np.ndarray
    surface: np.ndarray
    local: np.ndarray
    local_rate: np.ndarray
    separation: np.ndarray
    separation_rate: np.ndarray
    phase: np.ndarray
    equilibrium: float
    info: dict

def verify_sources():
    records=json.loads((DATA/'sources.json').read_text())
    for name,record in records.items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==record['sha256'],name

def krypton_radial():
    unit=json.loads((DATA/'krypton_units.json').read_text())
    sigma=unit['sigma_m']*unit['size_ratio']
    epsilon=unit['energie_J']/unit['size_ratio']**6
    length,amp,p0,p1,p2,p3,p4,p5=np.loadtxt(DATA/'krypton_pair.dat')[:8]
    def potential(r):
        x=r/length
        return amp*(p0*np.exp(p1*x)/x+p2*np.exp(p3*x)+p4)/(1+p5*x**6)
    def slope(r):
        x=r/length;top=p0*np.exp(p1*x)/x+p2*np.exp(p3*x)+p4
        dt=p0*np.exp(p1*x)*(p1/x-1/x**2)+p2*p3*np.exp(p3*x);bottom=1+p5*x**6
        return amp/length*(dt*bottom-6*top*p5*x**5)/bottom**2
    eq=brentq(slope,.9,1.5);larger=eq*1.11;e0=potential(larger)
    smaller=brentq(lambda x:potential(x)-e0,.8*eq,eq)
    reduced_mass=.5
    def dt(a):
        r=(larger+smaller)/2+(larger-smaller)*np.cos(a)/2
        return (larger-smaller)*np.sin(a)/2/np.sqrt(2*(e0-potential(r))/reduced_mass)
    T=2*quad(dt,0,np.pi,epsrel=1e-10,epsabs=2e-10)[0]
    phase=PHASE+12/92
    sol=solve_ivp(lambda t,y:[y[1],-slope(y[0])/reduced_mass],(0,2*T),[larger,0.],
                  method='DOP853',rtol=2e-12,atol=2e-13,max_step=T/300,dense_output=True).sol
    curve,velocity=sol(phase*T);ang=sigma*1e10
    T_s=T*sigma*np.sqrt(unit['masse_kg']/epsilon)
    energy_err=np.ptp(reduced_mass*velocity**2/2+potential(curve))
    relative_excitation=(e0-potential(eq))*epsilon/elementary_charge
    assert energy_err<1e-10 and 0<relative_excitation<.005
    return eq*ang,curve*ang,velocity*T*ang,phase,dict(period_ps=T_s*1e12,
        turning_A=[smaller*ang,larger*ang],chosen_outer_ratio=1.11,
        radial_energy_above_minimum_meV=relative_excitation*1000,frequency_THz=1e-12/T_s,
        fundamental_spacing_meV=h/T_s/elementary_charge*1000,
        stored_energy_span_reduced=energy_err)

def carbon_hessian(x):
    """Hessian of the archived V65 reduced pair energy r^-12/12 - r^-1."""
    n=len(x);H=np.zeros((3*n,3*n))
    for i in range(n):
        for j in range(i):
            dx=x[i]-x[j];r=np.linalg.norm(dx);u=dx/r
            first=-r**(-13)+r**(-2);second=13*r**(-14)-2*r**(-3)
            block=(second-first/r)*np.outer(u,u)+first/r*np.eye(3)
            a=slice(3*i,3*i+3);b=slice(3*j,3*j+3)
            H[a,a]+=block;H[b,b]+=block;H[a,b]-=block;H[b,a]-=block
    return H

def carbon_motion(x,surface):
    # Project out rigid translations and rotations before the local harmonic calculation.
    rigid=np.asarray([np.tile(e,(len(x),1)).ravel() for e in np.eye(3)]+
                     [np.cross(e,x).ravel() for e in np.eye(3)]).T
    complement=null_space(rigid.T);H=carbon_hessian(x)
    values,Q=np.linalg.eigh(complement.T@H@complement)
    assert values.min()>0 and np.max(abs(H-H.T))<1e-10
    basis=complement@Q;omega=np.sqrt(values)
    # A radial initial displacement is propagated by all positive modes, without a fitted phase law.
    initial=(.012*x).ravel();coefficients=basis.T@initial
    dom=np.argmax(coefficients**2);native_span=2*np.pi/omega[dom]
    native_time=(PHASE+.07)*native_span
    vib=np.cos(native_time[:,None]*omega)*coefficients
    derivative=-np.sin(native_time[:,None]*omega)*coefficients*omega*native_span
    disp=(vib@basis.T).reshape(FRAMES,len(x),3)
    rate=(derivative@basis.T).reshape(FRAMES,len(x),3)
    e=.5*np.sum(((derivative/native_span)**2+(vib*omega)**2),axis=1)
    assert np.ptp(e)/e.mean()<1e-12 and np.linalg.norm(disp.mean(axis=1),axis=1).max()<1e-12
    # Check Hessian independently against central finite differences of the V65 gradient.
    def gradient(a):
        delta=a[:,None]-a[None,:];r=np.linalg.norm(delta,axis=2);np.fill_diagonal(r,np.inf)
        return np.einsum('ij,ijk->ik',-r**(-14)+r**(-3),delta).ravel()
    direction=initial/np.linalg.norm(initial);eps=1e-6;v=direction.reshape(-1,3)
    fd=(gradient(x+eps*v)-gradient(x-eps*v))/(2*eps)
    defect=np.linalg.norm(fd-H@direction)/np.linalg.norm(fd)
    assert defect<1e-7
    return x+disp,rate,dict(positive_modes=int(len(values)),surface_sites=int(surface.sum()),
        local_radial_preparation=.012,internal_interval_native=native_span,mode_energy_relative_span=np.ptp(e)/e.mean(),
        hessian_fd_relative_error=defect)

def load_pair(species):
    if species=='Kr2':
        eq,d,dd,phase,report=krypton_radial()
        with np.load(DATA/'krypton_modes.npz') as raw:
            x=raw['positions'].copy();x-=x.mean(axis=0)
            modes=raw['modes'];omega=raw['omega_native']
        mask=np.zeros(len(x),dtype=bool);mask[ConvexHull(x).vertices]=True
        assert modes.shape==(8,309,3)
        span=2*np.pi/omega.min();pos=[];rate=[]
        for offset in (0.,.39):
            t=(PHASE+.079)*span+offset
            pos.append(x+np.einsum('fm,mna->fna',np.cos(t[:,None]*omega),modes))
            rate.append(np.einsum('fm,mna->fna',-span*omega*np.sin(t[:,None]*omega),modes))
        report.update(sites=309,positive_modes_replayed=8,internal_offsets_native=[0.,.39],
                      internal_interval_native=float(span),site_displacement_gain=1)
        transform=[CAMERA,CAMERA]
    else:
        with np.load(DATA/'carbon_geometry.npz') as model:
            x=model['positions'].copy();x-=x.mean(axis=0);mask=model['surface']
        p,v,report=carbon_motion(x,mask);pos=[p,p];rate=[v,v]
        # A preparation, not a claim about the orientation or brightness of a real emitting molecule.
        transform=[CAMERA,np.diag([-1.,1.,-1.])@CAMERA]
        eq=1.2661;wavenumber=1788.22;amplitude=.025;phase=PHASE+.17
        d=eq+amplitude*np.cos(2*np.pi*phase);dd=-2*np.pi*amplitude*np.sin(2*np.pi*phase)
        report.update(sites=len(x),equilibrium_A=eq,wavenumber_cm=wavenumber,
                      molecular_classical_amplitude_A=amplitude,period_fs=1e15/(c*100*wavenumber),
                      optical_band_heads_nm=[473.7,516.5,563.5],visible_head_eV=h*c/(516.5e-9)/elementary_charge)
    # One common scalar anchors the median surface radius to half of the reference separation.
    # This preserves the full shape, without assigning a measured radius to a BST site.
    radius_native=np.median(np.linalg.norm(x[mask],axis=1));scale=.5*eq/radius_native
    rest=np.stack([x@q.T*scale for q in transform])
    local=np.stack([p@q.T*scale for p,q in zip(pos,transform)],axis=1)
    speed=np.stack([v@q.T*scale for v,q in zip(rate,transform)],axis=1)
    assert np.linalg.norm(local.mean(axis=2),axis=2).max()<1e-9
    surface_r=np.linalg.norm(rest[0,mask],axis=1)
    report.update(equilibrium_A=float(eq),surface_radius_A_range=[float(surface_r.min()),float(surface_r.max())],
                  surface_anchor_A=float(eq/2),surface_markers=int(mask.sum()),native_length_scale_A=float(scale))
    return Pair(species,rest,mask,local,speed,d,dd,phase,float(eq),report)

def local_strain(positions,velocities,count):
    """Local affine strain-rate, invariant to a common translation or instantaneous rigid rotation."""
    dist,ind=cKDTree(positions).query(positions,k=count+1)
    dist=dist[:,1:];ind=ind[:,1:]
    dx=positions[ind]-positions[:,None];dv=velocities[ind]-velocities[:,None]
    weights=np.exp(-1.5*(dist/dist[:,-1:])**2)
    lhs=np.einsum('nk,nki,nkj->nij',weights,dx,dx)
    rhs=np.einsum('nk,nki,nkj->nij',weights,dx,dv)
    gradient=np.linalg.solve(lhs,rhs)
    return (gradient+np.swapaxes(gradient,1,2))*.5

def strain_series(pair):
    traces=[];max_boost=0.;max_rotation=0.
    for frame,(x,v) in enumerate(zip(pair.local,pair.local_rate)):
        x=x.copy();v=v.copy()
        x[:, :, 0]+=np.array([-1,1])[:,None]*pair.separation[frame]/2
        v[:, :, 0]+=np.array([-1,1])[:,None]*pair.separation_rate[frame]/2
        p=x.reshape(-1,3);speed=v.reshape(-1,3);n=24 if pair.species=='Kr2' else 14
        D=local_strain(p,speed,n)
        if frame in [0,FRAMES//3,FRAMES//2]:
            boost=np.array([6.,-7.,3.]);spin=np.cross([-.4,1.3,.7],p)
            max_boost=max(max_boost,np.max(abs(D-local_strain(p,speed+boost,n))))
            max_rotation=max(max_rotation,np.max(abs(local_strain(p,spin+boost,n))))
            assert np.max(abs(local_strain(p,.23*p,n)-.23*np.eye(3)))<1e-12
        traces.append(np.linalg.norm(D,axis=(1,2)).reshape(x.shape[:2]))
    assert max_boost<1e-11 and max_rotation<1e-11
    pair.info.update(max_boost_error=float(max_boost),max_rigid_rotation_norm=float(max_rotation))
    return np.stack(traces)
