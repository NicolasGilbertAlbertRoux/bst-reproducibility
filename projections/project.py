"""Build two diffuse-envelope readings and one fine site reading of the same trajectories."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, label
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from trajectories import FRAMES,load_pair,strain_series,verify_sources

WIDTH,HEIGHT=512,248
ZOOM=96.0                      # Pixels per reference atomic radius in the molecular view.
GRID=59
EXTENT=1.85                    # Local voxel half-width in reference atomic radii.
READINGS=(.76,1.02)            # Gaussian standard deviations, in nearest-site spacings.
import os
DATA_OUT=Path(os.environ.get('BST_PROJECTION_WORK', Path(__file__).resolve().parent.parent/'projection-output'))/'cache'
DATA_OUT.mkdir(parents=True,exist_ok=True)
def deposit(x,w,step,n):
    """Cloud-in-cell deposit on a Cartesian grid; grid boundary is outside the sites."""
    f=x/step+(n-1)/2;indices=np.floor(f).astype(int);frac=f-indices
    assert indices.min()>=0 and indices.max()+1<n
    shape=(n,)*x.shape[1];out=np.zeros(shape,dtype=np.float32)
    for delta in np.ndindex(*(2,)*x.shape[1]):
        d=np.array(delta);weights=w*np.prod(np.where(d,frac,1-frac),axis=1)
        np.add.at(out,tuple((indices+d).T),weights)
    return out
def volume(x,w,step,sigma):
    return gaussian_filter(deposit(x,w,step,GRID),sigma/step,mode='constant',truncate=3.75)
def plane_on_screen(plane,centre,step,reference_radius):
    """Sample a projected local grid at its actual molecular centre on a fixed camera."""
    scale=ZOOM/reference_radius;yy,xx=np.ogrid[:HEIGHT,:WIDTH]
    mx=(xx-(WIDTH-1)/2-scale*centre)/(scale*step)+(plane.shape[1]-1)/2
    my=(yy-(HEIGHT-1)/2)/(scale*step)+(plane.shape[0]-1)/2
    ys,xs=np.broadcast_arrays(my,mx)
    return map_coordinates(plane,(ys,xs),order=1,mode='constant',prefilter=False)
def fine_projection(x,speed,spacing,reference_radius,density_scale):
    n=169;step=2*EXTENT*reference_radius/(n-1)
    depth=np.clip(.56+.30*x[:,2]/reference_radius,.16,1.)
    motion=(1-np.exp(-np.minimum(8.,speed/density_scale)))*depth
    impulse=deposit(x[:,:2],motion,step,n)
    rr=.33*spacing/step;ks=int(np.ceil(2.3*rr))
    u,v=np.mgrid[-ks:ks+1,-ks:ks+1];distance=np.hypot(u,v)
    kernel=(.70*np.exp(-.5*((distance-rr)/(.25*rr+.45))**2)+.15*np.exp(-.5*(distance/(rr+.15))**2))
    img=fftconvolve(impulse,kernel.astype('f'),mode='same')
    return np.flipud(np.maximum(img.T,0.)).astype('f'),step

def build(species):
    verify_sources();p=load_pair(species);strains=strain_series(p)
    R=p.equilibrium/2;step=2*EXTENT*R/(GRID-1)
    spacing=float(np.median(cKDTree(p.rest[0]).query(p.rest[0],k=2)[0][:,1]))
    sigma=np.array(READINGS)*spacing
    reflevels=[];component_counts=[];mirrors=[]
    shape_stats=[]
    for a in range(2):
        per=[]
        for sm in sigma:
            ref=volume(p.rest[a],np.ones(len(p.surface)),step,sm)
            index=p.rest[a,p.surface].T/step+(GRID-1)/2
            level=float(np.median(map_coordinates(ref,index,order=1)))
            per.append(level)
            count=label(ref>=level)[1];component_counts.append(count)
            inv=ref[::-1,::-1,::-1]
            mirrors.append(np.linalg.norm(ref-inv)/np.linalg.norm(ref))
            assert level>0 and count==1
        reflevels.append(per)
    internal_speed=np.linalg.norm(p.local_rate,axis=-1)
    fine_scale=max(1e-13,np.quantile(internal_speed,.9))
    all_planes=np.empty((FRAMES,3,HEIGHT,WIDTH),np.float32)
    for i in range(FRAMES):
        fields=np.zeros((3,HEIGHT,WIDTH),np.float32)
        for a in range(2):
            x=p.local[i,a];centre=(a-.5)*p.separation[i]
            fine,fstep=fine_projection(x,internal_speed[i,a],spacing,R,fine_scale)
            fields[0]+=plane_on_screen(fine,centre,fstep,R)
            for level,sm in enumerate(sigma):
                rho=volume(x,np.ones(len(x)),step,sm)
                weighted=volume(x,strains[i,a],step,sm)
                rho0=reflevels[a][level]
                ratio=rho/rho0;activity=weighted/np.maximum(rho,1e-11)
                # A diffuse band about the reference surface. No radial or forward-edge mask.
                width=(.20,.265)[level]
                band=np.exp(-.5*((ratio-1)/width)**2)*np.minimum(ratio,1.)**2
                intensity=np.sum(band*activity,axis=2)*step/R
                projected=np.flipud(intensity.T)
                fields[level+1]+=plane_on_screen(projected,centre,step,R)
                shape_stats.append(label(rho>=rho0)[1])
        all_planes[i]=fields
    assert set(shape_stats)=={1}
    # A single exposure for each species' two mantle readings; dry frames always stay black.
    # Fine and mantle channels are different observables and have separate fixed exposures.
    shell=all_planes[:,1:]
    shell_ref=float(np.quantile(shell[shell>.015*shell.max()],.93))
    interior=all_planes[:,0];inner_ref=float(np.quantile(interior[interior>.02*interior.max()],.93))
    normalized=np.empty_like(all_planes,dtype='uint8')
    normalized[:,0]=np.rint(255*(-np.expm1(-1.72*interior/inner_ref))).astype('uint8')
    normalized[:,1:]=np.rint(255*(-np.expm1(-1.80*shell/shell_ref))).astype('uint8')
    p.info.update(gaussian_widths_A=sigma.tolist(),nearest_spacing_A=spacing,
                  reference_density_levels=reflevels,reference_inversion_residuals=[float(v) for v in mirrors],
                  rest_component_counts=component_counts,dynamic_connectivity_checks=len(shape_stats),
                  dynamic_component_range=[min(shape_stats),max(shape_stats)],
                  fixed_mantle_exposure=shell_ref,fixed_interior_exposure=inner_ref,
                  visual_time='one prepared radial cycle synchronized to an independent native atomic window',
                  optical_meaning=('mechanical false colour' if species=='Kr2' else 'measured band; hypothetical spatial weight'))
    np.savez_compressed(DATA_OUT/(species+'.npz'),channels=normalized,separation=p.separation,
                        phase=p.phase,info=json.dumps(p.info,indent=2))
    return p.info
if __name__=='__main__':
    import sys
    for name in sys.argv[1:] or ['Kr2','C2']:
        info=build(name);print(name,info['reference_inversion_residuals'],info['fixed_mantle_exposure'],flush=True)
