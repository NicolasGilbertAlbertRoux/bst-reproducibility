"""Reference plates for the elementary beat and the 309-site krypton preparation."""
from dataclasses import dataclass
from pathlib import Path
from plate_labels import translate
from functools import lru_cache
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.special import j1
from project import DATA_OUT
from films import atomic_uv
from optics import spectral_colour
HERE=Path(__file__).resolve().parent
INPUT=HERE/'inputs'
OUT=DATA_OUT.parent/'movies';OUT.mkdir(parents=True,exist_ok=True)
REV=DATA_OUT.parent/'stills';REV.mkdir(parents=True,exist_ok=True)
FRAMES = 92
PHASE = np.arange(FRAMES)/FRAMES
GAIN = 12.0
azimuth,elevation=.59,.37
CAMERA=np.array([[np.cos(azimuth),-np.sin(azimuth),0.],
                 [-np.sin(azimuth)*np.sin(elevation),-np.cos(azimuth)*np.sin(elevation),np.cos(elevation)]])
CAMERA=np.vstack((CAMERA,np.cross(*CAMERA)))
@dataclass
class Atom:
    xy: np.ndarray
    covariance: np.ndarray
    energy: np.ndarray
    velocity: np.ndarray
    reference: np.ndarray
    strain: np.ndarray
    time: np.ndarray
    report: dict

def krypton(offset_native=0.):
    raw=np.load(INPUT/'krypton_modes.npz'); x=raw['positions'].copy()
    x-=x.mean(axis=0)
    modes=raw['modes']; omega=raw['omega_native']
    assert modes.shape==(8,309,3)
    # The atomic time is native. A finite record of all eight modes is replayed.
    T=2*np.pi/omega.min()
    t=(PHASE+.079)*T+offset_native
    displacement=np.einsum('fm,min->fin',np.cos(t[:,None]*omega),modes)
    velocity=np.einsum('fm,min->fin',-omega*np.sin(t[:,None]*omega),modes)
    centre_drift=np.linalg.norm(displacement.mean(axis=1),axis=1).max()
    # Twelve neighbours fit an affine deformation at each recorded site.
    distances,ids=cKDTree(x).query(x,k=13); neigh=ids[:,1:]
    basis=x[neigh]-x[:,None,:]; pinv=np.linalg.pinv(basis)
    dv=displacement[:,neigh]-displacement[:,:,None,:]
    local=np.einsum('nij,fnjk->fnik',pinv,dv)
    F=np.eye(3)[None,None]+GAIN*local
    jac=np.linalg.det(F)
    assert jac.min()>0.45 and np.all(np.linalg.matrix_rank(basis)==3)
    # A spherical readout volume at rest becomes an ellipsoid with the fitted
    # gradient. This is a visual interpolation and assigns no fundamental radius.
    radius=.47*distances[:,1]
    camera_map=np.einsum('pa,fnba->fnpb',CAMERA[:2],F)
    cov=np.einsum('fnpi,fnqi->fnpq',camera_map,camera_map)*radius[None,:,None,None]**2
    xy=np.einsum('pa,fna->fnp',CAMERA,x[None]+GAIN*displacement)
    vcam=np.einsum('pa,fna->fnp',CAMERA,velocity)
    activity=np.sum(velocity**2,axis=2)
    kinetic=.5*np.sum((raw['coefficients']*omega)[None,:]**2*np.sin(t[:,None]*omega)**2,axis=1)
    potential=.5*np.sum((raw['coefficients']*omega)[None,:]**2*np.cos(t[:,None]*omega)**2,axis=1)
    drift=np.ptp(kinetic+potential)/(kinetic+potential)[0]
    assert drift<1e-12
    spectrum_sha=hashlib.sha256((INPUT/'krypton_modes.npz').read_bytes()).hexdigest()
    return Atom(xy,cov,activity,vcam,x@CAMERA.T,jac,t,dict(sites=len(x),modes=len(modes),frequencies=omega.tolist(),
        input_sha256=spectrum_sha,max_native_centroid_drift=float(centre_drift),display_gain=GAIN,min_jacobian=float(jac.min()),
        max_jacobian=float(jac.max()),modal_energy_relative_drift=float(drift),internal_interval_native=t[[0,-1]].tolist()))

def mantle_paths():
    mu,kappa,k,G=.3,2.,3.,.4; radius0=.8; omega=np.sqrt(kappa/mu); db0=np.sqrt(2*.30/mu)
    time=PHASE*2*2*np.pi/omega
    free=radius0+db0*np.sin(time*omega)/omega; d_free=db0*np.cos(time*omega)
    def deriv(t,y):
        d,b,p,pb=y; overlap=max(0.,2*(radius0+b)-d)
        return [p,pb/mu,-G/d**2+k*overlap,-kappa*b-2*k*overlap]
    def integrate(step):
        return solve_ivp(deriv,[0,5.1],[2.,0.,0.,mu*db0],method='DOP853',rtol=1e-11,atol=2e-13,max_step=step,dense_output=True).sol
    sol=integrate(.007); q=sol(time);fine=integrate(.003)(time)
    d,b,p,pb=q; O=np.maximum(0.,2*(radius0+b)-d)
    energy=p*p/2+pb*pb/(2*mu)-G/d+kappa*b*b/2+k*O*O/2
    assert np.max(np.abs(q-fine))<1e-7 and np.ptp(energy)<1e-7 and free.min()>0 and O.max()>.2
    ds=(HERE.parent/'observatory/data.js').read_text()
    baseline=np.asarray(json.loads(ds[ds.index('{'):ds.rindex('}')+1])['baselineBeat'])
    bt=baseline[:51,0]; ref=sol(bt)
    err=float(abs(ref[:2].T-baseline[:51,1:3]).max())
    assert err<.001
    return time,free,d_free,d,radius0+b,pb/mu,O,dict(reference_peak_abs_position_error=err,max_step_refinement=float(np.max(np.abs(q-fine))),energy_span=float(np.ptp(energy)),overlap_range=O[[O.argmin(),O.argmax()]].tolist())

def shell(rho):
    # Integral through a translucent shell of finite thickness, softened at the limb.
    thickness=.165
    p=np.sqrt(np.maximum(0.,1.-rho*rho))
    m=np.sqrt(np.maximum(0.,(1-thickness)**2-rho*rho))
    column=(p-m)/np.sqrt(1-(1-thickness)**2)
    edge=np.exp(-.5*((rho-.89)/.104)**2)
    halo=np.exp(-.5*((rho-.94)/.168)**2)
    return .56*column+.34*edge+.10*halo
def lphoto(a):
    return Image.fromarray(np.rint(np.clip(a,0,1)*255).astype('uint8'))
TILE_SIZE=432
def atomic_tile(model,i,exposure):
    S=TILE_SIZE; centre=S/2; scale=48.7
    canvas=np.zeros((S,S),dtype=np.float32)
    xy=model.xy[i]; C=model.covariance[i]; vcam=model.velocity[i]; energy=model.energy[i]
    order=np.argsort(xy[:,2]); dmin,dmax=-4.10,4.10
    for j in order:
        X=centre+scale*xy[j,0];Y=centre-scale*xy[j,1]
        mat=C[j].copy();mat[0,1]*=-1;mat[1,0]*=-1
        F=mat*scale**2
        a,b=F[0,0],F[0,1];cc=F[1,1];dd=a*cc-b*b
        dx=np.sqrt(a)*1.43;dy=np.sqrt(cc)*1.43
        left=max(0,int(X-dx));right=min(S,int(X+dx+1));top=max(0,int(Y-dy));bottom=min(S,int(Y+dy+1))
        if right<=left or bottom<=top:continue
        yy,xx=np.ogrid[top:bottom,left:right]
        u,v=xx-X,yy-Y
        rho=np.sqrt(np.maximum(0.,(cc*u*u-2*b*u*v+a*v*v)/dd))
        m=shell(rho)**1.33
        depth=.10+.90*((xy[j,2]-dmin)/(dmax-dmin))**1.34
        exposure_local=energy[j]/exposure
        white=(-np.expm1(-1.48*exposure_local))**.93
        # Slightly lift the contours above black, preserving a fixed exposure for the whole film.
        vx,vy,vz=vcam[j]/max(np.linalg.norm(vcam[j]),1e-13)
        nx=np.clip(u/np.sqrt(a),-1.,1.);ny=-np.clip(v/np.sqrt(cc),-1.,1.)
        nz=np.sqrt(np.maximum(0.,1-rho**2))
        aligned=np.minimum(1.,np.square(nx*vx+ny*vy+nz*vz))
        radiance=(.99*white*(.38+.62*aligned))*depth
        # Occlude rear sites gently to retain the three-dimensional arrangement.
        opacity=np.minimum(.85,.27*np.maximum(0.,1-rho**2)**.5+.49*m)*(.72+.28*depth)
        area=canvas[top:bottom,left:right]
        area[:]=area*(1-opacity)+radiance*(1-np.exp(-2.35*m))
    return lphoto(1-np.exp(-1.86*canvas))

def crop_target(model):
    # A forward-facing neighbourhood with both bright and quiet sites.
    ref=model.reference
    candidates=np.flatnonzero((ref[:,0]>.7)&(ref[:,0]<2.7)&(ref[:,1]>-.4)&(ref[:,1]<1.5)&(ref[:,2]>1.9))
    k=candidates[np.mean(model.energy[:,candidates],axis=0).argmax()]
    c=np.array([TILE_SIZE/2+48.7*ref[k,0],TILE_SIZE/2-48.7*ref[k,1]])
    a=118; top=np.rint(c-a/2).astype(int)
    return int(top[0]),int(top[1]),int(top[0]+a),int(top[1]+a)
SZ=(840,540)
WHITE=(245,244,239); GREY=(169,177,182); INK=(0,0,0); RULE=(59,67,73)
GOLD=(255,179,72); CYAN=(90,217,218); VIOLET=(199,155,244)
MANTLE_ATTRACTION=.4
MANTLE_CONTACT_STIFFNESS=3.
FOLDER=str(HERE/'fonts')+'/'
@lru_cache(None)
def font(which,size):
    return ImageFont.truetype(FOLDER+{'regular':'NotoSansDisplay-Regular','bold':'NotoSansDisplay-Bold','title':'NotoSerifDisplay-Regular'}[which]+'.ttf',size)
def put(im,x,y,s,size=17,fill=WHITE,face='regular',anchor='lm'):
    ImageDraw.Draw(im).text((x,y),translate(s),font=font(face,size),fill=fill,anchor=anchor)
def rule(im,xy,col=RULE,width=1):ImageDraw.Draw(im).line(xy,fill=col,width=width)
def frame(title,kicker,sub):
    im=Image.new('RGB',SZ);put(im,30,34,title,32,face='title')
    put(im,809,35,kicker,15,GOLD,'bold','rm')
    put(im,31,73,sub,18,GREY);rule(im,(30,96,809,96));return im
def foot(im,a,b):
    rule(im,(30,451,810,451));put(im,30,476,a,17);put(im,30,507,b,15,GREY)
def fmt(x,n=2):return f'{x:.{n}f}'.replace('.',',')
def curve(im,box,values,i,colour):
    x,y,w,h=box;values=np.asarray(values);coord=[(x+j*w/(len(values)-1),y+h*(1-(v-values.min())/(np.ptp(values) or 1))) for j,v in enumerate(values)]
    ImageDraw.Draw(im).line(coord,fill=RULE,width=2);a,b=coord[i];ImageDraw.Draw(im).ellipse((a-4,b-4,a+4,b+4),fill=colour)
def optical_spot(lam,size=142,exposure=1.0):
    y,x=np.mgrid[:size,:size]-(size-1)/2;z=np.hypot(x,y)*.118*550/lam
    field=np.ones_like(z);nz=z>0;field[nz]=(2*j1(z[nz])/z[nz])**2
    field=(np.clip(field*exposure,0,1))**.50
    tint=np.asarray(spectral_colour(lam), dtype=float)
    return Image.fromarray(np.rint(field[:,:,None]*tint).astype('uint8'))
def palette():
    vals=[(g,g,g) for g in np.linspace(0,255,81).round().astype(int)]
    for col in [GOLD,CYAN,VIOLET,tuple(spectral_colour(557.02894)),tuple(spectral_colour(587.09160)),tuple(spectral_colour(473.7)),tuple(spectral_colour(516.5)),tuple(spectral_colour(563.5))]:
        vals.extend(tuple(int(c*f) for c in col) for f in np.linspace(.08,1,18))
    vals.extend([WHITE,GREY,RULE,(242,223,180),(100,107,112),(28,32,38),(37,46,54),(120,132,142),(231,237,239)])
    vals.extend([(0,0,0)]*(256-len(vals)))
    p=Image.new('P',(1,1));p.putpalette([int(c) for v in vals for c in v]);return p
PAL=palette()
def quant(im):return im.quantize(palette=PAL,dither=Image.Dither.NONE)
def save(ims,name,comment,msec):
    ims[0].save(OUT/name,save_all=True,append_images=ims[1:],duration=msec,loop=0,disposal=1,optimize=False,comment=comment.encode('ascii'))
def field(radii,centres,rd,vc,w,h,scale,overlap=0.,history=None,peak_overlap=1.):
    yy,xx=np.mgrid[:h,:w];xx=(xx-w/2)/scale;yy=(yy-h/2)/scale
    white=np.zeros((h,w)); amber=np.zeros_like(white)
    for r,c,u in zip(radii,centres,vc):
        dist=np.hypot(xx-c,yy)
        nx=(xx-c)/np.maximum(dist,1e-9)
        speed=abs(rd+u*nx)
        bright=np.minimum(1,speed/np.sqrt(2))**.82
        add=(1-np.exp(-3.1*shell(dist/r)))*bright
        white=1-(1-white)*(1-add)
    if history:
        for age,(clist,old_radius,old_speed) in enumerate(history):
            fade=(1-age/(len(history)+1))
            for old_c,new_c in zip(clist,centres):
                displacement=abs(new_c-old_c)
                wake=min(1.,displacement/.055)*min(1.,abs(old_speed)/.32)
                rho=np.hypot(xx-old_c,yy)/old_radius
                trail=.26*fade*wake*np.exp(-.5*((rho-.945)/.075)**2)
                point=.17*fade*np.exp(-((xx-old_c)**2+yy**2)/.004)
                white=np.maximum(np.maximum(white,trail),point)
    if overlap>0:
        zs=[r*r-(xx-c)**2-yy**2 for r,c in zip(radii,centres)]
        depth=np.sqrt(np.maximum(0.,np.minimum(zs[0],zs[1])))
        amber=(1-np.exp(-7.3*depth))*(min(1.,(overlap/peak_overlap)**2)**.52)
    base=white[:,:,None]*np.array([1.,1.,1.]);colour=np.array(GOLD)/255
    # Amber ink makes stored energy visible even at a radial turning point.
    rgb=base*(1-amber[:,:,None]) + amber[:,:,None]*colour
    return Image.fromarray((np.clip(rgb,0,1)*255).round().astype('uint8'))
def mantle_data():
    mu=.3;kap=2.;k=MANTLE_CONTACT_STIFFNESS;G=MANTLE_ATTRACTION;rad=.8;w=np.sqrt(kap/mu);v0=np.sqrt(2*.30/mu)
    def force(t,z):
        d,b,p,pb=z; O=max(0,2*(rad+b)-d)
        return (p,pb/mu,-G/d**2+k*O,-kap*b-2*k*O)
    def turning(t,z):return z[3]
    turning.direction=-1
    sol=solve_ivp(force,(0,4*np.pi/w),[2.,0.,0.,mu*v0],rtol=2e-12,atol=1e-14,max_step=.004,method='DOP853',dense_output=True,events=turning)
    tt=sol.t_events[0][0]
    base=np.arange(96)/96*4*np.pi/w
    times=np.unique(np.r_[base,tt]);d,b,p,pb=sol.sol(times)
    R=rad+b;O=np.maximum(0,2*R-d);E=.5*k*O**2;P=k*O*p
    freerad=rad+v0/w*np.sin(times*w);freev=v0*np.cos(times*w);freev[abs(freev)<1e-12]=0
    coupledv=pb/mu; idx=np.argmin(abs(times-tt));coupledv[idx]=0
    # Independent regression check against the archived console trajectory.
    ref=mantle_paths();np.testing.assert_allclose(sol.sol(ref[0])[0],ref[3],atol=1e-7)
    val={'t':tt,'overlap':O[idx],'energy':E[idx],'outgoing_power':P[idx],'separation_speed':p[idx],'radial_speed':coupledv[idx],'free_extinctions':list(base[np.isclose(np.cos(base*w),0,atol=1e-12)])}
    return times,d,R,p,coupledv,O,E,P,freerad,freev,idx,val
def wave_images():
    times,d,r,dc,dr,O,E,P,rf,vf,idx,report=mantle_data();imlist=[]
    net_force=MANTLE_CONTACT_STIFFNESS*O-MANTLE_ATTRACTION/d**2
    for i,t in enumerate(times):
        im=frame('Le battement et le contact','BST   /   ONDES','Le blanc suit le mouvement local du manteau. L’ambre suit l’énergie du contact.')
        put(im,40,128,'Une onde libre',21,WHITE,'bold');put(im,361,128,'Deux ondes en interaction',21,WHITE,'bold')
        centre=d[i]/2;hist=[((-d[j]/2,d[j]/2),r[j],dc[j]) for j in range(i-1,max(-1,i-10),-1)]
        im.paste(field([rf[i]],[0],vf[i],[0],284,205,68),(39,153))
        im.paste(field([r[i],r[i]],[-centre,centre],dr[i],[-dc[i]/2,dc[i]/2],438,205,72,O[i],hist,peak_overlap=O.max()),(351,153))
        state='Retournement · halo éteint' if vf[i]==0 else ('Dilatation' if vf[i]>0 else 'Contraction')
        put(im,40,368,state,18,WHITE);put(im,362,368,'Énergie au contact   '+fmt(E[i],3),18,GOLD)
        put(im,40,398,'Rayon   '+fmt(rf[i]),16,GREY)
        put(im,362,398,'Puissance vers la séparation   '+fmt(P[i],3),16,CYAN)
        curve(im,(40,417,248,17),rf,i,WHITE);curve(im,(363,417,391,17),E,i,GOLD)
        force_label='Attraction dominante' if net_force[i]<0 else 'Répulsion dominante' if net_force[i]>0 else 'Forces équilibrées'
        signed_force=f'{net_force[i]:+.3f}'.replace('.',',')
        foot(im,force_label+'   ·   Force nette sur la séparation   '+signed_force,
             't = '+fmt(t,3)+' (unités du modèle)   ·   '+('Retournement du battement en contact' if i==idx else 'Le sillage gris conserve jusqu’à neuf images du trajet.'))
        if i in [idx,12,58]:im.save(REV/f'wave-{i}.png')
        imlist.append(quant(im))
    durations=[70]*len(times);durations[idx]=600
    for i in np.nonzero(vf==0)[0]:durations[i]=290
    save(imlist[5:],'beat-contact.gif','Reduced BST mantle model. Net separation force is -G/d^2 + k*overlap. Motion is white; amber shows stored contact energy. Time starts at 0.253 in model units.',durations[5:])
    return report
def atom_images():
    a=krypton();b=krypton(.39);box=crop_target(a)
    e=float(np.quantile(np.concatenate((a.energy,b.energy)),.925));atoms=[]
    spots=[optical_spot(w) for w in [557.02894,587.09160]]
    for i in range(FRAMES):
        l=atomic_tile(a,i,e)
        im=frame('Un atome de krypton','BST   /   Kr','La géométrie calculée à gauche et deux vues filtrées d’un émetteur excité à droite.')
        im.paste(l.resize((331,331),Image.Resampling.LANCZOS),(16,105))
        x0,y0,x1,y1=box;S=331/432
        for xx,yy,sx,sy in [(x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)]:
            tx,ty=16+S*xx,105+S*yy;rule(im,(tx,ty,tx+6*sx,ty),GREY);rule(im,(tx,ty,tx,ty+6*sy),GREY)
        im.paste(l.crop(box).resize((140,140),Image.Resampling.LANCZOS),(347,187))
        put(im,416,350,'Détail × 1,6',14,GREY,anchor='mm')
        rule(im,(501,120,501,430))
        put(im,527,143,'Krypton neutre excité',21,WHITE,'bold')
        for j,lam in enumerate([557.02894,587.09160]):
            x=512+145*j;im.paste(spots[j],(x,181));put(im,x+71,342,fmt(lam,2)+' nm',17,tuple(int(v) for v in spectral_colour(lam)),face='bold',anchor='mm')
        put(im,656,386,'Deux raies de Kr I (NIST)',16,GREY,anchor='mm');put(im,656,411,'Même ouverture optique',15,GREY,anchor='mm')
        foot(im,'Blanc : mouvement des 309 sites du calcul, huit modes. Déplacements agrandis × 12.',
             'Couleurs : émetteur non résolu dans deux filtres. Profils optiques illustratifs, éclats normalisés.')
        atoms.append(quant(im))
    name='atomic-base.gif'
    save(atoms,name,'Atomic motion before the spectral panel is drawn.',70)
    atomic_uv(OUT/name)
    (OUT/name).unlink()
    (OUT/'BST_Atome_de_krypton.gif').replace(OUT/'atomic-krypton.gif')

if __name__=='__main__':
    wave_images()
    atom_images()
