'Deterministic rendering of the native scenes delivered; no model of invented atom.\n\nAtomic movement: archived modes, unit mass and internal time.\nMolecules: site current covectors, fixed nuclear laying.\nBroadcast is a candidate graphical reading, never an optical emission.'
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def police(size, path=None):
    candidates=[path,'/System/Library/Fonts/Supplemental/Arial.ttf',
                '/Library/Fonts/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for p in candidates:
        if p and Path(p).is_file():
            return ImageFont.truetype(str(p),size)
    return ImageFont.load_default(size=size)

def projection():
    az,el=.7,.35
    x=np.array([np.cos(az),np.sin(az),0.])
    y=np.array([-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)])
    return np.stack([x,y]),np.cross(x,y)

def sprite(radius):
    n=int(2*radius)+3;c=(n-1)/2;y,x=np.mgrid[:n,:n]
    return np.clip(1-np.sqrt((x-c)**2+(y-c)**2)/radius,0,1)**1.8

def store(canvas,x,y,ball,gain):
    n=len(ball);r=n//2;x0=int(round(x))-r;y0=int(round(y))-r
    H,W=canvas.shape;xa,xb=max(0,x0),min(W,x0+n);ya,yb=max(0,y0),min(H,y0+n)
    if xa<xb and ya<yb:
        z=ball[ya-y0:yb-y0,xa-x0:xb-x0]*gain;reg=canvas[ya:yb,xa:xb]
        reg[:]=reg*(1-z)+z

def arrow(d,a,v,width=2):
    b=a+v;d.line([tuple(a),tuple(b)],fill='#c1cbd0',width=width)
    n=np.linalg.norm(v)
    if n>3:
        u=v/n;p=np.array([-u[1],u[0]]);h=min(7,n*.3)
        d.polygon([tuple(b),tuple(b-h*u+.4*h*p),tuple(b-h*u-.4*h*p)],fill='#c1cbd0')

def charger(entry,root):
    with np.load(root/entry['file'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
    pos=np.asarray(d['positions'],dtype=float);assert pos.ndim==2 and pos.shape[1]==3
    assert np.isfinite(pos).all() and len(pos)>1
    d['positions']=pos-pos.mean(0);r=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
    np.fill_diagonal(r,np.inf);d['spacing']=float(np.median(r.min(1)))
    assert d['spacing']>0
    if entry['kind']=='native_nodes':
        modes=d['modes'];w=d['omega_native']
        assert modes.shape==(len(w),len(pos),3) and np.all(w>0)
        assert np.isfinite(modes).all() and np.isfinite(w).all()
        norm=np.linalg.norm(modes.reshape(len(w),-1),axis=1)
        assert np.all(norm>0)
        scores=abs(np.einsum('mnc,nc->m',modes,d['positions']))/norm/np.linalg.norm(d['positions'])
        d['selected']=int(scores.argmax());d['overlap']=float(scores.max())
    else:
        assert d['current_modes'].shape==(len(d['frequency_ratios']),len(pos),3)
        assert np.all(d['frequency_ratios']>0) and np.isfinite(d['current_modes']).all()
        assert np.isclose(d['frequency_ratios'].min(),1.),'frequency-ratios shall be normalised by the slow frequency'
    return d

def graph(entry,d,root):
    with np.load(root/entry['graph_file'],allow_pickle=False) as z:
        target=z['lam'];degree=float(z['common_degree_reference'])
    pos=d['positions'];r=np.linalg.norm(pos[:,None]-pos[None,:],axis=-1)
    J=(r*r+4)**(-1.25)*np.mean([1/(1+(.35*x*r**.9)**2) for x in np.linspace(0,3,6)],axis=0)
    np.fill_diagonal(J,0);D=(np.diag(J.sum(1))-J)/degree
    lam,U=np.linalg.eigh(D);zero=abs(lam)<1e-10*max(lam)
    assert zero.sum()==1
    defect=float(np.max(abs(lam[~zero]/target-1)));assert defect<1e-9
    lam[zero]=0;return lam,U,defect

def sequence(entry,d,variant,opts,root):
    N=opts['frames'];res=opts['size'];pos=d['positions'];P,depth=projection()
    native=entry['kind']=='native_nodes';factor=opts['magnification'] if variant=='superposition' else opts['radial_magnification']
    scale_font=res/500;main=police(max(11,round(15*scale_font)),opts.get('font'));small=police(max(9,round(11*scale_font)),opts.get('font'))
    palette=Image.new('P',(1,1));palette.putpalette([v for i in range(128) for v in [round(255*i/127)]*3]+[0]*384)
    graphdef=None
    if native:
        modes=d['modes'];w=d['omega_native'];j=d['selected'];ws=w[j];period=2*np.pi/ws
        span=1. if variant!='superposition' else min(1.,N*ws/(12*max(w)))
        times=(np.arange(N)/N*span+.125)*period
        normbound=.5*np.max(np.sum((np.abs(modes)*w[:,None,None]).sum(0)**2,axis=1))
        if variant!='superposition':normbound=float((.5*ws*ws*np.sum(modes[j]**2,axis=1)).max())
        motionbound=factor*(np.abs(modes).sum(0) if variant=='superposition' else abs(modes[j]))
        if variant=='diffusion':
            lam,U,graphdef=graph(entry,d,root);a=lam**entry['diffusion_exponent'];tau=entry['tau_native']
            k=.5*ws*ws*np.sum(modes[j]**2,axis=1);keig=U.T@k
            aa=np.ones_like(a);np.divide(-np.expm1(-a),a,out=aa,where=a>0)
            b=a+2j*ws*tau;bb=-np.expm1(-b)/b
        units='t/period of the selected mode';title={'motion':'White - local kinetic activity','diffusion':'Displayed, with memory','superposition':'Overlaying native modes'}[variant]
    else:
        ratios=d['frequency_ratios'];span=1. if variant=='current_mode_lent' else min(1.,N/(12*max(ratios)))
        times=np.arange(N)/N*span+(.125 if variant=='current_mode_lent' else 0.)
        if variant=='current_mode_lent':
            j=int(np.argmin(ratios));current_history=np.sin(2*np.pi*times[:,None,None]*ratios[j])*d['current_modes'][j][None,:,:]
        else:current_history=np.einsum('tm,mnc->tnc',np.sin(2*np.pi*times[:,None]*ratios),d['current_modes'])
        # One normalization for the entire displayed time window, never per frame.
        normbound=float(np.max(np.sum(current_history**2,axis=2)))
        motionbound=np.zeros_like(pos);factor=1;units='t/ slow internal period'
        title='Slow archived mode: site currents' if variant=='current_mode_lent' else 'White - standard of squared site currents'
    extent=float(np.max(abs(pos@P.T))+np.max(motionbound@abs(P.T))+1.3*d['spacing'])
    scale=(res*.72)/(2*extent);ball=sprite((.66 if native else .58)*d['spacing']*scale)
    frames=[];peak_white=0.;min_white=np.inf
    for it,t in enumerate(times):
        if native:
            if variant=='superposition':
                q=np.einsum('m,mnc->nc',np.cos(w*t),modes)
                v=np.einsum('m,mnc->nc',-w*np.sin(w*t),modes);white=.5*np.sum(v*v,axis=1)
            else:
                q=modes[j]*np.cos(ws*t);white=.5*ws*ws*np.sum(modes[j]**2,axis=1)*np.sin(ws*t)**2
                if variant=='diffusion':white=U@(keig*(.5*aa-.5*np.real(np.exp(2j*ws*t)*bb)))
            current=pos+factor*q;phase=t/period
        else:
            currents=current_history[it]
            white=np.sum(currents**2,axis=1);current=pos;phase=t
        assert white.min()>-1e-9*max(normbound,1e-30)
        white=np.clip(white/max(normbound,1e-300),0,1)
        peak_white=max(peak_white,float(white.max()));min_white=min(min_white,float(white.max()))
        uv=current@P.T;px=uv[:,0]*scale+res/2;py=-uv[:,1]*scale+res*.55;canvas=np.zeros((res,res))
        for k in np.argsort(current@depth):store(canvas,px[k],py[k],ball,float(white[k]))
        im=Image.fromarray((canvas*255).astype(np.uint8)).convert('RGB');dr=ImageDraw.Draw(im)
        text=[entry['species']+' · '+title,
              f"{len(pos)} points · déplacement visuel ×{factor:g}" if native else ('Noyaux fixes · un mode de courant' if variant=='current_mode_lent' else 'Fixed cores - superimposed internal currents'),
              f'{units} = {phase:.3f}']
        def fitted(line,size):
            f=police(size,opts.get('font'))
            while dr.textbbox((0,0),line,font=f)[2]>res-24*scale_font and size>5:
                size-=1;f=police(size,opts.get('font'))
            return f
        for k,line in enumerate(text):dr.text((12*scale_font,(12+24*k)*scale_font),line,fill='white',font=fitted(line,max(11,round(15*scale_font))))
        if not native:
            xy=np.c_[px,py]
            for k,a0 in enumerate(xy):
                if k:dr.line([tuple(xy[0]),tuple(a0)],fill='#515a60',width=2)
                arrow(dr,a0,(currents[k]@P.T)*np.array([1,-1])*d['spacing']*scale*.6/max(np.sqrt(normbound),1e-300))
                dr.text(tuple(a0+np.array([8,-18])),entry['site_labels'][k],fill='white',font=main)
        foot=['Standard mechanical reading - no assigned optical emission',
              'Isolated mode: periodic loop' if (native and variant!='superposition') or variant=='current_mode_lent' else 'Finished window: loop connection possibly discontinuous']
        for k,line in enumerate(foot):dr.text((12*scale_font,res-(42-18*k)*scale_font),line,fill='#abb6bb',font=fitted(line,max(9,round(11*scale_font))))
        frames.append(im.quantize(palette=palette,dither=Image.Dither.NONE))
    meta={'species':entry['species'],'variant':variant,'frames_sampled':N,'time_span_in_label_periods':span,
          'periodic_solution':(native and variant!='superposition') or variant=='current_mode_lent','nuclear_positions_animated':False if not native else None,
          'magnification':factor,'normalization_fixed_over_sequence':True,'peak_white_normalized':peak_white,
          'minimum_frame_max_white':min_white,'graph_eigenvalue_relative_defect':graphdef,'optical_emission':False}
    if native:meta.update(selected_mode=d['selected'],dilation_overlap=d['overlap'],period_internal=float(period))
    return frames,meta
