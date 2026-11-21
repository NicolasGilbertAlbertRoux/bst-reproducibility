'Correct the packet resolution error using an auxiliary problem of the same discretisation.'
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.integrate import simpson
from grid import stencil, ROOT

I=ROOT/'inputs'
O=ROOT/'results'
H0=json.loads((I/'MOUVEMENT_N56_impedance_dt0.09.json').read_text())['h_F']
A=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
def kernel(tmp):
    lib=Path(tmp)/'vide.so'
    opts=['c++','-O3','-std=c++17','-shared','-fPIC']
    src=Path(__file__).with_suffix('.cpp')
    result=subprocess.run([*opts,'-fopenmp',str(src),'-o',str(lib)],capture_output=True)
    if result.returncode:subprocess.run([*opts,str(src),'-o',str(lib)],check=True)
    fun=ctypes.CDLL(str(lib)).acceleration
    fun.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[A]*4;fun.restype=None
    return fun
def run(fun,n,dtmax,width):
    scale=55/(n-1)
    g=np.arange(n)*scale-27.5
    xs=g[g>=12];zs=g[abs(g)<=width];shape=(len(xs),len(zs),len(zs))
    y,z=np.meshgrid(zs,zs,indexing='ij');xi=(xs[:,None,None]-22.5)/1.5
    ex=np.maximum(1-xi**2,0);rad=np.maximum(1-(y*y+z*z)/25,0)**3
    theta=np.arctan(.025*ex**3*rad);v=.025*6*xi*ex**2*rad*np.cos(theta)/(1.5*H0)
    nsteps=round(.36/dtmax);dt=.36/nsteps;h=scale*H0
    matrices={}
    # Same probes, same transverse abscissa and same centered difference as in the field of seven sites.
    zq=np.linspace(-7.,7.,65)
    for plane in [23.5,24.5]:
        for offset in [0.,.22,-.22]:
            matrices[plane,offset]=[stencil(len(a),(t-a[0])/scale) for a,t in [(xs,np.array([plane+offset])),(zs,zq),(zs,zq)]]
    def sample(f,plane,offset):
        out=f
        for d,m in enumerate(matrices[plane,offset]):
            b=np.moveaxis(out,d,0);o=m@b.reshape(b.shape[0],-1)
            out=np.moveaxis(o.reshape((o.shape[0],)+b.shape[1:]),0,d)
        return out[0]
    w=np.ones((65,65));w[[0,-1],:]*=.5;w[:,[0,-1]]*=.5
    def flux(angle,speed):
        c=np.cos(angle);s=np.sin(angle)
        f=np.stack([c,s,-s*speed,c*speed],axis=-1)
        p=[]
        for plane in [23.5,24.5]:
            mid=sample(f,plane,0);length=np.hypot(mid[...,0],mid[...,1])
            tangent=np.stack([-mid[...,1],mid[...,0]],axis=-1)/length[...,None]
            tv=np.sum(tangent*mid[...,2:],axis=-1)/length
            grad=(sample(f[...,:2],plane,.22)-sample(f[...,:2],plane,-.22))/(.44*H0)
            dx=np.sum(tangent*grad,axis=-1)/length
            p.append(float(np.sum(-tv*dx*w)*(H0*(zq[1]-zq[0]))**2))
        return p
    work=np.empty(shape+(3,))
    def acc(q,p):
        a=np.empty_like(q);fun(*shape,h,q,p,a,work);return a
    rows=[]
    for k in range(nsteps+1):
        rows.append(flux(theta,v))
        if k==nsteps:break
        b1=acc(theta,v);a2=v+.5*dt*b1;b2=acc(theta+.5*dt*v,a2)
        a3=v+.5*dt*b2;b3=acc(theta+.5*dt*a2,a3)
        a4=v+dt*b3;b4=acc(theta+dt*a3,a4)
        theta+=(dt/6)*(v+2*a2+2*a3+a4);v+=(dt/6)*(b1+2*b2+2*b3+b4)
    energy=simpson(rows,x=np.linspace(0,.36,nsteps+1),axis=0)
    return dict(N=n,step_over_h0=scale,step_time=dt,half_width=width,dimensions=shape,flux=rows,energy_field=energy.tolist())
if __name__=='__main__':
    with tempfile.TemporaryDirectory() as d:
        f=kernel(d)
        for arg in sys.argv[1:]:
            n,s,ww=arg.split(',');ans=run(f,int(n),float(s),float(ww))
            (O/f'VIDE_N{n}_dt{s}_W{ww}.json').write_text(json.dumps(ans,indent=2,ensure_ascii=False)+'\n')
            print(n,ans['energy'],flush=True)
