'Independent verification of the correction by the infinite vacuum Kirchhoff solution.'
import ctypes
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from scipy.special import roots_legendre
from defect import A,H0,O

def gauss(n,a,b):
    x,w=roots_legendre(n);return a+(x+1)*(b-a)*.5,w*(b-a)*.5
def once(fun,order,eps):
    u,w=gauss(order,-1,1)
    phi=np.arange(2*order)*np.pi/order
    nx=np.broadcast_to(u[:,None],(order,len(phi)));t=np.sqrt(1-nx**2)
    xyz=np.ascontiguousarray(np.stack([nx,t*np.cos(phi),t*np.sin(phi)],axis=-1).reshape(-1,3))
    ws=np.repeat(w/(2*len(phi)),len(phi))
    # The cone of the signal stops at 5-3.6/h0. 6. The integration on the disk is identical to that of the half-sided square 7 .
    radial,rw=gauss(2*order,0,6);tt,tw=gauss(order,0,.36)
    def E(plane,probe):
        return fun(plane,probe,H0,eps,len(ws),xyz,ws,len(rw),radial,rw,len(tw),tt,tw)
    return dict(order=order,epsilon_t=eps,sonde_native=[E(x,.22) for x in (23.5,24.5)],derivative_local=[E(x,eps) for x in (23.5,24.5)])
def main():
    with TemporaryDirectory() as tmp:
        dest=Path(tmp)/'kirchhoff.so';src=Path(__file__).with_suffix('.cpp')
        opt=['c++','-O3','-std=c++17','-shared','-fPIC']
        if subprocess.run([*opt,'-fopenmp',str(src),'-o',str(dest)],capture_output=True).returncode:subprocess.run([*opt,str(src),'-o',str(dest)],check=True)
        fun=ctypes.CDLL(str(dest)).energy;fun.argtypes=[ctypes.c_double]*4+[ctypes.c_int,A,A]*3;fun.restype=ctypes.c_double
        results=[once(fun,n,eps) for n,eps in [(12,.0002),(20,.0001),(32,.0001),(40,.00005)]]
        print(results,flush=True)
        (O/'KIRCHHOFF_R573.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
