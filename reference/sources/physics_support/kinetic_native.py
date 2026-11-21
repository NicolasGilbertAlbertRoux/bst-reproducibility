'Exact functions from R235, with only their standalone imports.'
import itertools,numpy as np

def jets(n,h,tangent=None):
    for bits in itertools.product([0,1],repeat=3):
        axes=tuple(k for k,b in enumerate(bits) if b)
        f=np.flip(n,axis=axes) if axes else n
        base=f[:-1,:-1,:-1];ds=[]
        for k,nb in enumerate([f[1:,:-1,:-1],f[:-1,1:,:-1],f[:-1,:-1,1:]]):
            scalar=np.sum(base*nb,axis=-1);assert scalar.min()>0
            ds.append(((-1)**bits[k])*(nb/scalar[...,None]-base)/h)
        d=np.stack(ds,axis=-2).reshape(-1,3,4)
        if tangent is not None:
            t=np.flip(tangent,axis=tuple(a+1 for a in axes)) if axes else tangent
            t=t[:,:-1,:-1,:-1].reshape(len(t),-1,4).transpose(1,0,2)
        else:t=None
        yield d,t,h**3/8

def kinetic_blocks(n,h,t,A):
    k=len(t);mass=np.zeros((k+3,k+3));M2=np.zeros_like(mass);ai=np.linalg.inv(A)
    for d,v,w in jets(n,h,t):
        dp=np.einsum('ij,pjc->pic',ai.T,d);modes=np.concatenate([v,-dp],axis=1)
        g=dp@dp.swapaxes(1,2);tr=np.trace(g,axis1=1,axis2=2)
        weight=np.linalg.det(A)*w
        base=np.einsum('pac,pbc->ab',modes,modes)
        # Positive quartic kinetic form pointwise; same static invariants.
        weighted=np.einsum('pac,pbc,p->ab',modes,modes,tr)
        pd=np.einsum('pac,pic->pai',modes,dp)
        correction=weighted-np.einsum('pai,pbi->ab',pd,pd)
        M2+=weight*base;mass+=weight*(base+correction)
    return (mass+mass.T)/2,(M2+M2.T)/2
