'History signed with the tetrahedral interpolation of R383.'
from itertools import permutations, product

import numpy as np

OFFSETS=tuple(product((0,1),repeat=3))
EXTRA=np.array([-1.,0.,0.,0.])

def preimages(q, target, *, length=12.):
    'Use the same normalized affine interpolation as geometric_degree.\n\nA cell can be skipped if its projected vertex coordinates cannot span\nzero in a direction orthogonal to the requested target.'
    y=target/np.linalg.norm(target)
    _,_,frame=np.linalg.svd(y[None,:],full_matrices=True)
    a=q@frame[1:].T
    N=len(q)
    end=N-1
    low=a[:end,:end,:end].copy()
    high=low.copy()
    for i,j,k in OFFSETS[1:]:
        sub=a[i:i+end,j:j+end,k:k+end]
        np.minimum(low,sub,out=low)
        np.maximum(high,sub,out=high)
    tol=1e-12
    cells=np.argwhere(np.all((low<=tol)&(high>=-tol),axis=-1))
    found=[]
    for order in permutations(range(3)):
        e=np.eye(3,dtype=int)[list(order)]
        vertices=cells[:,None,:]+np.vstack((np.zeros(3,dtype=int),np.cumsum(e,axis=0)))
        matrix=np.moveaxis(q[vertices[...,0],vertices[...,1],vertices[...,2]],1,2)
        determinant=np.linalg.det(matrix)
        idx=np.flatnonzero(abs(determinant)>1e-12)
        coef=np.linalg.solve(matrix[idx],np.broadcast_to(y,(len(idx),4))[...,None])[...,0]
        choose=np.all(coef>1e-10,axis=1)
        orientation=round(np.linalg.det(e))
        for j,c in zip(idx[choose],coef[choose],strict=True):
            bary=c/c.sum()
            p=bary@vertices[j]
            reconstructed=matrix[j]@bary
            error=float(np.linalg.norm(reconstructed/np.linalg.norm(reconstructed)-y))
            assert error<1e-10
            found.append(dict(sign=int(orientation*np.sign(determinant[j])),
                xyz_native=(p*(length/(N-1))-length/2).tolist(),
                barycentric_min=float(min(bary)),target_error=error))
    found.sort(key=lambda v:(-v["sign"],v["xyz_native"]))
    return dict(target=y.tolist(),preimages=found,degree=sum(x["sign"] for x in found),
                positive=sum(x["sign"]==1 for x in found),negative=sum(x["sign"]==-1 for x in found))
