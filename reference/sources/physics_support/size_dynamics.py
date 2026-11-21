'Native support radius and the exact nodal covector away from sign changes.'

import ctypes
from pathlib import Path

from refit_dependencies import ARRAY,load_c
import numpy as np


SOURCE = Path(__file__).with_name('derivee_taille_R357.c')


class Radius:
    def __init__(self):
        self.function,self.sha256 = load_c(SOURCE,'bst_r357_radius',
            [ctypes.c_int,ctypes.c_double,ctypes.c_int,ARRAY,ARRAY,ARRAY,ARRAY])

    def __call__(self,n,h,rate=None):
        count = round(len(n)**(1/3))
        if n.shape!=(count**3,4):
            raise ValueError('The radius requires a cube of S3 nodes')
        n = np.ascontiguousarray(n,dtype=float)
        if rate is None:
            t,g = np.empty(0),np.empty(0)
        else:
            t = np.ascontiguousarray(rate,dtype=float)
            g = np.zeros_like(n)
            if t.shape!=n.shape:
                raise ValueError('The radius covector requires a nodal direction')
        out = np.zeros(3)
        status = self.function(count,h,int(rate is not None),n,t,g,out)
        if status:
            raise ArithmeticError(f'Native support radius left its qualified chart ({status})')
        return (*out,g)
