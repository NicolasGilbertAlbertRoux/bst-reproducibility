'Positive nonlinear substrate weights in the native kinetic action.'

import ctypes

from flux_nodal import replace_once
from kernel_topological import source_code as parent_code
from refit_dependencies import ARRAY,ROOT,load_c
import numpy as np


SOURCE = ROOT/'scripts'/'cinetique_courbes_R352.c'
RESPONSE = r'''
#include <math.h>

static double nonlinear_response(double rho,int count,const double *terms,double *derivative) {
    if (derivative) *derivative=0.;
    double x=rho*rho, value=1.;
    if (!(x>0.)) return value;
    for (int j=0;j<count;j++) {
        double center=terms[3*j], width=terms[3*j+1], amplitude=terms[3*j+2];
        double t=(log(x)-log(center))/width;
        double gain=amplitude*exp(-.5*t*t);
        value+=gain;
        if (derivative && gain>0.) *derivative-=2.*gain*t/(width*rho);
    }
    return value;
}

'''


def source_code():
    src = replace_once(parent_code(),'int bst_topological_q(',RESPONSE+'int bst_curve_q(')
    src = replace_once(src,'int bst_topological_mass(','int bst_curve_mass(')
    marker = 'double alpha,double beta,double lambda,const double*field'
    if src.count(marker)!=2:
        raise ValueError('The two received signatures do not match the extension')
    src = src.replace(marker,'double alpha,double beta,int windows,const double*terms,const double*field')
    marker = 'double rho=dot4(base,cofactor), weighted=weight*(1.+lambda*rho*rho);'
    if src.count(marker)!=2:
        raise ValueError('Unexpected number of topology-weighted stencils')
    src = src.replace(marker,'''double rho=dot4(base,cofactor), derivative;
   double weighted=weight*nonlinear_response(rho,windows,terms,&derivative);''',1)
    src = replace_once(src,marker,'double rho=dot4(base,cofactor), weighted=weight*nonlinear_response(rho,windows,terms,NULL);')
    return replace_once(src,'double topological_derivative=2*weight*lambda*rho*q;',
                        'double topological_derivative=weight*derivative*q;')


class CurveKinetic:
    def __init__(self,beta,terms):
        self.beta = beta
        if any(r['chi_power']!=0 or r['center']<=0 or r['width_log']<=0 or r['amplitude']<0 for r in terms):
            raise ValueError('Expected nonnegative pure-density windows')
        self.windows = np.ascontiguousarray([[r['center'],r['width_log'],r['amplitude']] for r in terms],dtype=float)
        text = source_code()
        if not SOURCE.exists():
            SOURCE.write_text(text)
        elif SOURCE.read_text()!=text:
            raise ValueError('The nonlinear C source differs from its qualified transformation')
        args = [ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ARRAY]
        self.qfun,self.hash = load_c(SOURCE,'bst_curve_q',[*args,ARRAY,ARRAY,ARRAY,ARRAY,ARRAY,ARRAY,ctypes.POINTER(ctypes.c_double)])
        self.mfun,_ = load_c(SOURCE,'bst_curve_mass',[*args,ARRAY,ARRAY,ARRAY,ARRAY])

    def quadratic(self,n,h,t,v):
        gn,gt,gv = np.zeros_like(n),np.zeros_like(n),np.zeros(9)
        Q = ctypes.c_double()
        status = self.qfun(round(len(n)**(1/3)),h,1.,self.beta,len(self.windows),self.windows,
                           np.ascontiguousarray(n),np.ascontiguousarray(t),np.ascontiguousarray(v),
                           gn,gt,gv,ctypes.byref(Q))
        if status:
            raise ValueError('Field outside the nonlinear kinetic chart')
        return Q.value,gn,gt,gv

    def mass(self,n,h):
        A,C,G = np.zeros((len(n),4,4)),np.zeros((len(n),4,9)),np.zeros((9,9))
        if self.mfun(round(len(n)**(1/3)),h,1.,self.beta,len(self.windows),self.windows,np.ascontiguousarray(n),A,C,G):
            raise ValueError('Mass outside the nonlinear kinetic chart')
        return A,C,G
