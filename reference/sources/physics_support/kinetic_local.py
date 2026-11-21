"R357's gated support radius and kinetic density, with both nodal adjoints."

import ctypes
from pathlib import Path

from flux_nodal import replace_once
from refit_dependencies import ARRAY,ROOT,load_c
from size_dynamics import Radius,SOURCE as RADIUS_SOURCE
import numpy as np

PARENT = ROOT/'sources'/'cinetique_complete_R343.c'
SOURCE_K = Path(__file__).with_name('cinetique_locale_R358.c')
SOURCE_R = Path(__file__).with_name('rayon_local_R358.c')

GATE = """static double gate(double x, double lo, double hi, double *dg) {
    *dg = 0.;
    if (x <= lo) return 1.;
    if (x >= hi) return 0.;
    const double t = (hi-x)/(hi-lo);
    *dg = -30.*t*t*(1.-t)*(1.-t)/(hi-lo);
    return t*t*t*(10.+t*(-15.+6.*t));
}
"""


def generate_kinetic():
    src = PARENT.read_text()
    src = replace_once(src,'int bst_general_q(',GATE+'\nint bst_local_q(')
    src = replace_once(src,'int bst_general_mass(','int bst_local_mass(')
    sig = 'double alpha,double beta,const double*field'
    if src.count(sig)!=2:
        raise ArithmeticError('Unexpected parent kinetic signatures')
    src = src.replace(sig,'double alpha,double beta,double lo,double hi,double a,const double*field')
    src = replace_once(src,'const double weight=h*h*h/8.;double total=0.;',
                       'const double volume=h*h*h/8.;double total=0.,Qg=0.;')
    src = replace_once(src,'const double weight=h*h*h/8.;','const double volume=h*h*h/8.;')
    anchor = '  for(int bits=0;bits<8;bits++){'
    if src.count(anchor)!=2:
        raise ArithmeticError('Unexpected parent quadrature')
    src = src.replace(anchor,'''  double dg;
  const double g=gate(base[0],lo,hi,&dg);
  const double weight=volume*(1.+a*g);
  if (!(weight>0.)) return 2;
'''+anchor)
    src = replace_once(src,'total+=weight*q;',
        'total+=weight*q;Qg+=volume*g*q;gn[pos]+=volume*a*dg*q;')
    return replace_once(src,'*out=total;','out[0]=total;out[1]=Qg;')


def generate_radius():
    src = replace_once(RADIUS_SOURCE.read_text(),'int bst_r357_radius(',
                       GATE+'\nint bst_r358_radius(')
    src = replace_once(src,'double h, int differentiate','double h, double lo, double hi, int differentiate')
    src = replace_once(src,'                if (pass==0) {\n                    double w=fabs(rho);',
        '''                double dg;
                const double g=gate(v[0][0],lo,hi,&dg);
                if (pass==0) {
                    double w=g*fabs(rho);''')
    src = replace_once(src,'const double scale = sign*lever/(2.*radius*moment[0]);',
        '''const double normal = lever/(2.*radius*moment[0]);
                const double scale = g*sign*normal;
                gradient[at[0]] += dg*fabs(rho)*normal;''')
    return replace_once(src,'cusp+=fabs(lever*directional);','cusp+=g*fabs(lever*directional);')


def qualified_source(path,content):
    if not path.exists():
        path.write_text(content)
    elif path.read_text()!=content:
        raise ArithmeticError(f'Unexpected generated source {path.name}')
    return path


class LocalRadius(Radius):
    def __init__(self,lo,hi):
        source = qualified_source(SOURCE_R,generate_radius())
        fun,self.sha256 = load_c(source,'bst_r358_radius',
            [ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ARRAY,ARRAY,ARRAY,ARRAY])
        self.function = lambda n,h,*args:fun(n,h,lo,hi,*args)


class LocalKinetic:
    def __init__(self,beta,lo,hi):
        if not lo<hi<=1.:
            raise ValueError('Expected a regular gate with zero loading at the vacuum')
        self.constants = (1.,beta,lo,hi)
        source = qualified_source(SOURCE_K,generate_kinetic())
        args = [ctypes.c_int]+[ctypes.c_double]*6
        self.qfun,self.sha = load_c(source,'bst_local_q',args+[ARRAY]*7)
        self.mfun,_ = load_c(source,'bst_local_mass',args+[ARRAY]*4)

    def quadratic(self,n,h,t,v,a):
        gn,gt,gv = np.zeros_like(n),np.zeros_like(n),np.zeros(9)
        out = np.zeros(2)
        N = round(len(n)**(1/3))
        if self.qfun(N,h,*self.constants,a,np.ascontiguousarray(n),np.ascontiguousarray(t),
                     np.ascontiguousarray(v),gn,gt,gv,out):
            raise ArithmeticError('Invalid field or kinetic weight')
        return out[0],gn,gt,gv,out[1]

    def mass(self,n,h,a):
        A = np.zeros((len(n),4,4))
        C,G = np.zeros((len(n),4,9)),np.zeros((9,9))
        N = round(len(n)**(1/3))
        if self.mfun(N,h,*self.constants,a,np.ascontiguousarray(n),A,C,G):
            raise ArithmeticError('Invalid field or mass weight')
        return A,C,G
