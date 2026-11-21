'The received kinetic action with one local, positive S3 density weight.'

import ctypes

from flux_nodal import replace_once
from refit_dependencies import ARRAY, ROOT, load_c
import numpy as np


PARENT = ROOT/'sources'/'cinetique_complete_R343.c'
SOURCE = ROOT/'scripts'/'cinetique_topologique_R351.c'

COFACTORS = r'''
static double minor3(const double *a, const double *b, const double *c, int i, int j, int k) {
    return a[i]*(b[j]*c[k]-b[k]*c[j])-a[j]*(b[i]*c[k]-b[k]*c[i])+a[k]*(b[i]*c[j]-b[j]*c[i]);
}

static void row_cofactors(const double *a, const double *b, const double *c, double *g) {
    g[0]=minor3(a,b,c,1,2,3);
    g[1]=-minor3(a,b,c,0,2,3);
    g[2]=minor3(a,b,c,0,1,3);
    g[3]=-minor3(a,b,c,0,1,2);
}

'''


def source_code():
    src = PARENT.read_text()
    src = replace_once(src,'int bst_general_q(',COFACTORS+'int bst_topological_q(')
    src = replace_once(src,'int bst_general_mass(','int bst_topological_mass(')
    if src.count('double alpha,double beta,const double*field')!=2:
        raise ValueError('The native kernel no longer has the two expected entry points')
    src = src.replace('double alpha,double beta,const double*field','double alpha,double beta,double lambda,const double*field')
    src = replace_once(src,'double v[4],r[3],pd[3],nu[4];','''double cofactor[4];
   row_cofactors(d[0],d[1],d[2],cofactor);
   double rho=dot4(base,cofactor), weighted=weight*(1.+lambda*rho*rho);
   double v[4],r[3],pd[3],nu[4];''')
    src = replace_once(src,'total+=weight*q;', '''total+=weighted*q;
   double topological_derivative=2*weight*lambda*rho*q;
   for(int j=0;j<4;j++)gn[pos+j]+=topological_derivative*cofactor[j];''')
    src = replace_once(src,'nu[j]=2*weight*(alpha+beta*tr)*v[j];for(int a=0;a<3;a++)nu[j]-=2*weight*beta*pd[a]*d[a][j];',
        'nu[j]=2*weighted*(alpha+beta*tr)*v[j];for(int a=0;a<3;a++)nu[j]-=2*weighted*beta*pd[a]*d[a][j];')
    src = replace_once(src,'for(int a=0;a<3;a++){double momentum[4];for(int j=0;j<4;j++)momentum[j]=sign[a]*(2*weight*beta*(norm*d[a][j]-pd[a]*v[j])-advection[a]*nu[j])/h;link(base,field+index[a],s[a],momentum,gn+pos,gn+index[a]);}',
        '''for(int a=0;a<3;a++){
    double momentum[4], cofactor_d[4];
    if(a==0)row_cofactors(base,d[1],d[2],cofactor_d);
    if(a==1)row_cofactors(base,d[0],d[2],cofactor_d);
    if(a==2)row_cofactors(base,d[0],d[1],cofactor_d);
    const double alternation=a==1?1.:-1.;
    for(int j=0;j<4;j++)momentum[j]=sign[a]*(2*weighted*beta*(norm*d[a][j]-pd[a]*v[j])
      -advection[a]*nu[j]+alternation*topological_derivative*cofactor_d[j])/h;
    link(base,field+index[a],s[a],momentum,gn+pos,gn+index[a]);
   }''')
    src = replace_once(src,'double K[4][4],R[9][4]={{0.}},KR[4][9];','''double cofactor[4];
   row_cofactors(d[0],d[1],d[2],cofactor);
   double rho=dot4(base,cofactor), weighted=weight*(1.+lambda*rho*rho);
   double K[4][4],R[9][4]={{0.}},KR[4][9];''')
    return replace_once(src,'K[j][k]=weight*v;','K[j][k]=weighted*v;')


class TopologicalKinetic:
    def __init__(self,beta,lam):
        self.beta = beta
        self.lam = lam
        expected = source_code()
        if not SOURCE.exists():
            SOURCE.write_text(expected)
        elif SOURCE.read_text()!=expected:
            raise ValueError('The topological kinetic source differs from the audited extension')
        args = [ctypes.c_int,*([ctypes.c_double]*4)]
        self.qfun,self.hash = load_c(SOURCE,'bst_topological_q',
            [*args,ARRAY,ARRAY,ARRAY,ARRAY,ARRAY,ARRAY,ctypes.POINTER(ctypes.c_double)])
        self.mfun,_ = load_c(SOURCE,'bst_topological_mass',[*args,ARRAY,ARRAY,ARRAY,ARRAY])

    def quadratic(self,n,h,t,v):
        gn,gt,gv = np.zeros_like(n),np.zeros_like(n),np.zeros(9)
        Q = ctypes.c_double()
        if self.qfun(round(len(n)**(1/3)),h,1.,self.beta,self.lam,np.ascontiguousarray(n),
                     np.ascontiguousarray(t),np.ascontiguousarray(v),gn,gt,gv,ctypes.byref(Q)):
            raise ValueError('Weighted gnomonic chart failed')
        return Q.value,gn,gt,gv

    def mass(self,n,h):
        A,C,G = np.zeros((len(n),4,4)),np.zeros((len(n),4,9)),np.zeros((9,9))
        if self.mfun(round(len(n)**(1/3)),h,1.,self.beta,self.lam,np.ascontiguousarray(n),A,C,G):
            raise ValueError('Weighted gnomonic mass chart failed')
        return A,C,G
