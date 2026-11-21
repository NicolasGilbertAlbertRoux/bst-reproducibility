'Compact material window for both the support radius and its kinetic loading.'
import ctypes
from pathlib import Path

from sweep_sf4 import read
from kinetic_local import LocalKinetic,qualified_source
from dynamics_size import Molecule,ROOT
from flux_nodal import replace_once
from motion_local import CALIBRATION,GATE,LocalMolecule
from refit_dependencies import ARRAY,load_c
from size_dynamics import Radius

OUTPUT = ROOT/'results'/'R360'
PARENT_R = Path(__file__).with_name('rayon_local_R358.c')
PARENT_K = Path(__file__).with_name('cinetique_locale_R358.c')
RADIUS_SOURCE = Path(__file__).with_name('rayon_fenetre_R360.c')
KINETIC_SOURCE = Path(__file__).with_name('cinetique_fenetre_R360.c')
RATIOS = (3.,5.)
RADIUS_GATE = """static double spatial_window(const double *p, double rin2, double rout2) {
    const double r2 = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
    double unused;
    return gate(r2,rin2,rout2,&unused);
}
"""


def radius_source():
    source = PARENT_R.read_text()
    source = replace_once(source,'int bst_r358_radius(',RADIUS_GATE+'\nint bst_r360_radius(')
    source = replace_once(source,'double lo, double hi, int differentiate',
                           'double lo, double hi, double rin2, double rout2, int differentiate')
    return replace_once(source,'const double g=gate(v[0][0],lo,hi,&dg);',
        'const double w_space=spatial_window(point,rin2,rout2);\n'
        '                const double g=w_space*gate(v[0][0],lo,hi,&dg);\n'
        '                dg *= w_space;')


def kinetic_source():
    source = PARENT_K.read_text()
    source = replace_once(source,'int bst_local_q(',RADIUS_GATE+'\nint bst_window_q(')
    source = replace_once(source,'int bst_local_mass(','int bst_window_mass(')
    old = 'double lo,double hi,double a,const double*field'
    assert source.count(old)==2
    source = source.replace(old,'double lo,double hi,double a,double rin2,double rout2,const double*field')
    old = 'const double g=gate(base[0],lo,hi,&dg);'
    assert source.count(old)==2
    return source.replace(old,'const double w_space=spatial_window(xi,rin2,rout2);\n'
                              '  const double g=w_space*gate(base[0],lo,hi,&dg);\n'
                              '  dg *= w_space;')


class WindowRadius(Radius):
    def __init__(self,lo,hi,radii):
        source = qualified_source(RADIUS_SOURCE,radius_source())
        fun,self.sha256 = load_c(source,'bst_r360_radius',
            [ctypes.c_int]+[ctypes.c_double]*5+[ctypes.c_int]+[ARRAY]*4)
        self.function = lambda n,h,*args:fun(n,h,lo,hi,radii[0]**2,radii[1]**2,*args)


class WindowKinetic(LocalKinetic):
    def __init__(self,beta,lo,hi,radii):
        if not lo<hi<=1. or not 0.<radii[0]<radii[1]:
            raise ValueError('Expected regular compact gates')
        self.constants = (1.,beta,lo,hi)
        source = qualified_source(KINETIC_SOURCE,kinetic_source())
        args = [ctypes.c_int]+[ctypes.c_double]*8
        q,self.sha = load_c(source,'bst_window_q',args+[ARRAY]*7)
        m,_ = load_c(source,'bst_window_mass',args+[ARRAY]*4)
        radii2 = [x*x for x in radii]
        self.qfun = lambda n,h,alpha,b,lo,hi,a,*row:q(n,h,alpha,b,lo,hi,a,*radii2,*row)
        self.mfun = lambda n,h,alpha,b,lo,hi,a,*row:m(n,h,alpha,b,lo,hi,a,*radii2,*row)


def law():
    return next(x for x in read(CALIBRATION)['gate_catalog_all_results'] if tuple(x['thresholds'])==GATE)


def window():
    return tuple(x*law()['reference_radius_native'] for x in RATIOS)


class WindowMolecule(LocalMolecule):
    def __init__(self,species,padding=0):
        Molecule.__init__(self,species,padding=padding,workers=4)
        self.local_law = law()
        self.p,self.ref = (self.local_law[k] for k in ('exponent','reference_radius_native'))
        self.kernel,self.radius = WindowKinetic(self.beta,*GATE,window()),WindowRadius(*GATE,window())
