'Same discrete action, double local arithmetic and extended energy sums.\n\nThis arithmetic variant is qualified against the full-long-double R341\nvariant and by the unchanged native derivative controls before search use.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import ctypes,json,time,hashlib
import numpy as np
from refit_dependencies import load_c,ROOT

class AccurateSumAction:
    def __init__(self,beta):
        self.beta=float(beta)
        original=(ROOT/'sources/action_parametree_R331.c').read_text()
        code=original.replace('bst_action_params','bst_action_compensated').replace('double e2 = 0., e4 = 0.;','long double e2 = 0., e4 = 0.;').replace('double *out)','long double *out)')
        if code==original or 'long double e2' not in code:raise ValueError('Arithmetic source transformation failed')
        target=ROOT/'sources/action_somme_R342.c';target.write_text(code)
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        extended=np.ctypeslib.ndpointer(dtype=np.longdouble,flags='C_CONTIGUOUS')
        assert np.dtype(np.longdouble).itemsize==ctypes.sizeof(ctypes.c_longdouble)
        self.f,self.hash=load_c(target,'bst_action_compensated',[ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double,array,array,extended])
    def __call__(self,n,h):
        n=np.ascontiguousarray(n,dtype=float);g=np.zeros_like(n);out=np.zeros(3,dtype=np.longdouble)
        if self.f(round(len(n)**(1/3)),h,1.,self.beta,n,g,out):raise ValueError('Unqualified energy chart')
        return out[0],g

def main():
    from frames_relative_R342 import FramedRouth
    row=json.loads((ROOT/'resultats/SELECTION_RECALAGE_R341.json').read_text())['row'];m=FramedRouth(row)
    points=[('fit',m.x0[:m.nq].copy())]
    p=json.loads((ROOT/'resultats/PROLONGATION_LEVEE_R341.json').read_text())['result'];points.append(('fin_free',np.array(p['full_coordinates'])))
    precise=m.model.action.action;fast=AccurateSumAction(m.beta);rows=[]
    for name,x in points:
        m.model.action.action=precise;t=time.time();E,g=m.model.evaluate(x);s1=time.time()-t
        m.model.action.action=fast;t=time.time();e,h=m.model.evaluate(x);s2=time.time()-t
        rows.append({'point':name,'extended_potential':float(E),'compensated_potential':float(e),
                     'energy_absolute_difference':float(abs(e-E)),'gradient_inf_absolute_difference':float(max(abs(g-h))),
                     'extended_seconds':s1,'compensated_seconds':s2})
    out={'revision':342,'same_action_terms_coefficients_epsilon_and_tolerances':True,'comparisons':rows,
         'arithmetic_variant_not_yet_native_derivative_qualified':True,'extended_action_source_sha256':precise.hash,
         'compensated_action_source_sha256':fast.hash,'physical_validation':False}
    from mix import dump
    dump(ROOT/'resultats/COMPARAISON_PRECISIONS_R342.json',out)
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
