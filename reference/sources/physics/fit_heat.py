'A single temperature-dependent free energy patch, tested on the 36 states and 24 viriels.'
import json
import numpy as np
from scipy.constants import R
from scipy.optimize import minimize
from readout_ir import O,decode,save
def main():
    in0=json.loads((O/'CALORIMETRIE_ET_VIRIELS_R573.json').read_text());parent=decode('THERMO_R569.json');size=decode('RAYONS_R569.json')['radii_recalcules']
    solutions={}
    for mod in (0,1):
        lines=in0['results'][f'25_{mod}']['rows'];pair=parent['trials'][mod]['rows']
        T=np.array([r['K'] for r in lines]);p=np.array([r['bar'] for r in lines])*1e5;g=np.array([np.log(size[r['gas']]) for r in lines])
        jt=np.array([r['jets_viriels'] for r in lines]);ref=np.array([[r['reference_rho'],r['reference_Ures'],r['Cv_ref'],r['Cv_res_ref']] for r in lines])
        Tv=np.array([r['T'] for r in pair]);gv=np.array([np.log(size[r['g']]) for r in pair])
        b2=np.array([r['B2_calc'] for r in pair]);b2ref=np.array([r['B2'] for r in pair]);u2=np.array([r['d2'] for r in pair]);b3ok=max(abs(r['B3_calc']-r['B3'])/r['d3'] for r in pair)
        def basis(tt,gg,m):
            x=400./tt-1;dx=-400./tt**2;ddx=800./tt**3
            js=[2,0,1,3][:m]
            vals=np.column_stack([x**j for j in js])
            d=np.column_stack([j*x**(j-1)*dx if j else np.zeros_like(x) for j in js])
            dd=np.column_stack([(j*(j-1)*x**(j-2)*dx**2 if j>=2 else 0)+(j*x**(j-1)*ddx if j else np.zeros_like(x)) for j in js])
            return [np.c_[v,v*gg[:,None]] for v in (vals,d,dd)]
        tries=[]
        for m in [1,2,3,4]:
            Z=basis(T,g,m);B=basis(Tv,gv,m)[0]
            def observ(q):
                b,bt,btt=(jt[:,i,0]+z@q*1e-6 for i,z in enumerate(Z));c,ct,ctt=jt[:,:,1].T
                den=p/(R*T)
                for _ in range(5):den-=(den*(1+b*den+c*den**2)-p/(R*T))/(1+2*b*den+3*c*den**2)
                u=-R*T*T*(bt*den+.5*ct*den**2)
                cc=-R*T*((2*bt+T*btt)*den+.5*(2*ct+T*ctt)*den**2)
                vals=np.column_stack([den,u,1.5*R+cc,cc])
                return vals, (b2+B@q-b2ref)/u2
            def margins(x):
                y,z=observ(x[:-1]);u=(y/ref-1)*10
                return np.r_[x[-1]-u.ravel(),x[-1]+u.ravel(),1-z,1+z]
            s0=np.zeros(2*m+1);s0[-1]=1.5
            f=minimize(lambda q:q[-1],s0,method='SLSQP',bounds=[(-50,50)]*(2*m)+[(0,5)],constraints=[dict(type='ineq',fun=margins)],options=dict(maxiter=450,ftol=1e-10))
            vv,b2em=observ(f.x[:-1]);er=abs(vv/ref-1)
            row=dict(powers=[2,0,1,3][:m],coefficients=f.x[:-1].tolist(),status_opt=f.message,maxima_pct=(er.max(axis=0)*100).tolist(),B2_max_over_marge=max(abs(b2em)),B3_max_over_marge=b3ok,passe=bool(np.max(er)<=.1 and max(abs(b2em))<=1+1e-8 and b3ok<=1),all_values=vv.tolist(),B2_units_of_marge=b2em.tolist())
            tries.append(row)
            if row['passe']:break
        solutions[str(mod)]=tries
    save('AJUSTEMENT_ENERGIE_LIBRE_R573.json',dict(law='A_res/(nRT) reçoit rho_molaire × deltaB2 ; deltaB2 en cm³/mol = somme (a_j+b_j ln(r/r_Ar))(400 K/T−1)^j. Dérivées d’ordre un et deux exactes. Même rayon R183 que l’optique. Le coefficient de troisième viriel appartient à la loi parente. Ce terme effectif de l’équation d’état n’a pas été réintroduit dans un potentiel microscopique de transport.',order_outputs=['densité','Residual U','Cv total','Residual cv'],measurements='36 HEOS reference states, 24 virial values with R569. All are used for adjustment.',branches=solutions))
    print([(k,[(i['passe'],i['maximum'] if 'maximum' in i else max(i['maxima_pct']),i['B2_max_sur_marge']) for i in v]) for k,v in solutions.items()])
if __name__=='__main__':main()
