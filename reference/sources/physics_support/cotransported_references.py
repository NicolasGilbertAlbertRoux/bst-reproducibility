'Full common target SO(3), in the co-transported reference preparation.\n\nExtension of the explicit R249 common-z preparation to the current finite\nembedding. The reference is transported as part of this chosen recipe. No\nindependent reference rotor, mass or fitted conservation term is introduced.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time
import numpy as np
from fields_and_references import Configuration,rotate,dump,plain,ROOT
from dynamics_full import Dynamics
from integrate_dynamics import integrate,target_momentum

class CoTransportedConfiguration(Configuration):
    def __init__(self):
        super().__init__();self.received_reference=self.model.reference
        self.model.reference=lambda i:rotate(self.received_reference(i),self.target_rotation)
    def potential(self,q):
        E,g=super().potential(q)
        field_torque=self.last['target_reference_torque_native'].copy()
        # U(R n,R b)=U(n,b), including normalization and current curvature.
        # Reference differentiation supplies the opposite target covector.
        self.last['field_target_torque_native']=field_torque
        self.last['co_transported_reference_target_torque_native']=-field_torque
        self.last['target_reference_torque_native']=np.zeros(3)
        g[84:87]=0.;self.last['gradient']=g.copy()
        return E,g

class CoTransportedDynamics(Dynamics):
    def __init__(self):
        super().__init__();self.configuration=CoTransportedConfiguration()
    def evaluate(self,y,diagnostics=False):
        dy,H=super().evaluate(y,diagnostics)
        for key in ['field_target_torque_native','co_transported_reference_target_torque_native']:
            self.last[key]=self.configuration.last[key].copy()
        self.last['reference_protocol']='CO_TRANSPORTED_COMMON_SO3'
        return dy,H

def run():
    start=time.time();d=CoTransportedDynamics();q=d.configuration.q0.copy();q[72:84]=np.tile([.12,-.08,.10],4);q[84:87]=[.17,-.11,.09]
    c=d.configuration;E,g=c.potential(q);checks=[]
    for name,sl in [('fields',slice(0,60)),('frames',slice(72,84)),('target_and_references',slice(84,87))]:
        v=np.zeros(87);v[sl]=np.sin(np.arange(len(v[sl]))*.31+.23);v/=np.linalg.norm(v);exact=float(g@v)
        for eps in [2e-5,1e-5]:
            fd=(c.potential(q+eps*v)[0]-c.potential(q-eps*v)[0])/(2*eps);error=abs(exact-fd)
            assert error<8e-5*max(1.,abs(exact)),(name,eps,error)
            checks.append({'family':name,'step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':error})
    y0=d.initial();dy,H=d.evaluate(y0,diagnostics=True);initial=plain(d.last);assert initial['target_reaction_balance_inf_defect']<2e-8
    # H derivatives also test moving-reference dependence at nonzero rotation.
    yn=y0.copy();yn[:87]=q;dyn,Hn=d.evaluate(yn,diagnostics=True);gH=d.last['Hamiltonian_configuration_gradient'].copy()
    for name,sl in [('Hamiltonian_fields',slice(0,60)),('Hamiltonian_target_and_references',slice(84,87))]:
        v=np.zeros(177);v[sl]=np.sin(np.arange(len(v[sl]))*.31+.23);v/=np.linalg.norm(v);exact=float(gH@v[:87]);eps=1e-5
        fd=(d.evaluate(yn+eps*v)[1]-d.evaluate(yn-eps*v)[1])/(2*eps);error=abs(fd-exact);assert error<1e-4*max(1.,abs(exact))
        checks.append({'family':name,'step':eps,'analytic':exact,'finite_difference':fd,'absolute_defect':error})
    dump(ROOT/'resultats/CONTROLES_COTRANSPORT_SO3_R343.json',{'revision':343,'full':True,'checks':checks,'initial':initial,
        'reference_source_protocol_assumed_not_empirically_selected':True,'physical_validation':False})
    print('Co-transported full SO3 controls passed; momentum balance',initial['target_reaction_balance_inf_defect'],flush=True)
    endpoints=[];records=[]
    for n in [4,8]:
        end,record=integrate(d,y0,.4,n,f'CO_N{n}');endpoints.append(end);records.append(record)
    ell0=target_momentum(y0[:87],y0[87:174]);end=endpoints[-1];ell=target_momentum(end[:87],end[87:174]);scale=np.maximum(1.,abs(y0))
    result={'revision':343,'completed':True,'reference_protocol':'CO_TRANSPORTED_COMMON_SO3','duration_native':.4,'steps':[4,8],
         'initial':initial,'last_fine_snapshot':records[-1]['history'][-1],
         'endpoint_scaled_state_inf_difference':float(np.max(abs(endpoints[1]-endpoints[0])/scale)),
         'fine_maximum_energy_relative_drift':records[-1]['maximum_energy_relative_drift'],
         'fine_maximum_spatial_Casimir_relative_drift':records[-1]['maximum_spatial_Casimir_relative_drift'],
         'fine_target_momentum_change_native':ell-ell0,
         'fine_target_momentum_conservation_inf_defect':float(np.max(abs(ell-ell0))),
         'no_added_reference_mass_or_material_coefficient':True,'same_nodal_alpha_beta_kinetic':True,
         'reference_co_transport_is_a_declared_preparation_not_a_derived_physical_selection':True,
         'no_electric_current_identity_or_physical_clock_claimed':True,'short_trajectory_not_a_stability_proof':True,
         'physical_validation':False,'physical_regions_excluded':0,'seconds':time.time()-start}
    dump(ROOT/'resultats/BILAN_COTRANSPORT_SO3_R343.json',result)
    print('Co-transported SO3 runs completed; target conservation',result['fine_target_momentum_conservation_inf_defect'],flush=True)

if __name__=='__main__':run()
