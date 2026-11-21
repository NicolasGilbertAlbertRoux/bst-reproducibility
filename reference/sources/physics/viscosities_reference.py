'Compare frozen transport predictions with reference correlations.'
from pathlib import Path

from CoolProp.CoolProp import PropsSI
import numpy as np
from scipy.constants import Avogadro, Boltzmann, R

MASSES = np.array([39.948, 83.798, 131.293]) * .001

FLUIDS = dict(Ar='Argon',Kr='Krypton',Xe='Xenon')
KR_DILUTE = [0.,.91297123,-.10014703,-.024547423,.031450093,-.0044562573,
             -.0045112433,.0022375443,-.00014554223,-.00020063853,.000083412883,
             -.000015202363,.0000011590853]
XE_DILUTE = [0.,.9652514,-.05237199,-.06758414,.02855787,.01002789,-.009639621,
             .001329770,.001114305,-.0005992234,.0001224218,-.000009584978]
XE_DENSITY = [-19.572881,219.73999,-1015.3226,2471.0125,-3375.1717,2491.6597,
              -787.26086,14.085455,-.34664158]
SOURCES = {
    'Ar':'https://doi.org/10.1023/B:IJOT.0000022327.04529.f3',
    'Kr':'https://doi.org/10.1007/s10765-021-02927-5',
    'Xe':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8356199/',
}


def density_virials(T,gas):
    return [PropsSI(key,'T',T,'Dmolar',1e-6,gas) for key in ('Bvirial','Cvirial')]


def krypton(T,rho):
    g = FLUIDS['Kr']
    gas_R = PropsSI('gas_constant',g)
    molecules = gas_R/Boltzmann
    mass = MASSES[1]/molecules
    a = np.log(T/298.15)
    eta0 = 25.3062*np.exp(np.polynomial.polynomial.polyval(a,KR_DILUTE))
    if rho < 1e-5:
        return eta0
    B2,dB2 = [PropsSI(k,'T',T,'Dmolar',1e-10,g) for k in ('Bvirial','dBvirial_dT')]
    beta = (B2+T*dB2)/molecules
    s = -PropsSI('Smolar_residual','T',T,'Dmolar',rho,g)/gas_R
    residual = 1.05*np.expm1(np.polynomial.polynomial.polyval(s,[0.,.125364,.220795,-.0313726,.00313907]))
    factor = (rho*molecules/s)**(2./3.)
    return float(factor*(eta0*beta**(2./3.)+residual*1e6*np.sqrt(mass*Boltzmann*T)))


def xenon(T,rho):
    eta0 = 23.0183*np.exp(np.polynomial.polynomial.polyval(np.log(T/298.15),XE_DILUTE))
    tr = T/289.733
    dr = rho*MASSES[2]/1102.8612
    reduced_T = T/250.
    exponents = np.r_[-.25*np.arange(7),-2.5,-5.5]
    initial = np.dot(XE_DENSITY,reduced_T**exponents)*Avogadro*(.396e-9)**3*rho
    c0,c1,c2,c3 = (1.396328251,.0005418871011,4.478809952,24.91698858)
    residual = dr**(2./3.)*np.sqrt(tr)*(tr+c0*tr*dr**4+c1*dr**7/tr+(c2+c3*dr)/tr**2)
    # The critical enhancement is omitted. Dense near-critical points
    # in this scan are sensitivity checks against the regular contribution.
    return float(eta0*(1.+initial)+residual)


def reference(gas,T,rho):
    if gas=='Kr':
        return krypton(T,rho)
    if gas=='Xe':
        return xenon(T,rho)
    return PropsSI('viscosity','T',T,'Dmolar',rho,FLUIDS[gas])*1e6

