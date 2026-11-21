'Exact archived R453 mixture function, only the self-contained standard branch adapter added.\nThe fraction=1 branch is deliberately unavailable; it requires a separate archive solver.'
import numpy as np
from scipy.integrate import quad,solve_ivp
from scipy.optimize import root

def standard(matter,radiation):
    constant=1-matter-radiation
    inverse=lambda a:1/np.sqrt(radiation+matter*a+constant*a**4)
    distance=lambda z:quad(inverse,1/(1+z),1.,epsabs=5e-12,epsrel=5e-12)[0]
    return inverse,distance

def expansion(*args):
    raise ValueError('Pure flux is not included in the preregistered R479 bank.')

def mixture(matter, radiation, sigma, fraction, tolerance=2e-9, initial_z=99.):
    if fraction==0.:
        f,d=standard(matter,radiation)
        return f,d,lambda z:1.
    if fraction==1.:
        f,d,_,state=expansion(matter,radiation,sigma,tolerance,initial_z)
        return f,d,lambda z:state(z)["rho_DE_sur_rho_crit0"]/state(0.)["rho_DE_sur_rho_crit0"]

    full=1.-matter-radiation
    constant=(1.-fraction)*full
    dark_flux=fraction*full
    ast=1/(1+initial_z)
    n0=np.log(ast)
    finit=1.5*matter*ast/(radiation+1.5*matter*ast)
    nu0=1.686/sigma

    def terms(n,y,normalization):
        a=np.exp(n)
        m,r=matter/a**3,radiation/a**4
        log_d,f=y[:2]
        nu=nu0*np.exp(-log_d)
        g=normalization*nu*np.exp(-.5*nu**2)
        hub2=m+r+constant+g*f
        df=(1.5*m/hub2-f*f-2*f-f*(-3*m-4*r+g*(nu*nu-1)*f*f)/(2*hub2))/(1+g*f/(2*hub2))
        return hub2,df,constant+g*f

    def calculate(x,dense):
        normalization=np.exp(x[1])

        def rhs(n,y):
            hub2,df,_=terms(n,y,normalization)
            return y[1],df,np.exp(-n)/np.sqrt(hub2)

        solution=solve_ivp(rhs,(n0,0.),[x[0],finit,0.],method="DOP853",rtol=tolerance,atol=.06*tolerance,
                           max_step=.13,dense_output=dense)
        assert solution.success
        return solution,normalization

    def mismatch(x):
        solve,amp=calculate(x,False)
        y=solve.y[:,-1]
        _,_,of=terms(0.,y,amp)
        return y[0],(of-full)/dark_flux

    starting=dark_flux/(nu0*np.exp(-.5*nu0**2)*matter**.55)
    x=root(mismatch,[n0+.27,np.log(starting)],tol=2e-9)
    np.testing.assert_allclose(mismatch(x.x),0.,atol=3e-9)
    solve,amp=calculate(x.x,True)

    def early(a):
        return 1/np.sqrt(radiation+matter*a+constant*a**4)

    def inverse(a):
        if a<=ast:
            return early(a)
        n=np.log(a)
        hub2,_,_=terms(n,solve.sol(n),amp)
        return 1/(a*a*np.sqrt(hub2))

    def distance(z):
        a=1/(1+z)
        if a<ast:
            return solve.y[2,-1]+quad(early,a,ast,epsabs=1e-12,epsrel=1e-10)[0]
        return solve.y[2,-1]-solve.sol(np.log(a))[2]

    def relative(z):
        n=-np.log1p(z)
        return (constant if z>initial_z else terms(n,solve.sol(n),amp)[2])/full

    np.testing.assert_allclose(inverse(1.),1.,atol=1e-9)
    return inverse,distance,relative
