#include <cmath>
// Kirchhoff's spherical mean solution of the three-dimensional wave equation.
// X, rho and T are expressed in h0 units. The weights on S2 sum to one.
static double theta(double X,double rho,double T,int ns,const double* nodes,const double* ww) {
    double sum=0.;
    for(int i=0;i<ns;++i) {
        const double* n=nodes+3*i;
        double x=X+T*n[0], y=rho+T*n[1], z=T*n[2];
        double xi=(x-22.5)/1.5, e=1-xi*xi, r=1.-(y*y+z*z)/25.;
        if(e<=0.||r<=0.) continue;
        double a=.025*e*e*e*r*r*r;
        double dx=-.1*xi*e*e*r*r*r, dyz=-.006*e*e*e*r*r;
        double grad=(dx*n[0]+dyz*(y*n[1]+z*n[2]))/(1.+a*a);
        double initial_speed=-dx/std::sqrt(1.+a*a);
        sum+=ww[i]*(std::atan(a)+T*(grad+initial_speed));
    }
    return sum;
}
extern "C" double energy(double X,double probe,double h0,double eps,
                         int ns,const double* n,const double* w,int nr,const double* r,const double* rw,
                         int nt,const double* ts,const double* tw) {
    double sum=0.;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for(int i=0;i<nt;++i) for(int j=0;j<nr;++j) {
        double t=ts[i]/h0;
        double base=theta(X,r[j],t,ns,n,w);
        double plus=theta(X+probe,r[j],t,ns,n,w),minus=theta(X-probe,r[j],t,ns,n,w);
        double u=(theta(X,r[j],t+eps,ns,n,w)-theta(X,r[j],t-eps,ns,n,w))/(2.*eps);
        // Project two samples of the S1 field onto the tangent at the central field.
        double grad=(std::sin(plus-base)-std::sin(minus-base))/(2.*probe);
        sum-=tw[i]*rw[j]*r[j]*u*grad*2.*std::acos(-1.);
    }
    return sum;
}
