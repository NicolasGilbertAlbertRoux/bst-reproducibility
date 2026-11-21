// Discrete geodesic action for the native packet. A great-circle reduction represents the field by angle q.
#include <cmath>
#include <cstddef>
extern "C" void acceleration(int nx, int ny, int nz, double h, const double* q,
                             const double* v, double* a, double* forces) {
    const int sizes[3]={nx,ny,nz}, s[3]={ny*nz,nz,1};
    #pragma omp parallel for schedule(static)
    for(int i=0;i<nx;++i) for(int j=0;j<ny;++j) for(int k=0;k<nz;++k) {
        long z=(long)i*s[0]+j*s[1]+k;
        int idx[3]={i,j,k};
        for(int d=0;d<3;++d) {
            double t=idx[d]+1<sizes[d] ? std::tan(q[z+s[d]]-q[z]) : 0.;
            forces[3*z+d]=t*(1.+t*t);
        }
    }
    #pragma omp parallel for schedule(static)
    for(int i=0;i<nx;++i) for(int j=0;j<ny;++j) for(int k=0;k<nz;++k) {
        long z=(long)i*s[0]+j*s[1]+k;
        int idx[3]={i,j,k},faces=0; double value=0.;
        for(int d=0;d<3;++d) {
            value+=forces[3*z+d];
            if(idx[d]) value-=forces[3*(z-s[d])+d];
            if(idx[d]==0 || idx[d]+1==sizes[d]) ++faces;
        }
        a[z]=std::ldexp(value,faces)/(h*h)-2.*faces*v[z]/h;
    }
}
