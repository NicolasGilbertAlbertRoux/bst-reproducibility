/* RMS support of the absolute native topological density and its adjoint. */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static double dot4(const double *a, const double *b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
}

static double minor3(const double *a, const double *b, const double *c,
                     int i, int j, int k) {
    return a[i]*(b[j]*c[k]-b[k]*c[j])
         - a[j]*(b[i]*c[k]-b[k]*c[i])
         + a[k]*(b[i]*c[j]-b[j]*c[i]);
}

static void cofactors(const double *a, const double *b, const double *c, double *cof) {
    cof[0] =  minor3(a,b,c,1,2,3);
    cof[1] = -minor3(a,b,c,0,2,3);
    cof[2] =  minor3(a,b,c,0,1,3);
    cof[3] = -minor3(a,b,c,0,1,2);
}

static double gate(double x, double lo, double hi, double *dg) {
    *dg = 0.;
    if (x <= lo) return 1.;
    if (x >= hi) return 0.;
    const double t = (hi-x)/(hi-lo);
    *dg = -30.*t*t*(1.-t)*(1.-t)/(hi-lo);
    return t*t*t*(10.+t*(-15.+6.*t));
}

int bst_r358_radius(int n, double h, double lo, double hi, int differentiate, const double *field,
                   const double *rate, double *gradient, double *result) {
    const int side = n-1;
    const size_t samples = 8*(size_t)side*side*side;
    double *density = differentiate ? malloc(samples*sizeof(double)) : NULL;
    if (differentiate && !density) return 1;
    const double centre = .5*(n-1);
    const ptrdiff_t stride[3] = {4*(ptrdiff_t)n*n,4*n,4};
    double moment[5] = {0.};
    double radius = 0.;
    double centroid[3] = {0.};
    double cusp = 0.;
    double min_dot = 1.;
    for (int pass=0; pass<=differentiate; ++pass) {
        size_t j=0;
        for (int orientation=0; orientation<8; ++orientation) {
            const int s[3] = {orientation&1 ? -1 : 1,
                              orientation&2 ? -1 : 1,
                              orientation&4 ? -1 : 1};
            for (int i=0;i<side;++i) for (int k=0;k<side;++k) for (int l=0;l<side;++l,++j) {
                const int x[3] = {s[0]>0 ? i : n-1-i, s[1]>0 ? k : n-1-k, s[2]>0 ? l : n-1-l};
                const ptrdiff_t pos = stride[0]*x[0]+stride[1]*x[1]+stride[2]*x[2];
                const ptrdiff_t at[4] = {pos,pos+s[0]*stride[0],pos+s[1]*stride[1],pos+s[2]*stride[2]};
                const double *v[4] = {field+at[0],field+at[1],field+at[2],field+at[3]};
                const double dot[3] = {dot4(v[0],v[1]),dot4(v[0],v[2]),dot4(v[0],v[3])};
                for (int a=0;a<3;++a) {
                    if (!(dot[a]>0.)) { free(density); return 2; }
                    if (dot[a]<min_dot) min_dot=dot[a];
                }
                const double inv = 1./(dot[0]*dot[1]*dot[2]);
                double cof[4][4];
                cofactors(v[1],v[2],v[3],cof[0]);
                double rho = pass==0 ? inv*dot4(v[0],cof[0]) : density[j];
                double point[3] = {(x[0]-centre)*h,(x[1]-centre)*h,(x[2]-centre)*h};
                double dg;
                const double g=gate(v[0][0],lo,hi,&dg);
                if (pass==0) {
                    double w=g*fabs(rho);
                    moment[0]+=w;
                    for (int a=0;a<3;++a) moment[a+1]+=w*point[a];
                    moment[4]+=w*(point[0]*point[0]+point[1]*point[1]+point[2]*point[2]);
                    if (differentiate) density[j]=rho;
                    continue;
                }
                cofactors(v[0],v[2],v[3],cof[1]);
                cofactors(v[0],v[1],v[3],cof[2]);
                cofactors(v[0],v[1],v[2],cof[3]);
                double lever=-radius*radius;
                for (int a=0;a<3;++a) {
                    double delta=point[a]-centroid[a];
                    lever+=delta*delta;
                }
                const double sign = (rho>0.)-(rho<0.);
                const double normal = lever/(2.*radius*moment[0]);
                const double scale = g*sign*normal;
                gradient[at[0]] += dg*fabs(rho)*normal;
                double directional=0.;
                for (int c=0;c<4;++c) {
                    const double d0 = inv*cof[0][c]-rho*(v[1][c]/dot[0]+v[2][c]/dot[1]+v[3][c]/dot[2]);
                    gradient[at[0]+c]+=scale*d0;
                    if (sign==0.) directional+=d0*rate[at[0]+c];
                    for (int a=1;a<4;++a) {
                        const double derivative=(a%2 ? -1. : 1.)*inv*cof[a][c]-rho*v[0][c]/dot[a-1];
                        gradient[at[a]+c]+=scale*derivative;
                        if (sign==0.) directional+=derivative*rate[at[a]+c];
                    }
                }
                if (sign==0.) cusp+=g*fabs(lever*directional);
            }
        }
        if (pass==0) {
            if (!(moment[0]>0.)) { free(density); return 3; }
            double q=moment[4]/moment[0];
            for (int a=0;a<3;++a) {
                centroid[a]=moment[a+1]/moment[0];
                q-=centroid[a]*centroid[a];
            }
            if (!(q>0.)) { free(density); return 4; }
            radius=sqrt(q);
        }
    }
    result[0]=radius;
    result[1]=cusp/(2.*radius*moment[0]);
    result[2]=min_dot;
    free(density);
    return 0;
}
