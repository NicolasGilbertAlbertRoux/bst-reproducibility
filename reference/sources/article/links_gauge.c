/* Two discretizations of the same intrinsic first derivative of an S3 field.
 * Coefficients store the differential of d with respect to (a,b,s=a.b). */
#include <math.h>
#include <stddef.h>

#ifndef R383_MIN_GEODESIC_DOT
#define R383_MIN_GEODESIC_DOT 0.
#endif

typedef struct {
    ptrdiff_t offset;
    double d[4], ds[4], da, db, d2, b[4], c, s, rate;
    int valid;
} Link;

static double dot(const double *a,const double *b) {
    double s=0.;
    for (int j=0;j<4;++j) s+=a[j]*b[j];
    return s;
}

static void rotation(double c,double s,const double *b,double *out) {
    out[0]=b[0]; out[1]=c*b[1]-s*b[2]; out[2]=s*b[1]+c*b[2]; out[3]=b[3];
}

static int make_link(Link *z,const double *a,ptrdiff_t offset,double inv_h,int log_map,double angle,double rate) {
    z->c=cos(angle); z->s=sin(angle); z->rate=rate;
    rotation(z->c,z->s,a+offset,z->b);
    const double *b=z->b;
    double s=dot(a,b);
    if (s<=(log_map ? R383_MIN_GEODESIC_DOT : 0.)) return 1;
    z->valid=1;
    z->offset=offset;
    if (log_map) {
        double f,fp;
        const double t=1.-s;
        if (t<1.e-4) {
            f=1.+t*(1./3.+t*(2./15.+t*(2./35.+t*8./315.)));
            fp=-(1./3.+t*(4./15.+t*(6./35.+t*32./315.)));
        } else {
            const double den=1.-s*s;
            f=acos(s)/sqrt(den);
            fp=(s*f-1.)/den;
        }
        z->da=-inv_h*f*s;
        z->db=inv_h*f;
        for (int j=0;j<4;++j) {
            const double p=b[j]-s*a[j];
            z->d[j]=z->db*p;
            z->ds[j]=inv_h*(fp*p-f*a[j]);
        }
    } else {
        z->da=-inv_h;
        z->db=inv_h/s;
        for (int j=0;j<4;++j) {
            z->d[j]=z->db*b[j]-inv_h*a[j];
            z->ds[j]=-z->db*b[j]/s;
        }
    }
    z->d2=dot(z->d,z->d);
    return 0;
}

static void pull(const Link *l,const double *a,const double *p,double *g) {
    double raw[4], moved[4];
    double *gb=g+l->offset;
    const double s=dot(l->ds,p);
    for (int j=0;j<4;++j) {
        g[j]+=l->da*p[j]+s*l->b[j];
        raw[j]=l->db*p[j]+s*a[j];
    }
    rotation(l->c,-l->s,raw,moved);
    for (int j=0;j<4;++j) gb[j]+=moved[j];
}

/* Spatial energy, intrinsic kinetic energy and their exact discrete derivatives.
 * On the prescribed vacuum faces the velocity is zero. At each interior
 * site a signed link appears four times in the eight cell corners. */
static double neighbor_power(const Link *l,const double *a,const double *p,const double *v) {
    return l->db*dot(p,v)+dot(l->ds,p)*dot(a,v);
}

/* R411 uses global E2, E4 and T and an explicit site-dependent inertia. */
int bst_links_ledger(int n,double h,double alpha,const double *beta_nodes,int log_map,int dynamic,
              const double *q,const double *v,double *gu,double *gt,
              double *mass,double *adv,double *out,const double *rate,double *scale_force,double *current,const double *angles,const double *rates,
              const double *time_weight,const double *dlog_time_weight) {
    const ptrdiff_t stride[3]={(ptrdiff_t)4*n*n,4*n,4};
    const double vol=h*h*h, inv_h=1./h;
    double E2=0.,E4=0.,T=0.;
    const ptrdiff_t nodes=(ptrdiff_t)n*n*n;
    int invalid=0;
    /* Equal colors have disjoint x-neighbor writes to both gradients. */
    for(int color=0;color<3;++color) {
    #pragma omp parallel for schedule(static) reduction(+:E2,E4,T) reduction(|:invalid)
    for(int x=color;x<n;x+=3) for(int y=0;y<n;++y) for(int z=0;z<n;++z) {
        const int c[3]={x,y,z};
        const int inter[3]={x>0&&x<n-1,y>0&&y<n-1,z>0&&z<n-1};
        const ptrdiff_t pos=x*stride[0]+y*stride[1]+z*stride[2];
        const double *a=q+pos;
        const ptrdiff_t site=pos/4;
        const double beta=beta_nodes[site];
        Link links[6]={0};
        double e2=0.,e4=0.,ekin=0.;
        double spatial[6][4]={{0.}};
        int bad_site=0;
        for(int axis=0;axis<3;++axis) {
            for(int sign=-1;sign<=1;sign+=2) {
                if(c[axis]+sign<0||c[axis]+sign>=n) continue;
                const int i=2*axis+(sign>0);
                Link *l=links+i;
                const ptrdiff_t edge=sign>0 ? site : site-stride[axis]/4;
                const ptrdiff_t ei=3*edge+axis;
                if(make_link(l,a,sign*stride[axis],inv_h,log_map,sign*angles[ei],sign*rates[ei])) {
                    bad_site=1;
                    break;
                }
                if(sign>0) {
                    e2+=.5*vol*l->d2;
                    if(dynamic) for(int j=0;j<4;++j) spatial[i][j]=alpha*vol*l->d[j];
                }
            }
        }
        if(bad_site) { invalid=1; continue; }
        for(int ax=0;ax<3;++ax) for(int bx=ax+1;bx<3;++bx) {
            const double w=(1.+inter[3-ax-bx])*vol/16.;
            for(int i=2*ax;i<=2*ax+1;++i) for(int j=2*bx;j<=2*bx+1;++j) {
                Link *l=links+i, *r=links+j;
                if(!l->valid||!r->valid) continue;
                const double p=dot(l->d,r->d);
                e4+=w*(l->d2*r->d2-p*p);
                if(dynamic) for(int k=0;k<4;++k) {
                    spatial[i][k]+=2.*beta*w*(r->d2*l->d[k]-p*r->d[k]);
                    spatial[j][k]+=2.*beta*w*(l->d2*r->d[k]-p*l->d[k]);
                }
            }
        }
        E2+=e2;
        E4+=beta*e4;
        scale_force[site]=-2.*beta*e4;
        if(!dynamic) continue;
        for(int i=0;i<6;++i) if(links[i].valid) {
            const Link *l=links+i;
            pull(l,a,spatial[i],gu+pos);
            const double *b_current=l->b;
            const double generator[4]={0.,-b_current[2],b_current[1],0.};
            const ptrdiff_t edge=(i%2 ? site : site+l->offset/4);
            current[3*edge+i/2]+=-(i%2 ? 1. : -1.)*neighbor_power(l,a,spatial[i],generator);
        }
        if(dynamic==2 || !(inter[0]&&inter[1]&&inter[2])) continue;
        const double *u=v+pos;
        const double clock=time_weight[site], dlog=dlog_time_weight[site];
        const double v2=dot(u,u), w=.5*clock*beta*vol;
        double *A=mass+16*(pos/4), *convective=adv+pos;
        const double t2=.5*clock*vol*alpha*v2;
        ekin+=t2;
        scale_force[site]+=dlog*t2;
        for(int j=0;j<4;++j) A[5*j]=clock*alpha*vol;
        for(int i=0;i<6;++i) {
            const Link *l=links+i;
            double ub[4];
            rotation(l->c,l->s,u+l->offset,ub);
            ub[1]-=l->rate*l->b[2];
            ub[2]+=l->rate*l->b[1];
            const double sv=dot(u,l->b)+dot(a,ub);
            double dd[4], p[4];
            const double vu=dot(u,l->d);
            const double t4=.5*w*(l->d2*v2-vu*vu);
            ekin+=t4;
            scale_force[site]+=(2.+dlog)*t4;
            for(int j=0;j<4;++j) {
                dd[j]=l->da*u[j]+l->db*ub[j]+l->ds[j]*sv;
                p[j]=w*(v2*l->d[j]-vu*u[j]);
            }
            pull(l,a,p,gt+pos);
            const double *b_current=l->b;
            const double generator[4]={0.,-b_current[2],b_current[1],0.};
            const ptrdiff_t edge=(i%2 ? site : site+l->offset/4);
            current[3*edge+i/2]+=(i%2 ? 1. : -1.)*neighbor_power(l,a,p,generator);
            const double lu=dot(l->d,dd), vu_d=dot(u,dd);
            for(int j=0;j<4;++j) {
                convective[j]+=w*(2.*lu*u[j]-dd[j]*vu-l->d[j]*vu_d);
                for(int k=0;k<4;++k)
                    A[4*j+k]+=w*((j==k?l->d2:0.)-l->d[j]*l->d[k]);
            }
        }
        for(int j=0;j<4;++j) {
            double p=0.;
            for(int k=0;k<4;++k) p+=A[4*j+k]*u[k];
            const double p4=p-clock*alpha*vol*u[j];
            convective[j]+=rate[site]*(2.*p4+dlog*p);
        }
        T+=ekin;
    }
    }
    out[0]=E2;
    out[1]=E4;
    out[2]=T;
    return invalid;
}

/* The mass is alpha*h^3*I plus positive semidefinite link contributions. */
static int solve_mass_pair(const double *A,const double *force,const double *q,
                           double *a0,double *normal) {
    double C[16]={0.}, f[4], n[4];
    for (int i=0;i<4;++i) {
        for (int j=0;j<=i;++j) {
            double entry=A[4*i+j];
            for (int k=0;k<j;++k) entry-=C[4*i+k]*C[4*j+k];
            if (i==j) {
                if (!(entry>0.)) return 1;
                C[4*i+j]=sqrt(entry);
            } else {
                C[4*i+j]=entry/C[5*j];
            }
        }
        f[i]=force[i];
        n[i]=q[i];
        for (int j=0;j<i;++j) {
            f[i]-=C[4*i+j]*f[j];
            n[i]-=C[4*i+j]*n[j];
        }
        f[i]/=C[5*i];
        n[i]/=C[5*i];
    }
    for (int i=3;i>=0;--i) {
        a0[i]=f[i];
        normal[i]=n[i];
        for (int j=i+1;j<4;++j) {
            a0[i]-=C[4*j+i]*a0[j];
            normal[i]-=C[4*j+i]*normal[j];
        }
        a0[i]/=C[5*i];
        normal[i]/=C[5*i];
    }
    return 0;
}

int bst_acceleration(int n,const double *q,const double *v,const double *gu,
    const double *gt,const double *mass,const double *adv,double *a) {
    int invalid=0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(|:invalid)
    for (int x=1;x<n-1;++x) for (int y=1;y<n-1;++y) for (int z=1;z<n-1;++z) {
        const ptrdiff_t pos=4*((ptrdiff_t)x*n*n+y*n+z);
        double force[4], a0[4], normal[4];
        for (int i=0;i<4;++i) force[i]=gt[pos+i]-gu[pos+i]-adv[pos+i];
        if (solve_mass_pair(mass+4*pos,force,q+pos,a0,normal)) {
            invalid=1;
            continue;
        }
        const double multiplier=(dot(q+pos,a0)+dot(v+pos,v+pos))/dot(q+pos,normal);
        for (int i=0;i<4;++i) a[pos+i]=a0[i]-multiplier*normal[i];
    }
    return invalid;
}
