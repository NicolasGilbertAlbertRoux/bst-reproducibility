/* Same R235 alpha/beta kinetic density, with arbitrary intrinsic nodal rate.
 * Q=2T. Derivatives hold n, intrinsic rate and local rigid velocity independent.
 * The Python embedding subsequently applies their complete chain rules. */
#include <stddef.h>
static double dot4(const double*a,const double*b){double s=0;for(int j=0;j<4;j++)s+=a[j]*b[j];return s;}
static void cross3(const double*a,const double*b,double*c){c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
static void link(const double*a,const double*b,double s,const double*p,double*ga,double*gb){double v=dot4(b,p)/(s*s);for(int j=0;j<4;j++){ga[j]-=p[j]+b[j]*v;gb[j]+=p[j]/s-a[j]*v;}}

static double gate(double x, double lo, double hi, double *dg) {
    *dg = 0.;
    if (x <= lo) return 1.;
    if (x >= hi) return 0.;
    const double t = (hi-x)/(hi-lo);
    *dg = -30.*t*t*(1.-t)*(1.-t)/(hi-lo);
    return t*t*t*(10.+t*(-15.+6.*t));
}

static double spatial_window(const double *p, double rin2, double rout2) {
    const double r2 = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
    double unused;
    return gate(r2,rin2,rout2,&unused);
}

int bst_window_q(int n,double h,double alpha,double beta,double lo,double hi,double a,double rin2,double rout2,const double*field,const double*intrinsic,const double*velocity,double*gn,double*gt,double*gv,double*out){
 const ptrdiff_t stride[3]={(ptrdiff_t)4*n*n,4*n,4};const double volume=h*h*h/8.;double total=0.,Qg=0.;
 const double *omega=velocity,*Omega=velocity+3,*u=velocity+6;
 for(int x=0;x<n;x++)for(int y=0;y<n;y++)for(int z=0;z<n;z++){
  const int coord[3]={x,y,z};const ptrdiff_t pos=x*stride[0]+y*stride[1]+z*stride[2];const double*base=field+pos;
  double xi[3],advection[3];for(int a=0;a<3;a++)xi[a]=h*(coord[a]-.5*(n-1));cross3(Omega,xi,advection);for(int a=0;a<3;a++)advection[a]+=u[a];
  double dg;
  const double w_space=spatial_window(xi,rin2,rout2);
  const double g=w_space*gate(base[0],lo,hi,&dg);
  dg *= w_space;
  const double weight=volume*(1.+a*g);
  if (!(weight>0.)) return 2;
  for(int bits=0;bits<8;bits++){
   int valid=1,sign[3];ptrdiff_t index[3];double d[3][4],s[3],tr=0.;
   for(int a=0;a<3;a++){sign[a]=(bits&(1<<a))?-1:1;if(coord[a]+sign[a]<0||coord[a]+sign[a]>=n){valid=0;break;}index[a]=pos+sign[a]*stride[a];}if(!valid)continue;
   for(int a=0;a<3;a++){s[a]=dot4(base,field+index[a]);if(s[a]<=0.)return 1;for(int j=0;j<4;j++)d[a][j]=sign[a]*(field[index[a]+j]/s[a]-base[j])/h;tr+=dot4(d[a],d[a]);}
   double v[4],r[3],pd[3],nu[4];cross3(omega,base+1,r);for(int j=0;j<4;j++)v[j]=intrinsic[pos+j];for(int j=0;j<3;j++)v[j+1]-=r[j];
   for(int a=0;a<3;a++)for(int j=0;j<4;j++)v[j]-=advection[a]*d[a][j];
   double norm=dot4(v,v),q=(alpha+beta*tr)*norm;for(int a=0;a<3;a++){pd[a]=dot4(v,d[a]);q-=beta*pd[a]*pd[a];}total+=weight*q;Qg+=volume*g*q;gn[pos]+=volume*a*dg*q;
   for(int j=0;j<4;j++){nu[j]=2*weight*(alpha+beta*tr)*v[j];for(int a=0;a<3;a++)nu[j]-=2*weight*beta*pd[a]*d[a][j];gt[pos+j]+=nu[j];}
   double direct[3],mom[3];cross3(omega,nu+1,direct);cross3(nu+1,base+1,mom);for(int j=0;j<3;j++){gn[pos+j+1]+=direct[j];gv[j]+=mom[j];}
   double adv[3],space[3];for(int a=0;a<3;a++)adv[a]=-dot4(nu,d[a]);cross3(xi,adv,space);for(int a=0;a<3;a++){gv[a+3]+=space[a];gv[a+6]+=adv[a];}
   for(int a=0;a<3;a++){double momentum[4];for(int j=0;j<4;j++)momentum[j]=sign[a]*(2*weight*beta*(norm*d[a][j]-pd[a]*v[j])-advection[a]*nu[j])/h;link(base,field+index[a],s[a],momentum,gn+pos,gn+index[a]);}
  }
 }
 out[0]=total;out[1]=Qg;return 0;
}

int bst_window_mass(int n,double h,double alpha,double beta,double lo,double hi,double a,double rin2,double rout2,const double*field,double*A,double*C,double*G){
 const ptrdiff_t stride[3]={(ptrdiff_t)4*n*n,4*n,4};const double volume=h*h*h/8.;
 for(int x=0;x<n;x++)for(int y=0;y<n;y++)for(int z=0;z<n;z++){
  const int coord[3]={x,y,z};const ptrdiff_t pos=x*stride[0]+y*stride[1]+z*stride[2];const double*base=field+pos;const ptrdiff_t node=pos/4;
  double xi[3];for(int a=0;a<3;a++)xi[a]=h*(coord[a]-.5*(n-1));
  double dg;
  const double w_space=spatial_window(xi,rin2,rout2);
  const double g=w_space*gate(base[0],lo,hi,&dg);
  dg *= w_space;
  const double weight=volume*(1.+a*g);
  if (!(weight>0.)) return 2;
  for(int bits=0;bits<8;bits++){
   int valid=1,sign[3];ptrdiff_t index[3];double d[3][4],tr=0.;
   for(int a=0;a<3;a++){sign[a]=(bits&(1<<a))?-1:1;if(coord[a]+sign[a]<0||coord[a]+sign[a]>=n){valid=0;break;}index[a]=pos+sign[a]*stride[a];}if(!valid)continue;
   for(int a=0;a<3;a++){double s=dot4(base,field+index[a]);if(s<=0.)return 1;for(int j=0;j<4;j++)d[a][j]=sign[a]*(field[index[a]+j]/s-base[j])/h;tr+=dot4(d[a],d[a]);}
   double K[4][4],R[9][4]={{0.}},KR[4][9];
   for(int j=0;j<4;j++)for(int k=0;k<4;k++){double v=(j==k)?alpha+beta*tr:0.;for(int a=0;a<3;a++)v-=beta*d[a][j]*d[a][k];K[j][k]=weight*v;A[node*16+4*j+k]+=K[j][k];}
   for(int a=0;a<3;a++){
    double axis[3]={0.},rot[3],adv[3];axis[a]=1.;cross3(axis,base+1,rot);cross3(axis,xi,adv);
    for(int j=0;j<3;j++)R[a][j+1]=-rot[j];
    for(int j=0;j<4;j++){R[a+6][j]=-d[a][j];for(int b=0;b<3;b++)R[a+3][j]-=adv[b]*d[b][j];}
   }
   for(int j=0;j<4;j++)for(int a=0;a<9;a++){double v=0.;for(int k=0;k<4;k++)v+=K[j][k]*R[a][k];KR[j][a]=v;C[node*36+9*j+a]+=v;}
   for(int a=0;a<9;a++)for(int b=0;b<=a;b++){double v=0.;for(int j=0;j<4;j++)v+=R[a][j]*KR[j][b];G[9*a+b]+=v;if(a!=b)G[9*b+a]+=v;}
  }
 }
 return 0;
}
