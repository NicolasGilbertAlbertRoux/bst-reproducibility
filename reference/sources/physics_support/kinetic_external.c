/* Same R235 alpha/beta kinetic density, with arbitrary intrinsic nodal rate.
 * Q=2T. Derivatives hold n, intrinsic rate and local rigid velocity independent.
 * The Python embedding subsequently applies their complete chain rules. */
#include <stddef.h>
static double dot4(const double*a,const double*b){double s=0;for(int j=0;j<4;j++)s+=a[j]*b[j];return s;}
static void cross3(const double*a,const double*b,double*c){c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
static void link(const double*a,const double*b,double s,const double*p,double*ga,double*gb){double v=dot4(b,p)/(s*s);for(int j=0;j<4;j++){ga[j]-=p[j]+b[j]*v;gb[j]+=p[j]/s-a[j]*v;}}

int bst_exterior_q(int n,int pad,double h,double alpha,double beta,const double*field,const double*intrinsic,const double*velocity,double*gn,double*gt,double*gv,double*out){
 const ptrdiff_t stride[3]={(ptrdiff_t)4*n*n,4*n,4};const double weight=h*h*h/8.;double total=0.;
 const double *omega=velocity,*Omega=velocity+3,*u=velocity+6;
 for(int x=0;x<n;x++)for(int y=0;y<n;y++)for(int z=0;z<n;z++){
  const int coord[3]={x,y,z};const ptrdiff_t pos=x*stride[0]+y*stride[1]+z*stride[2];const double*base=field+pos;
  double xi[3],advection[3];for(int a=0;a<3;a++)xi[a]=h*(coord[a]-.5*(n-1));cross3(Omega,xi,advection);for(int a=0;a<3;a++)advection[a]+=u[a];
  for(int bits=0;bits<8;bits++){
   int valid=1,sign[3];ptrdiff_t index[3];double d[3][4],s[3],tr=0.;
   for(int a=0;a<3;a++){sign[a]=(bits&(1<<a))?-1:1;if(coord[a]+sign[a]<0||coord[a]+sign[a]>=n){valid=0;break;}index[a]=pos+sign[a]*stride[a];}if(!valid)continue;
   int interior=1;
   for(int a=0;a<3;a++){
    const int neighbor=coord[a]+sign[a];
    if(coord[a]<pad||coord[a]>=n-pad||neighbor<pad||neighbor>=n-pad)interior=0;
   }
   if(interior)continue;

   for(int a=0;a<3;a++){s[a]=dot4(base,field+index[a]);if(s[a]<=0.)return 1;for(int j=0;j<4;j++)d[a][j]=sign[a]*(field[index[a]+j]/s[a]-base[j])/h;tr+=dot4(d[a],d[a]);}
   double v[4],r[3],pd[3],nu[4];cross3(omega,base+1,r);for(int j=0;j<4;j++)v[j]=intrinsic[pos+j];for(int j=0;j<3;j++)v[j+1]-=r[j];
   for(int a=0;a<3;a++)for(int j=0;j<4;j++)v[j]-=advection[a]*d[a][j];
   double norm=dot4(v,v),q=(alpha+beta*tr)*norm;for(int a=0;a<3;a++){pd[a]=dot4(v,d[a]);q-=beta*pd[a]*pd[a];}total+=weight*q;
   for(int j=0;j<4;j++){nu[j]=2*weight*(alpha+beta*tr)*v[j];for(int a=0;a<3;a++)nu[j]-=2*weight*beta*pd[a]*d[a][j];gt[pos+j]+=nu[j];}
   double direct[3],mom[3];cross3(omega,nu+1,direct);cross3(nu+1,base+1,mom);for(int j=0;j<3;j++){gn[pos+j+1]+=direct[j];gv[j]+=mom[j];}
   double adv[3],space[3];for(int a=0;a<3;a++)adv[a]=-dot4(nu,d[a]);cross3(xi,adv,space);for(int a=0;a<3;a++){gv[a+3]+=space[a];gv[a+6]+=adv[a];}
   for(int a=0;a<3;a++){double momentum[4];for(int j=0;j<4;j++)momentum[j]=sign[a]*(2*weight*beta*(norm*d[a][j]-pd[a]*v[j])-advection[a]*nu[j])/h;link(base,field+index[a],s[a],momentum,gn+pos,gn+index[a]);}
  }
 }
 *out=total;return 0;
}

