'Seven fields S3, seven movable marks, same pulse and even six contacts on all steps.'
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.integrate import simpson
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT/'inputs'/'heritage_R360'
sys.path[:0] = [str(HIST/'scripts'), str(HIST/'sources')]
from flux_shape import ShapeHalo
from dynamics_size import Molecule


def stencil(n, coordinates):
    'Local cubic interns. A column of the transpose redistributes the virtual work.'
    x = np.asarray(coordinates)
    left = np.clip(np.floor(x).astype(int)-1,0,n-4)
    indices = left[:,None]+np.arange(4)
    weights = np.stack([np.prod([(x-indices[:,l])/(j-l) for l in range(4) if l!=j],axis=0) for j in range(4)],axis=1)
    return csr_matrix((weights.ravel(),(np.repeat(np.arange(len(x)),4),indices.ravel())),shape=(len(x),n))


def transport(a, mat):
    for axis in range(3):
        b=np.moveaxis(a,axis,0)
        d=mat@b.reshape(len(b),-1)
        a=np.moveaxis(d.reshape((d.shape[0],)+b.shape[1:]),0,axis)
    return np.ascontiguousarray(a)


class Seven(ShapeHalo):
    def __init__(self, n, boundary):
        super().__init__(4)
        self.h0=self.h.copy()
        self.received=self.n0
        self.boundary=boundary
        self.N=n
        self.count=n
        self.scale=55/(n-1)
        self.h=self.scale*self.h0
        self.shape=(7,n**3,4)
        self.nfield=int(np.prod(self.shape))
        self.padding=0
        self.mask=np.ones(n**3,dtype=bool)
        self.model.mask=np.ones(48**3,dtype=bool)
        ijk=np.indices((n,)*3,dtype=float)
        face_count=np.sum((ijk==0)|(ijk==n-1),axis=0)
        if boundary=='tenu':
            self.mask=face_count.ravel()==0
        self.surface=(face_count*0.5**np.maximum(face_count-1,0)).ravel()  # sum of the trapezoidal quadratures of each face
        u=ijk*self.scale
        def carried(field):
            cube=field.reshape(56,56,56,4)
            if n==56:return cube.copy().reshape(-1,4)
            r=np.stack([map_coordinates(cube[...,j],u,order=3,mode='mirror') for j in range(4)],axis=-1)
            return (r/np.linalg.norm(r,axis=-1,keepdims=True)).reshape(-1,4)
        self.n0=np.stack([carried(s) for s in self.received])
        self.physical=(u-27.5).reshape(3,-1)
        self.R=stencil(n,(np.arange(48)+4)/self.scale)
        self.Rt=self.R.T.tocsr()
        self._local=None
        self.probe_spacing=.22
        self.aperture=7.0
        z=np.linspace(-self.aperture,self.aperture,65)
        self.yz=np.meshgrid(z,z,indexing='ij')
        self.darea=(self.h0[1]*(z[1]-z[0]))**2
        self.w_area=np.ones((65,65));self.w_area[[0,-1],:]*=.5;self.w_area[:,[0,-1]]*=.5
        self.release_norm=[]
    def contact_field(self,n):
        if self.N==56:return np.ascontiguousarray(n.reshape(7,56,56,56,4)[:,4:52,4:52,4:52]).reshape(7,-1,4)
        out=[];norms=[]
        for site in n:
            raw=transport(site.reshape((self.N,)*3+(4,)),self.R)
            norm=np.linalg.norm(raw,axis=-1,keepdims=True)
            if norm.min()<.8:raise ArithmeticError('Champ reconstruit hors de sa carte.')
            norms.append(norm);out.append((raw/norm).reshape(-1,4))
        self._norms=norms
        return np.asarray(out)
    def crop(self,n,count=48):
        return self._local
    def potential(self,n,x,O):
        core=self.contact_field(n)
        self._local=core
        try:
            ucore,gcore,force,torque,r=Molecule.potential(self,n,x,O)
        finally:self._local=None
        action=self.model.m.action.action
        def one(i):
            nc=core[i]
            us,gs=action(nc,self.h0[i]); uf,gf=action(n[i],self.h[i])
            # The covectors of the own action are subtracted from the tangent plan of each node.
            dc=gcore[i]-(gs-nc*np.sum(nc*gs,axis=-1,keepdims=True))
            dc-=nc*np.sum(nc*dc,axis=-1,keepdims=True)
            small=dc.reshape((48,)*3+(4,))
            if self.N==56:
                gf.reshape(56,56,56,4)[4:52,4:52,4:52]+=small
            else:
                gf+=transport(small/self._norms[i],self.Rt).reshape(-1,4)
            return uf-us,gf
        add=list(self.pool.map(one,range(7)))
        return ucore+sum(v[0] for v in add),np.array([v[1] for v in add]),force,torque,r
    def site(self,task):
        n,t,h,v,gu,load=task
        if self.boundary=='impedance':
            gu=gu+(self.surface*h*h)[:,None]*t
        return super().site((n,t,h,v,gu,load))
    def dissipated_power(self,s):
        if self.boundary=='tenu':return 0.
        return float(sum(h*h*np.dot(self.surface,np.sum(vel*vel,axis=1)) for h,vel in zip(self.h,s.t)))
    def initial(self,pulse):
        s=self.unpack(self.rest());n,t=s.n.copy(),s.t.copy()
        if pulse:
            x,y,z=self.physical;xi=(x-22.5)/1.5;env=np.maximum(1-xi**2,0);rad=np.maximum(1-(y*y+z*z)/25,0)**3
            n[1,:,2]+=.025*env**3*rad;n[1]/=np.linalg.norm(n[1],axis=-1,keepdims=True)
            t[1,:,2]=.025*6*xi*env**2*rad/(1.5*self.h0[1])
            t[1]-=n[1]*np.sum(t[1]*n[1],axis=-1,keepdims=True)
        return self.project(self.pack(n,t,s.x,s.q,s.v))[0]
    def sample(self,field,matrices):
        a=field.reshape((self.N,)*3+(4,))
        for axis,mat in enumerate(matrices):
            b=np.moveaxis(a,axis,0);k=mat@b.reshape(len(b),-1)
            a=np.moveaxis(k.reshape((k.shape[0],)+b.shape[1:]),0,axis)
        return a[0]
    def probes(self,s):
        # Current of the field in the instantaneous repository of the site. The integral is signed.
        # The displacement, rotation and entrainment of the surface shall be taken into account.
        y,z=self.yz;rows=[]
        for plane in (23.5,24.5):
            def sample(field,shift):
                axes=[np.array([plane]),y[:,0],z[0,:]]
                mats=[stencil(self.N,(u+a+27.5)/self.scale) for u,a in zip(axes,shift)]
                return self.sample(field,mats)
            f=sample(s.n[1],(0,0,0));length=np.linalg.norm(f,axis=-1,keepdims=True);f/=length
            raw=sample(s.t[1],(0,0,0))
            inner=lambda a,b:np.sum(a*b,axis=-1)
            proj=lambda v:(v-f*inner(v,f)[...,None])/length
            td=proj(raw);d=[]
            for e in np.eye(3)*self.probe_spacing:
                d.append(proj((sample(s.n[1],e)-sample(s.n[1],-e))/(2*self.h0[1]*self.probe_spacing)))
            pos=np.stack((np.full_like(y,plane),y,z),-1)*self.h0[1]
            move=np.cross(s.v[1,:3],pos)+s.v[1,3:]
            nu=td-sum(move[...,a,None]*d[a] for a in range(3))
            # L(T,U) with alpha-beta-1 in a vacuum; the S-3 retains the tangent products.
            n2=inner(nu,nu);norm=[inner(g,g) for g in d];vx=[inner(nu,g) for g in d]
            p=nu.copy()
            for i in range(3):p+=norm[i][...,None]*nu-vx[i][...,None]*d[i]
            stress=d[0].copy()
            for j in (1,2):stress+=norm[j][...,None]*d[0]-inner(d[0],d[j])[...,None]*d[j]
            P=-(inner(nu,stress))  # the quartic kinetic terms compensate each other in this current
            kinetic=.5*inner(nu,p)
            E=kinetic+.5*sum(norm)+.5*sum(norm[i]*norm[j]-inner(d[i],d[j])**2 for i in range(3) for j in range(i+1,3))
            moving=P-move[...,0]*E
            rows.append(float((moving*self.w_area).sum()*self.darea))
        return rows


def run(n,bound,dt,duration):
    out=ROOT/'results'/f'MOUVEMENT_N{n}_{bound}_dt{dt:g}.json'
    m=Seven(n,bound);hist={}
    try:
        for case,pulse in [('background',False),('wave',True)]:
            y=m.initial(pulse);rows=[]
            for k in range(round(duration/dt)+1):
                s=m.unpack(y);der,d=m.evaluate(y)
                row=dict(t=round(k*dt,12),H=float(d['H_native']),loss_field=m.dissipated_power(s),currents=m.probes(s),
                         positions=s.x.tolist(),quaternions=s.q.tolist(),schur_min=float(min(d['site_schur_min_eigenvalues'])),
                         work_residual=float(d['energy_derivative']),EL_inf=max(map(float,d['field_and_body_EL_inf'])))
                rows.append(row)
                print(n,bound,case,k,row['currents'],round(time.monotonic(),1),flush=True)
                if k<round(duration/dt): y,_=m.step(y,dt,der)
            hist[case]=rows
        currents=np.array([r['currents'] for r in hist['wave']])-np.array([r['currents'] for r in hist['background']])
        t=np.array([r['t'] for r in hist['wave']])
        balances={}
        for key,rows in hist.items():
            H=np.array([r['H'] for r in rows]);loss=np.array([r['loss'] for r in rows])
            balances[key]={'energy_perdue':float(simpson(loss,x=t)),'defaut_H_plus_perte':float(H[-1]-H[0]+simpson(loss,x=t))}
        ans={'N':n,'h_F':m.h[1],'h_S':m.h[0],'condition':bound,'dt':dt,'duration':duration,'poses':7,'historique':hist,
             'radius_ouverture_face_h0':m.aperture,'currents_nets':currents.tolist(),'energy_nette_two_surfaces':simpson(currents,x=t,axis=0).tolist(),'bilans':balances,
             'direction_of_coupes':'Carrés centrés de demi-côté 7h₀, fluor F1 du SF6. Positions x=23,5h₀ et 24,5h₀ ; les valeurs donnent le flux d’énergie sur ces faces mobiles. Le bilan de R441 portait sur la totalité de deux volumes extérieurs et sur les sept sites.',
             'scope':'Unitary local dissipative impedance or held faces, same native energy, six contacts retained on the definition quadrature R354 (483), cubic interpolation and transposition in virtual work. Cubes of site F1 length 55h0. Conditions compared in the same reference frame and at the same duration.'}
        out.write_text(json.dumps(ans,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    finally:m.close()

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('n',type=int);cli.add_argument('condition',choices=['tenu','impedance']);cli.add_argument('--dt',type=float,default=.09);cli.add_argument('--duree',type=float,default=.36)
    a=cli.parse_args()
    if abs(round(a.duration_field/a.dt)*a.dt-a.duration_field)>1e-12:raise ValueError('Number of non-full steps.')
    run(a.n,a.condition,a.dt,a.duration_field)
