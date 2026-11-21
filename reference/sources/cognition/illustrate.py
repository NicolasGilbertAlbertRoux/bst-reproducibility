'Figures of the chapter. The option --pgf also generates the illustration to be incorporated into the TeX.'
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.lines import Line2D
import numpy as np

INK='#20343b'; TEAL='#006b68'; ORANGE='#aa4c20'; VIOLET='#684c85'
LAVENDER='#eee8f6'; YELLOW='#faf2df'; BLUE='#235a86'; GRIS='#43565b'
VERTE='#badbd0'; SKIN='#ecc9b3'; ROW='#d8e1e0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.2,'axes.edgecolor':'#93a3a3',
  'axes.labelcolor':INK,'xtick.color':GRIS,'ytick.color':GRIS,'text.color':INK,
  'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.65,
  'xtick.major.width':.6,'ytick.major.width':.6,'svg.hashsalt':'bst-porteurs-2026',
  'pgf.texsystem':'xelatex','pgf.rcfonts':False,
  'pgf.preamble':r'\usepackage{fontspec}\setmainfont{Latin Modern Roman}\setsansfont{Latin Modern Sans}'})

def fr(v,p=None):
    return ('%g'%v).replace('-', '−').replace('.',',')

def axis(a):
    a.tick_params(labelsize=9.5,length=3)
    a.set_axisbelow(True);a.grid(axis='y',color=ROW,lw=.6)
    a.xaxis.set_major_formatter(FuncFormatter(fr));a.yaxis.set_major_formatter(FuncFormatter(fr))

def header(fig,titre,under):
    fig.text(.046,.971,titre,ha='left',va='top',fontsize=16.5,weight='bold')
    fig.text(.047,.903,under,ha='left',va='top',fontsize=9.8,color=GRIS)

def save(fig,repertoire,name,pgf):
    fig.savefig(repertoire/(name+'.png'),dpi=205,facecolor='white')
    fig.savefig(repertoire/(name+'.svg'),metadata={'Date':None},facecolor='white')
    if pgf:
        fig.savefig(repertoire/(name+'.pgf'),facecolor='white')
    plt.close(fig)

def channel(data,out,pgf):
    w=np.genfromtxt(data/'Canal_H528_continu.csv',delimiter=',',names=True)
    checks=np.genfromtxt(data/'Canal_H528_controles.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    f=plt.figure(figsize=(6.6,4.0));ax=f.add_axes([.115,.224,.84,.565]);axis(ax)
    header(f,'Quand le canal devient aveugle','H528 - Contrast between responses from two traces of opposite signs')
    ax.axvspan(0,.2,color=LAVENDER); ax.axvline(.2,color=VIOLET,ls=':',lw=1)
    v=w['eta'][::4];ax.fill_between(v,w['contrast_min_J_002'][::4],w['contrast_max_J_020'][::4],color=VERTE)
    ax.plot(v,w['contrast_max_J_020'][::4],lw=.85,color=TEAL);ax.plot(v,w['contrast_min_J_002'][::4],lw=.85,color=TEAL)
    ax.plot(v,w['contrast_J_010'][::4],color=INK,lw=1.9)
    for test in checks:
        if test['bath']=='m1' and np.isclose(test['J'],.1):
            ax.plot(test['eta'],test['estimation_trajectoires'],'o',ms=4.3,mec=INK,mew=.9,mfc='white',zorder=8)
    ax.set(xlim=(.8,-.014),ylim=(-.14,4.25),xticks=[.8,.6,.4,.2,0],yticks=[0,1,2,3,4],
           xlabel='Couplage de la trace à la voie de lecture  $\\eta$',ylabel='Contraste initial')
    ax.text(.105,3.65,'Faible couplage\nexploré ici',color=VIOLET,fontsize=9.3,ha='center',va='center')
    f.text(.048,.066,'Bande verte : impulsion J de 0,02 à 0,20. Trait : J = 0,10. Cercles : calcul des trajectoires.',
           color=GRIS,fontsize=8.6)
    save(f,out,'BST_Canal_vers_le_latent',pgf)


def directions(ax, position, angles):
    # The global directions Uj are shown in their common rotational frame of reference.
    drawing=DrawingArea(25,25); center=np.array([12.5,12.5])
    drawing.add_artist(Circle(center,11,fill=False,edgecolor='#87a29d',lw=.7))
    for theta in sorted(set(angles)):
        d=np.array([np.cos(theta),np.sin(theta)]); per=np.array([-d[1],d[0]])
        bout=center+10*d
        drawing.add_artist(Line2D([center[0],bout[0]],[center[1],bout[1]],color=TEAL,lw=.95))
        drawing.add_artist(Polygon([bout,bout-3*d+1.3*per,bout-3*d-1.3*per],closed=True,facecolor=TEAL))
    ax.add_artist(AnnotationBbox(drawing,position,frameon=False,pad=0))

def memory(data,out,pgf):
    w=np.genfromtxt(data/'Ordre_collectif.csv',delimiter=',',names=True)
    f=plt.figure(figsize=(6.6,5.12));ax=f.add_axes([.105,.485,.84,.31]);axis(ax)
    header(f,'One same order defeated, two exits','Built-in witness - Eight pairs of waves carry an eight-bit word')
    t=w['t'][::5];ax.axvspan(5,8.5,color=LAVENDER)
    ax.fill_between(t,w['min_T_sampled'][::5],w['max_T_sampled'][::5],color=VERTE)
    ax.plot(t,w['T_4'][::5],color=TEAL,lw=2)
    ax.set(xlim=(0,8.5),ylim=(-.06,1.06),xticks=[0,2,4,6,8],yticks=[0,.5,1],
           xlabel='Control time (reduced units)',ylabel='Ordre global D')
    ax.text(6.6,.78,'Ordre global nul\ndans les neuf cas',color=VIOLET,fontsize=9.5,ha='center',va='center')
    ax.text(.15,.19,'Durées de 3 à 5\nTrait : durée 4',color=TEAL,fontsize=8.8,ha='left')
    directions(ax,(1.15,.53),[0.]*8)
    directions(ax,(6.6,.31),list(2*np.pi*np.arange(8)/8))
    ax2=f.add_axes([.047,.075,.92,.267]);ax2.set(xlim=(0,1),ylim=(0,1));ax2.axis('off')
    ax2.text(0,.99,'Lecture fine après la désorganisation (t = 8)',fontsize=11.0,weight='bold',va='top')
    rows=[(.52,'Protected relationship','10110100',TEAL,'256 mots distincts'),
          (.15,'Obscurelated relationship','00000000',GRIS,'1 common final statement')]
    for y,lab,mot,c,count in rows:
        ax2.text(0,y+.016,lab,fontsize=10,va='center')
        for i,bit in enumerate(mot):
            x=.301+.043*i
            ax2.add_patch(Rectangle((x,y-.105),.037,.245,facecolor=c if bit=='1' else '#f4f5f1',
                                    edgecolor=c,linewidth=.7))
            ax2.text(x+.0185,y+.016,bit,color='white' if bit=='1' else c,fontsize=8.9,ha='center',va='center')
        ax2.text(.665,y+.016,count,color=c,fontsize=9.4,va='center')
    f.text(.048,.032,'Initial Word Illustrated: 10110100. In the erased branch, all the words have the same output.',
           fontsize=8.5,color=GRIS)
    save(f,out,'BST_Order_et_memoire',pgf)


def window(data,out,pgf):
    w=np.genfromtxt(data/'Lecteur_et_resolution.csv',delimiter=',',names=True)
    par=json.loads((data/'Bilan_prolongement.json').read_text())['pairs_of_phases']
    f=plt.figure(figsize=(6.6,3.9));ax=f.add_axes([.125,.237,.82,.55]);axis(ax)
    header(f,'What a reader can distinguish','Same physical state - Two possible signs for a phase relationship')
    r=w['sensibilite'];threshold=par['threshold_of_readout']
    ax.axvspan(0,threshold,color=LAVENDER);ax.axvline(threshold,color=VIOLET,ls=':',lw=1)
    ax.fill_between(r,w['sign_plus_min'],w['sign_plus_max'],color=VERTE)
    ax.fill_between(r,w['sign_minus_min'],w['sign_minus_max'],color=SKIN,alpha=.85)
    ax.plot(r,w['sign_plus_min'],color=TEAL,lw=1.15);ax.plot(r,w['sign_minus_max'],color=ORANGE,lw=1.15,ls='--')
    ax.axhline(0,color=GRIS,lw=.7)
    ax.text(.69,.65,'Signe +',color=TEAL,fontsize=10.2,weight='bold',ha='center')
    ax.text(.69,-.64,'Signe −',color=ORANGE,fontsize=10.2,weight='bold',ha='center')
    ax.text(.08,.78,'Lecture\nambiguë',fontsize=9.2,color=VIOLET,ha='center',va='center')
    ax.set(xlim=(1,.0),ylim=(-1.2,1.2),xticks=[1,.8,.6,.4,.2,0],yticks=[-1,-.5,0,.5,1],
           xlabel='Sensitivity to phase shift r (left left access)',ylabel='Accessible value')
    f.text(.048,.06,'Reading failure limited to 0.12. The strips touch each other for r close to 0.170.',
           fontsize=9,color=GRIS)
    save(f,out,'BST_Fenetre_du_readout',pgf)


def time_h528(data,out,pgf):
    with np.load(data/'Trajectoires_H528_prolongees.npz') as z:
        time=z['time'];q=z['q'];m=z['sign'];x=z['x_sur_J'];pair=z['pair_illustree']
    f=plt.figure(figsize=(6.6,5.12))
    header(f,'The same dynamic up to time 24','H528 - 144 preparations, of which 72 for each initial sign')
    axes=[f.add_axes([.12,.535,.84,.26]),f.add_axes([.12,.185,.84,.27])]
    for a,mat,lab,yt,limits in [(axes[0],x,'Réponse x/J',[-.4,0,.4],(-.5,.5)),
                                (axes[1],q*m,'Module de q',[.98,1,1.02],(.978,1.022))]:
        axis(a);a.axvspan(8,24,color=YELLOW);a.axvline(8,color='#846733',lw=1,ls=':')
        v=time[::4]
        for sign,i,c,face,sty in [(-1,pair[0],TEAL,VERTE,'-'),(1,pair[1],ORANGE,SKIN,'--')]:
            l=mat[:,m==sign];a.fill_between(v,l.min(axis=1)[::4],l.max(axis=1)[::4],color=face,alpha=.64)
            a.plot(v,mat[::4,i],color=c,lw=1.25,ls=sty)
        a.set(xlim=(0,24),ylim=limits,yticks=yt,xticks=np.arange(0,25,4),ylabel=lab)
    axes[0].tick_params(labelbottom=False)
    axes[0].text(16,.39,'Calculated extension',ha='center',color='#715823',fontsize=9.1,
                 bbox=dict(fc=YELLOW,ec='none',pad=1.3))
    axes[1].set_xlabel('Reduced time (no conversion to human life)')
    f.text(.048,.052,'Bands: extremes of sweeping. Traits: the same reference preparation, sign plus (full) or (shots).',
           fontsize=8.3,color=GRIS)
    save(f,out,'BST_Prolongement_temporel',pgf)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--pgf',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for make in (channel,memory,window,time_h528):
        make(a.data_field,a.output,a.pgf)
