'Plank of the calculated values and export of each coordinate in the axes units.'
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

root=Path(__file__).resolve().parents[1]
fig_dir=root/'figures'
fig_dir.mkdir(exist_ok=True)
def read(path):
    return json.loads((root/path).read_text())
def decimal(value,_):
    return f'{value:g}'.replace('-', '−').replace('.',',')
def main():
    native=list(csv.DictReader((root/'entrees/COURBES_DU_SUBSTRAT_R572.csv').open(encoding='utf-8-sig')))
    contacts=read('entrees/CONTACT_LIBRE_ET_CYCLE_R571.json')
    actions=read('resultats/UNITES_ET_RUPTURE_R573.json')['actions_SFx']
    flux=read('resultats/FLUX_CORRIGE_R573.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.7,'axes.labelcolor':'#334a50','xtick.color':'#496167','ytick.color':'#496167',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a3b5b4','axes.axisbelow':True,'legend.frameon':False,'legend.fontsize':9.5,
        'figure.facecolor':'#fcfdfb','axes.facecolor':'#fcfdfb','lines.solid_capstyle':'round'})
    fig,axes=plt.subplots(3,2,figsize=(11.1,11.9))
    fig.subplots_adjust(left=.088,right=.978,top=.862,bottom=.075,hspace=.79,wspace=.34)
    fig.text(.088,.967,'BST   /   R573',color='#11665f',size=11,weight='bold')
    fig.text(.088,.930,'The curves of the substrate',size=25,color='#192f39',fontfamily='DejaVu Serif')
    fig.text(.088,.904,'Field and contacts in native units, bubble response and transport of the SF6 pulse',
             size=11,color='#425d63')
    green,copper,blue='#116f66','#a25530','#315f8b'
    export=[]
    def panel(a,title,x,y):
        a.set_title(title,loc='left',pad=10,fontweight='bold',fontsize=13,color='#1b3841')
        a.set_xlabel(x);a.set_ylabel(y);a.grid(axis='y',color='#dfE8E4',lw=.8)
        a.xaxis.set_major_formatter(FuncFormatter(decimal));a.yaxis.set_major_formatter(FuncFormatter(decimal))
        a.xaxis.set_major_locator(MaxNLocator(5));a.yaxis.set_major_locator(MaxNLocator(5))
    def series(a,group,label,x,y,ux,uy,origin,color,**kw):
        a.plot(x,y,label=label,color=color,lw=2.35,**kw)
        export.extend(dict(plot=group,series=label,x=float(xx),y=float(yy),unit_x=ux,unit_y=uy,origin=origin) for xx,yy in zip(x,y))
    a,b,c,d,e,f=axes.flat
    panel(a,'Inertia and stiffness of the field','Dimensionless local field','Coefficient relatif au vide')
    panel(b,'Raideur selon la direction','Cutting position (native unit)','Raideur native')
    panel(c,'Internal time of the xenon bubble','Temps relatif au rayon minimal (ns)','Facteur de temps interne')
    for aa,name,items in [(a,'Champ',[('Electrical damage',green),('Magnetic stiffness',copper)]),(b,'Coupe 49',[('Longitudinale',blue),('Transverse',copper)]),(c,'Bath',[('4 Torr',green),('50 Torr · A',copper),('50 Torr · B',blue)])]:
        for label,color in items:
            vals=[v for v in native if v['plot']==name and v['series']==label and (aa!=c or -65<=float(v['x'])<=90)]
            series(aa,name,label,[float(v['x']) for v in vals],[float(v['y']) for v in vals],vals[0]['unit_x'],vals[0]['unit_y'],vals[0]['origin'],color)
    c.axvline(0,color='#9badaa',lw=.85,ls=':');c.set_xlim(-65,90);c.set_ylim(.06,1.08)
    for aa,loc in [(a,'upper right'),(b,'upper right'),(c,'lower right')]:aa.legend(loc=loc)
    panel(d,'Energy of the apical contact of SF5','Geometric deviation (native length)','Native energy')
    q=np.linspace(-.033,.018,306)+contacts['q_at_point']
    for family,label,color in [('two_preparations','16 k₀',green),('1995_branche4','28,51 k₀',copper)]:
        k=actions[family]['stiffness_over_k0']*contacts['k0']
        y=.5*k*(q-contacts['q_at_point']-contacts['lambda_action_native']/k)**2
        series(d,'Contact SF5',label,q,y,'longueur native','native energy','R571 and calibration R573',color)
    d.legend(loc='upper left');d.set_xlim(q[0],q[-1])
    for j,aa,mult in [(0,e,1000),(1,f,10000)]:
        group=f'Flux · face {j+1}';power=3 if j==0 else 4;u='10⁻³' if j==0 else '10⁻⁴'
        panel(aa,f'Énergie traversant la face {j+1}','Nodes by side',f'Énergie ({u} unité native)')
        ns=[v['N'] for v in flux['estimations']]
        for key,label,color,mark in [('sept_centers','Seven centres, gross value',copper,'o'),('estimation_meme_sonde','After propagation correction',green,'D')]:
            series(aa,group,label,ns,[v[key][j]*mult for v in flux['estimations']],'nœuds',f'10^(-{power}) énergie native','R573, same probe, t from 0 to 0.36',color,marker=mark,ms=4.7)
        aa.set_xticks(ns);aa.legend(loc='best',fontsize=8.7);aa.margins(y=.22)
    fig.text(.088,.022,'The xenon curves are calibrated simulations. The last two curves evaluate the same pulse at the same surfaces.',fontsize=9,color='#4b6668')
    fig.savefig(fig_dir/'COURBES_DU_SUBSTRAT_R573.png',dpi=175)
    plt.close(fig)
    with (root/'resultats/COURBES_DU_SUBSTRAT_R573.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
if __name__=='__main__':main()
