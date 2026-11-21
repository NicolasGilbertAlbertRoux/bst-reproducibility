'Exports the same R528 trajectory pair in CSV, SVG and autonomous LaTeX drawing.'
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def produce(results, output):
    output.mkdir(parents=True, exist_ok=True)
    with (results / "BALAYAGE_COGNITIF_R528.csv").open(encoding="utf-8") as s:
        cases = list(csv.DictReader(s))
    with np.load(results / "TRAJECTOIRES_COGNITIVES_R528.npz") as data_field:
        if list(data_field["columns"])[:3] != ["t", "x", "q"]:
            raise ValueError("Ordre de colonnes inconnu.")
        series = data_field["records"]
    indices = [i for i,c in enumerate(cases) if c["tag"] == "m1" and
               all(float(c[k]) == v for k,v in [("eta",.5),("lam",.5),("mass",.5),("impulse",.2)])]
    if [int(cases[i]["sign"]) for i in indices] != [-1,1]:
        raise ValueError("Paire de signes introuvable.")
    left, right = (series[:, i, :] for i in indices)
    if not np.array_equal(left[:,0], right[:,0]):
        raise ValueError('The instants of the two trajectories differ.')
    time = left[:,0]
    with (output / "Paire_R528.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["t",'q_minus',"q_plus","x_sur_J_moins","x_sur_J_plus"])
        w.writerows(zip(time,left[:,2],right[:,2],left[:,1]/.2,right[:,1]/.2))
    colors=["#126666","#9d4828"]
    traits=["-","--"]
    fig,axs=plt.subplots(2,1,figsize=(7.2,4.8),sharex=True,layout="constrained")
    for axis, col, lim, titre in zip(axs,[2,1],[(-1.25,1.25),(-.45,.45)],
                                   ['Trace carried by the system q',"Réponse au même signal  x/J"]):
        axis.set_title(titre,loc="left",fontsize=11,weight="bold")
        for i,data in enumerate([left,right]):
            axis.plot(time,data[:,col]/(1 if col==2 else .2),traits[i],color=colors[i],
                     lw=1.9,label=['Trace prepared at 1','Trace prepared at 1'][i])
        axis.set(xlim=(0,4),ylim=lim)
        axis.set_yticks([-1,0,1] if col==2 else [-.4,0,.4],["−1","0","+1"] if col==2 else ["−0,4","0","+0,4"])
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y",alpha=.17)
    axs[0].legend(loc="center",fontsize=9,ncol=2,frameon=False)
    axs[1].set(xlabel='Time (reduced units)',xticks=range(5))
    matplotlib.rcParams["svg.hashsalt"]="bst-cognition-r528"
    fig.savefig(output/"Figure_persistance_R528.svg",metadata={"Date":None},format="svg")
    plt.close(fig)
    # One point out of two, up to t plus 4, for the vector drawing integrated into the manuscript.
    view=(time<=4)&(np.arange(len(time))%2==0)
    width=10.6
    tex=[r"\begingroup",r"\definecolor{bstTraceMoins}{HTML}{126666}",
         r"\definecolor{bstTracePlus}{HTML}{9D4828}",
         r"\begin{center}\begin{tikzpicture}[x=1cm,y=1cm,font=\footnotesize]"]
    for col,low,high,ext,titre in [(2,3.9,6.15,1.25,"Trace portée par le système $q$"),
                                   (1,.75,3.0,.45,"Réponse au même signal $x/J$")]:
        def ycoord(v):
            return low+(high-low)*(v+ext)/(2*ext)
        tex.append(r"\node[anchor=west,font=\small\bfseries] at (0,"+f"{high+.28:.3f}"+"){"+titre+"};")
        for tick,label in ([(-1,"$-1$"),(0,"0"),(1,"$+1$")] if col==2 else [(-.4,"$-0{,}4$"),(0,"0"),(.4,"$+0{,}4$")]):
            y=ycoord(tick)
            tex.append(rf"\draw[black!18] (0,{y:.3f}) -- ({width},{y:.3f}); \node[anchor=east] at (0,{y:.3f}) {{{label}}};")
        for i,data in enumerate([left,right]):
            yy=data[view,col]/(1 if col==2 else .2)
            xy=" ".join(f"({t*width/4:.4f},{ycoord(y):.4f})" for t,y in zip(time[view],yy))
            sty=["bstTraceMoins,thick","bstTracePlus,thick,dashed"][i]
            tex.append(rf"\draw[{sty}] plot coordinates {{{xy}}};")
    for i in range(5):
        tex.append(rf"\node at ({i*width/4:.3f},.48) {{{i}}};")
    tex.extend([r"\node at (5.3,.02) {Temps (unités réduites)};",
                r"\draw[bstTraceMoins,thick] (1.5,5.02)--(2.3,5.02); \node[anchor=west] at (2.3,5.02) {Trace préparée à $-1$};",
                r"\draw[bstTracePlus,thick,dashed] (6.15,5.02)--(6.95,5.02); \node[anchor=west] at (6.95,5.02) {Trace préparée à $+1$};",
                r"\end{tikzpicture}\end{center}\endgroup"])
    (output/"Courbes_R528.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")


if __name__ == "__main__":
    produce(Path(sys.argv[1]),Path(sys.argv[2]))
