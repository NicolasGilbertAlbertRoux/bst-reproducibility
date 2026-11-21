'Export the Planck bands for the twelve families, by the original calculation R573.'
import argparse
import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(rows_field, destination, channel):
    families = sorted({x['family'] for x in rows_field})
    reference = [x for x in rows_field if x['family'] == "C00"]
    ell, obs, sigma, official = (np.array([x[k] for x in reference])
                                 for k in ["ell", "observed", "sigma", "official_binned_bestfit"])
    models = np.array([[x["predicted"] for x in rows_field if x['family'] == fam] for fam in families])
    res = (models - obs) / sigma
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#233544", "text.color": "#152b36"})
    fig, (ax, residual) = plt.subplots(2, 1, figsize=(9.2, 5.9), sharex=True,
                                    gridspec_kw={"height_ratios": [2.8, 1], "hspace": .13})
    for a in (ax, residual):
        a.grid(axis="y", alpha=.18)
        a.spines["left"].set_color("#82939a")
        a.spines["bottom"].set_color("#82939a")
    ax.fill_between(ell, models.min(axis=0), models.max(axis=0), color="#7fc6ba", alpha=.65,
                    label='Extent of 12 families')
    ax.plot(ell, models[0], color="#006d65", linewidth=2.0, label="Famille C00")
    ax.plot(ell, official, color="#b25736", linewidth=1.2, linestyle="--", label='Planck reference adjusted')
    ax.errorbar(ell, obs, yerr=sigma, fmt=".", color="#1e2c44", markersize=3.5, elinewidth=.7,
                capsize=0, label="Mesures Planck")
    ax.set_ylabel(r"$D_\ell$  ($\mu$K$^2$)")
    h,l = ax.get_legend_handles_labels()
    fig.legend(h,l,loc="upper left", bbox_to_anchor=(.1,.98), ncol=2,frameon=False,fontsize=10)
    lower = np.minimum(res.min(axis=0),0)
    upper = np.maximum(res.max(axis=0),0)
    residual.fill_between(ell, res.min(axis=0), res.max(axis=0), color="#7fc6ba", alpha=.7)
    residual.axhline(0, color="#667780", linewidth=.8)
    residual.axhline(1, color="#a3aeb0", linestyle=":", linewidth=.6)
    residual.axhline(-1, color="#a3aeb0", linestyle=":", linewidth=.6)
    residual.plot(ell, res[0], color="#006d65", marker=".", markersize=2.6, linewidth=.8)
    residual.set_ylim(min(-2,np.min(lower)-.4),max(2,np.max(upper)+.4))
    residual.set_ylabel('Gap / . . .')
    residual.set_xlabel("Multipôle ℓ")
    ax.set_xlim(20,2020)
    fig.subplots_adjust(top=.84,left=.11,bottom=.12,right=.98)
    fig.savefig(destination / f"PLANCK_{channel}.png", dpi=145, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("r573", type=Path, help='file containing code/ and entrees/')
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.r573.resolve()
    sys.path.insert(0,str(root/"code"))
    import oscillations_cosmology as original
    original_read = original.bin_spectrum
    comparative = json.loads((root/"resultats/COSMOLOGIE_ET_OSCILLATIONS_R573.json").read_text())
    p = original.read(original.I/"COSMO_R479/entrees/CANDIDATS_COSMOLOGIQUES_R479.json")
    growth = original.read(original.I/"COSMO_R479/resultats/CROISSANCE_RESERVEE_R479.json")
    rows = []
    deltas = []
    for param, target in zip(p,comparative['families'],strict=True):
        def collect(ell, puissance, channel):
            summary = original_read(ell, puissance, channel)
            for row in summary["rows"]:
                rows.append(dict(family=target['family'],channel=channel,**row))
            return summary
        with patch.object(original,"bin_spectrum",side_effect=collect):
            result = original.compute(param, target['family'], comparative["masse_eV"], growth)
        assert result["compatible"] == target["compatible"]
        for channel, measurement in result['chi2_cmb_par_band'].items():
            deltas.append(abs(measurement-target['chi2_cmb_par_band'][channel]))
        print(target['family'], flush=True)
    assert len(p)==12 and len(deltas)==36 and max(deltas)<1e-6, deltas
    with (args.output/"BANDES_PLANCK.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for channel in ["TT","TE","EE"]:
        draw([x for x in rows if x['channel']==channel],args.output,channel)
    check = {'families':len(p),'bands_family_channel':len(rows),'count_scores_verifies':len(deltas),
                'difference_absolu_max_chi2_par_band_contre_R573':max(deltas),"CAMB":original.camb.__version__,
                'scope':'Diagnostic strips 30 . Diagonal errors. Effective closure and transfer R573. The extent of the twelve curves is a range of models.'}
    (args.output/"CONTROLE_PROJECTIONS.json").write_text(json.dumps(check,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
