'Reconstitute the cosmological table of the manuscript with the distributed data.'
import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R573 = ROOT / "references" / "R573"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize():
    result = read_json(R573 / "resultats/COSMOLOGIE_ET_OSCILLATIONS_R573.json")
    parent = R573 / "entrees/COSMO_R479"
    growth = read_json(parent / "resultats/CROISSANCE_RESERVEE_R479.json")
    families = result['families']
    assert [v['family'] for v in families] == [f"C{i:02d}" for i in range(12)]
    obs = growth["observations"]
    values = np.array([o["value"] for o in obs])
    predictions = np.array([v["predictions_eBOSS"] for v in families])
    cov = np.array(growth["covariance"])
    assert cov.shape == (12, 12) and predictions.shape == (12, 12)
    np.linalg.cholesky(cov)
    residuals = predictions - values
    scores = np.sum(residuals * np.linalg.solve(cov, residuals.T).T, axis=1)
    np.testing.assert_allclose(scores, [v["chi2_eBOSS"] for v in families], rtol=1e-12)
    desi = np.loadtxt(next(parent.rglob("DESI_DR2_mean.txt")), usecols=0)
    desi_cov = np.loadtxt(next(parent.rglob("DESI_DR2_cov.txt")))
    assert desi_cov.shape == (len(desi), len(desi))
    with (ROOT / "cosmologie/BANDES_PLANCK.csv").open(encoding="utf-8-sig") as f:
        bands = list(csv.DictReader(f))

    def plage(values):
        return [float(min(values)), float(max(values))]

    planck = {}
    for channel in ("TT", "TE", "EE"):
        bst = []
        reference = []
        for family in families:
            rows = [b for b in bands if b['channel'] == channel and b['family'] == family['family']]
            assert len(rows) == 66
            for field, series in (("predicted", bst), ("official_binned_bestfit", reference)):
                series.append(float(np.mean([
                    ((float(b[field]) - float(b["observed"])) / float(b["sigma"]))**2 for b in rows
                ])))
        np.testing.assert_allclose(bst, [v['chi2_cmb_par_band'][channel] for v in families], atol=1e-11)
        np.testing.assert_allclose(reference, reference[0], atol=1e-12)
        planck[channel] = {"plage_BST": plage(bst), "reference_ajustee": reference[0]}
    assert len(bands) == 2376
    return {
        "Planck": planck,
        "DESI": {bloc: {"composantes": int(mask.sum()),
                        "plage_chi2_R573": plage([v["chi2_DESI"][bloc] for v in families])}
                 for bloc, mask in (("near", desi < 1), ("far", desi >= 1))},
        "eBOSS": {"composantes": len(obs), 'plage_chi2_recomputed': plage(scores),
                  'readouts': [{**o, "sigma_marginal": float(np.sqrt(cov[i, i])),
                                "plage_BST": plage(predictions[:, i])} for i, o in enumerate(obs)]},
        'thresholds_of_benchmark': result['thresholds_earlier'],
    }


def tableau():
    r = summarize()

    def fr(v):
        return f"{v:.3f}".replace(".", ",")

    def row(label, count, plage, threshold):
        return f"| {label} | {count} | {fr(plage[0])} à {fr(plage[1])} | {fr(threshold)} |"

    rows = ['- Comparison - Components - Extended score of the twelve families - Bench threshold -',
              "|:---|---:|---:|---:|"]
    for k, legend in (("near", "DESI, $z<1$"), ("far", r"DESI, $z\geq1$")):
        d = r["DESI"][k]
        rows.append(row(legend, d["composantes"], d["plage_chi2_R573"], r['thresholds_of_benchmark']["DESI_par_bloc"]))
    g = r["eBOSS"]
    rows.append(row("eBOSS, distances et croissance", g["composantes"],
                        g['plage_chi2_recomputed'], r['thresholds_of_benchmark']["eBOSS"]))
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='JSON file to be written')
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
