'Recount the R573 bench and the diagnoses of the Planck strips of the paper.'
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from zipfile import ZipFile

DIRECTORY = Path(__file__).resolve().parents[1]
REPOSITORY = DIRECTORY / "references" / "BST_R573_AUDIT_ET_COURBES.zip"
TRAJECTORY = "BST_R573/resultats/MOUVEMENT_N72_impedance_dt0.09.json"

def score(bands, column):
    return statistics.fmean(
        ((float(b[column]) - float(b["observed"])) / float(b["sigma"])) ** 2
        for b in bands
    )

def check(trajectory):
    with ZipFile(REPOSITORY) as archive:
        matrix = json.loads(archive.read("BST_R573/resultats/ASSEMBLAGES_68_R573.json"))
        register = json.loads(archive.read("BST_R573/REGISTRE_68_LIGNES_R573.json"))
        reference = archive.read(TRAJECTORY)
    clauses = matrix['order_of_clauses']
    assert [row['row'] for row in register] == clauses
    assert len(set(clauses)) == len(clauses) == 68
    index_joint = clauses.index("J00.1")
    combinations = matrix['combinations']
    codes_established = {"C", "H", "D"}
    passes = sum(all(val in codes_established for i, val in enumerate(combination["codes"]) if i != index_joint)
                 for combination in combinations)
    integers = sum(set(combination["codes"]) <= codes_established for combination in combinations)
    assert (len(combinations), passes, integers) == (576, 192, 0)
    with (DIRECTORY / 'cosmology' / "BANDES_PLANCK.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    families = sorted({row['family'] for row in rows})
    assert len(families) == 12 and len(rows) == 2376
    scores = {}
    for channel in ("TT", "TE", "EE"):
        series = [[row for row in rows if row['family'] == fam and row['channel'] == channel]
                  for fam in families]
        assert all(len(group) == 66 for group in series)
        def controls(band):
            return tuple(band[k] for k in ("ell", "observed", "sigma", "official_binned_bestfit"))
        ref_bands = list(map(controls, series[0]))
        assert all(list(map(controls, group)) == ref_bands for group in series)
        values = [score(group, "predicted") for group in series]
        control = score(series[0], "official_binned_bestfit")
        assert control < min(values) <= max(values) < 2
        scores[channel] = {"min_BST": min(values), "max_BST": max(values), "reference_Planck": control}
    res = {"clauses": len(clauses), 'combinations': len(combinations), 'covering_67_clauses': passes,
           'covering_68_sans_reserve_of_correspondence': integers, "scores_diagonaux": scores,
           "sha256_reference_sf6": hashlib.sha256(reference).hexdigest()}
    if trajectory:
        current = trajectory.read_bytes()
        if current != reference:
            raise ValueError('The trajectory provided differs from the reference bytes.')
        res['sf6_recomputed_identical'] = True
        res['energy_nette_two_surfaces_units_natives'] = json.loads(current)['energy_nette_two_surfaces']
    return res

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectoire", type=Path, help='new result of the SF6 calculation on 72 nodes per side')
    print(json.dumps(check(p.parse_args().trajectory), ensure_ascii=False, indent=2))
