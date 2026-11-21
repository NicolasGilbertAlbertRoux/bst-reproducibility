'Replays the stored sources and compares the results with the archives.'
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
EXPECTED = ICI / 'expected'


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(ref, val, path):
    'Recursive comparison. Tolerances cover variations in digital libraries.'
    if isinstance(ref, dict) and isinstance(val, dict) and ref.keys() == val.keys():
        for key in ref:
            compare(ref[key], val[key], f"{path}/{key}")
    elif isinstance(ref, list) and isinstance(val, list) and len(ref) == len(val):
        for i, (a, b) in enumerate(zip(ref, val)):
            compare(a, b, f"{path}/{i}")
    elif type(ref) is float and type(val) in (float, int):
        if not math.isclose(ref, val, rel_tol=2e-6, abs_tol=2e-9):
            raise ValueError(f"{path} : référence {ref!r}, calcul {val!r}")
    elif type(ref) is not type(val) or ref != val:
        raise ValueError(f"{path} : résultat différent")


def replay(name_field, program, output):
    directory = output / name_field
    shutil.copytree(ICI / "sources" / name_field, directory)
    (directory / 'results').mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    with (directory / "journal.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, program], cwd=directory, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return directory


def check_sweep(recomputed):
    def rows(f):
        with f.open(encoding="utf-8", newline="") as stream:
            return list(csv.reader(stream))
    predicted = rows(EXPECTED / "BALAYAGE_COGNITIF_R528.csv")
    obtained = rows(recomputed)
    if len(predicted) != 145 or len(obtained) != len(predicted) or obtained[0] != predicted[0]:
        raise ValueError('Incomplete grid R528')
    for i, (a, b) in enumerate(zip(predicted[1:], obtained[1:]), 1):
        if len(a) != len(b) or len(b) != len(predicted[0]):
            raise ValueError(f"Colonnes de la ligne {i}")
        for j, (old, new_field) in enumerate(zip(a, b)):
            try:
                old, new_field = float(old), float(new_field)
            except ValueError:
                pass
            compare(old, new_field, f"R528/ligne{i}/{predicted[0][j]}")


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('The output directory must be empty.')
    historical = replay("A0_A9", "run_all_reproducibility.py", output)
    summary_a = read(historical / "REPRODUCTION_SUMMARY.json")
    if len(summary_a["stages"]) != 10 or not all(s["pass"] for s in summary_a["stages"]):
        raise ValueError('A historical audit failed.')
    count_tests = sum(int(re.search(r"Ran (\d+) tests?", "\n".join(s["test_log_tail"])).group(1))
                       for s in summary_a["stages"])
    references = sorted((ICI / "sources/A0_A9").glob("A*/*.json"))
    for file in references:
        copy = historical / file.relative_to(ICI / "sources/A0_A9")
        if copy.read_bytes() != file.read_bytes():
            raise ValueError(f"Différence dans {file.name}")
    if len(references) != 15 or count_tests != 97:
        raise ValueError("Corpus historique incomplet.")
    directories = {}
    for name_field, script, reference, result in [
        ("R173", "fenetres_latentes.py", "FENETRES_LATENTES_R173.json", "FENETRES_LATENTES.json"),
        ("R526", "cognition_flux_R526.py", "COGNITION_FLUX_ET_MEMOIRE_R526.json", "COGNITION_FLUX_ET_MEMOIRE_R526.json"),
        ("R528", "tester_persistance_R528.py", "PERSISTANCE_ET_CANAL_R528.json", "PERSISTANCE_ET_CANAL_R528.json"),
        ("R558", "code/test_cognition.py", "COGNITION_R558.json", "COGNITION.json"),
    ]:
        directory = replay(name_field, script, output)
        data = read(directory / 'results' / result)
        compare(read(EXPECTED / reference), data, name_field)
        directories[name_field] = data
    check_sweep(output / "R528/resultats/BALAYAGE_COGNITIF_R528.csv")
    subprocess.run([sys.executable, str(ICI / "figure_persistance.py"),
                    str(output / "R528/resultats"), str(output / "figure")], check=True)
    with (ICI / "figures/Paire_R528.csv").open(newline="") as ref, (output / "figure/Paire_R528.csv").open(newline="") as val:
        reference, new = list(csv.reader(ref)), list(csv.reader(val))
    if not (len(reference) == len(new) == 802 and reference[0] == new[0]):
        raise ValueError('Table of the two incomplete curves.')
    for i, (a, b) in enumerate(zip(reference[1:], new[1:]), 1):
        compare(list(map(float, a)), list(map(float, b)), f"Figure/instant{i}")
    trace = directories["R528"]
    summary = {'etapes_historical': len(summary_a["stages"]), 'tests_historical': count_tests,
             'json_historical_identiques_in_octets': len(references),
             'deuxieme_window_distingue_contenu': directories["R173"]["observability"]["window2_distinguishes_latent_payload"],
             'memory_of_phase': directories["R526"]["memory"],
             'contenus_bistables_read': trace["correct_analytic_decodings"], "trajectoires": trace["trajectories"],
             'comparisons_appariees': trace["paired_content_comparisons"], 'samples_figure_verifies': 801,
             'bound_energy': trace["analytic_interval_certificate"], "Horizon": directories["R558"]["Horizon"],
             "Crafter": directories["R558"]["Crafter"], 'origin': 'Historical programmes re-executed from the attachments.'}
    subprocess.run([sys.executable, str(ICI / "prolongement/rejouer.py"), str(output / "latent"),
                    "--reference", str(output / "R528/resultats/TRAJECTOIRES_COGNITIVES_R528.npz")], check=True)
    summary["prolongement_latent"] = read(output / "latent/BILAN_LATENT.json")
    (output / "BILAN_COGNITION.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Cognition vérifiée. Le détail est dans BILAN_COGNITION.json.")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
