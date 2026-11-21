'Replay the digital parts that accompany the progress list.'
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / "references"
STAGES = ("stage2_u1_flux_tube", "stage14b_faithful_contact_law", "stage14c_energy_conserving_beat",
          "stage25_phase_polarity", "stage76_colour_derivation_coleman_mandula", "stage79_scale_breaking_by_contact_closure")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_archive():
    expected = read_json(ROOT/"EMPREINTES.json")
    for name_field, digest in expected.items():
        if sha((ROOT/name_field).read_bytes()) != digest:
            raise ValueError(f"Contenu différent pour {name_field}")
    return len(expected)


def check_lean(destination):
    audit = REFERENCES/"BST_R251_RETOUR_FORMEL_AUDITE.zip"
    sources = REFERENCES/"BST_R251_VERIFICATION_FORMELLE_MAC.zip"
    journal = REFERENCES/"BST_R251_RETOUR_LEAN_20261103_174224_55181.zip"
    declarations = {}
    with ZipFile(audit) as za, ZipFile(sources) as zs, ZipFile(journal) as zj:
        a = json.loads(za.read(next(n for n in za.namelist() if n.endswith("AUDIT_RETOUR_FORMEL_R251.json"))))
        assert a["archive_sha256"] == sha(journal.read_bytes())
        r = json.loads(zj.read(next(n for n in zj.namelist() if n.endswith("RAPPORT_LEAN_R251.json"))))
        for f in r["source_integrity"]["files"]:
            match = next(n for n in zs.namelist() if n.endswith("/formalisation/"+f["path"]))
            assert sha(zs.read(match)) == f["sha256"]
        for n in zj.namelist():
            if not n.endswith(".log"):
                continue
            rows = zj.read(n).decode("utf-8")
            for m in re.finditer(r"'([^']+)' (does not depend on any axioms|depends on axioms:\s*\[([^\]]*)\])", rows):
                name_field, _, axioms = m.groups()
                value = " ".join((axioms or "").split())
                if name_field in declarations and declarations[name_field] != value:
                    raise ValueError(name_field)
                declarations[name_field] = value
        assert len(declarations) == 84 and sum(not x for x in declarations.values()) == 78
        assert not any("sorryAx" in v for v in declarations.values())
        assert len(r["steps"])==12 and all(s["returncode"]==0 for s in r["steps"])
        assert len(r["source_integrity"]["files"])==18
    with (destination/"DECLARATIONS_LEAN.csv").open("w",newline="",encoding="utf-8") as stream:
        w=csv.writer(stream); w.writerow(["declaration",'dependencies_logiques_in_the_journal'])
        w.writerows(sorted(declarations.items()))
    return {"sources":18,"declarations":len(declarations),'sans_axiome_list':78,'etapes_archived_reussies':12,
            "compilation_reference":"Mac de reference workstation, Lean 4.30.0, 3 novembre 2026"}


def check_gifs(output):
    from PIL import Image
    summary = read_json(output/"BILAN_GIFS.json")
    n = 0
    for input in summary["records"]:
        path = output/input["file"]
        assert sha(path.read_bytes())==input["gif_sha256"]
        with Image.open(path) as img:
            duration=0
            for i in range(img.n_frames):
                img.seek(i)
                img.load()
                duration += img.info["duration"]
            assert img.n_frames==input["encoded_frames"] and duration==7680 and img.size==(500,500)
            n += img.n_frames
    assert len(summary["records"])==13 and len(summary["species"])==5
    return {'files':13,'species':summary["species"],"images_effectivement_decodees":n,
            'images_echantillonnees_par_view':96,'duration_ms_par_view':7680}


def check_substrate(output):
    import numpy as np

    directory = ROOT / "redaction/donnees"
    name_field = "Horloges_R412.csv"
    with (directory/name_field).open(encoding="utf-8", newline="") as ref, (output/name_field).open(encoding="utf-8", newline="") as calc:
        expected_field, obtained = list(csv.DictReader(ref)), list(csv.DictReader(calc))
    assert len(obtained) == len(expected_field) == 36
    for row, expected in zip(obtained, expected_field, strict=True):
        assert row.keys() == expected.keys()
        for key in row:
            if key in ('check', 'family'):
                assert row[key] == expected[key]
            else:
                np.testing.assert_allclose(float(row[key]), float(expected[key]), rtol=1e-7, atol=5e-8)
    for family, target_battement, target_echo in (('common', 0.8, 0.8), ('control_scale', 0.6, 0.8),
                                                   ('control_gauge', 0.8, 0.6)):
        rows = [v for v in obtained if v['family'] == family and float(v["h"]) == 0.03125]
        assert len(rows) == 6
        for v in rows:
            assert abs(float(v['rate_battement']) - target_battement) < 0.0004
            assert abs(float(v['rate_echo']) - target_echo) < 0.0002
            assert abs(float(v['rate_echo_q']) - 0.8) < 0.0002
    local = read_json(output/"Reponse_locale_R412.json")
    np.testing.assert_allclose(local["inerties"], [1.25, 2, 2.25], atol=1e-12, rtol=0)
    for name_field, target in (('rest', -0.25), ('motion', 1.)):
        cases = local["preparations"][name_field]
        np.testing.assert_allclose([cases['force_scale'], cases['force_difference_centered']],
                                   [target, target], atol=2e-9, rtol=0)
    return {'readouts_clocks': len(obtained), "preparations_locales": len(local["preparations"])}


def check_battement(output):
    import numpy as np

    recomputed = read_json(output)
    reference = read_json(ROOT / "redaction/donnees/Battement_flux_strontium.json")
    cases = recomputed["battement"]["preparations"]
    assert cases.keys() == reference["battement"]["preparations"].keys()
    for name_field, values_field in cases.items():
        orig = reference["battement"]["preparations"][name_field]
        assert values_field.keys() == orig.keys()
        for field, value in values_field.items():
            np.testing.assert_allclose(value, orig[field], atol=2e-11, rtol=1e-8)
    flux = recomputed['current_field']['flux_mean_par_count_wave']
    np.testing.assert_allclose([flux[k] for k in ("-0.75", "0.0", "0.75")],
                               [-0.0046875, 0., 0.0046875], rtol=0, atol=1e-14)
    acts = recomputed['current_acts']
    assert acts == reference['current_acts']
    sr = recomputed["strontium"]
    assert sr['frequency_indirecte_Hz'] == 429228004340000
    assert sr['difference_indirect_moins_BIPM_Hz'] == "110127.008"
    for field, value in reference["strontium"].items():
        if isinstance(value, (int, float)):
            np.testing.assert_allclose(sr[field], value, atol=0, rtol=1e-12)
    return {"preparations": len(cases), 'currents_of_field': len(flux),
            'acts_publics': len(acts["trace_historique"]),
            'classes_of_source': len(acts['classes_of_source']),
            'frequency_indirecte_Hz': sr['frequency_indirecte_Hz'],
            'file': output.name}


def compare_readouts(obtained, expected, field=""):
    if isinstance(expected, dict):
        assert expected.keys() == obtained.keys(), field
        for key, value in expected.items():
            compare_readouts(obtained[key], value, field + "/" + key)
    elif isinstance(expected, list):
        for value, target in zip(obtained, expected, strict=True):
            compare_readouts(value, target, field)
    elif isinstance(expected, float):
        residuals_rounded = {'defect_transverse_normalized', 'defect_puissance_normalized',
                           'bound_aires_at_square', 'deviation_max_densities'}
        absolute = 1e-13 if field.rsplit("/", 1)[-1] in residuals_rounded else 0
        assert math.isclose(obtained, expected, rel_tol=3e-8, abs_tol=absolute), field
    else:
        assert obtained == expected, field


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sortie",type=Path,required=True,help='new job directory, off-file delivered')
    parser.add_argument("--choix",choices=("tout","archives","premiers-essais","animations",'cosmology',"cognition","latent","ouverture","reduction",'substrate',"battement","d01","domaines","physique-rapide"),default="tout")
    args=parser.parse_args()
    out=args.output.resolve()
    if out==ROOT or ROOT in out.parents:
        parser.error('Choose an output directory delivered.')
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        parser.error('Choose an empty output directory.')
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    env["PATH"]=str(Path(sys.executable).parent)+os.pathsep+env.get("PATH","")
    summary={'files_archives_verifies':verify_archive(),'choice':args.choice}
    with TemporaryDirectory() as rep:
        temp=Path(rep)
        if args.choice in ("tout", "domaines"):
            dest = out / "domaines"
            subprocess.run([sys.executable, str(ROOT/"redaction/relire_domaines.py"), str(dest)],
                           cwd=temp, env=env, check=True)
            directory_ref = ROOT/"redaction/donnees"
            sample = read_json(dest/"Relecture_des_domaines.json")
            compare_readouts(sample, read_json(directory_ref/"Relecture_des_domaines.json"))
            map = "Parcours_des_29_clauses.csv"
            assert (dest/map).read_bytes() == (directory_ref/map).read_bytes()
            from PIL import Image
            with Image.open(dest/"Marge_de_lecture_H528.png") as image:
                image.verify()
            summary["domaines"] = {"domaines": len(sample['order']), "clauses": len(sample["clauses_directes"]),
                                 'records': sample['records_distinct_of_register'],
                                 "amplitudes_R446": sample["D02"]["amplitudes"],
                                 'pairs_H528': sample["D10"]['pairs'], "sites_phase": sample["phase"]["sites"]}
        if args.choice in ("tout", "d01"):
            import numpy as np
            dest = out / "d01/Relecture_D01.json"
            subprocess.run([sys.executable, str(ROOT/"redaction/relire_d01.py"), str(dest)],
                           cwd=temp, env=env, check=True)
            d01 = read_json(dest)
            ref = read_json(ROOT/"redaction/donnees/Relecture_D01.json")
            assert d01["D01"] == ref["D01"]
            np.testing.assert_allclose(d01["Sr"]['values_eigen_positives'],
                                       ref["Sr"]['values_eigen_positives'], atol=2e-13, rtol=0)
            dynamics = d01["Sr87"]['two_fields']
            np.testing.assert_allclose(dynamics['angular_frequencies_basses_and_hautes'],
                                       ref["Sr87"]['two_fields']['angular_frequencies_basses_and_hautes'],
                                       atol=2e-13, rtol=0)
            assert d01['collection']['operators'] == 82 and dynamics['pairs'] == 561
            assert abs(dynamics['difference_minimum_absolu_Hz']-943017312660) < 10
            assert abs(d01["terme_electronique_SrI"]['deviation_sign_cm_1']+10.2785) < 1e-8
            from PIL import Image
            with Image.open(dest.with_name("Modes_du_strontium.png")) as figura:
                assert figura.size == (1600, 560)
                figura.verify()
            summary["d01"] = {'operators': 82, "modes_positifs_Sr": len(d01["Sr"]['values_eigen_positives']),
                            'pairs_two_fields': dynamics['pairs'],
                            'difference_minimum_absolu_Hz': dynamics['difference_minimum_absolu_Hz']}
        if args.choice in ("tout", "battement"):
            dest = out / "Battement_flux_strontium.json"
            subprocess.run([sys.executable, str(ROOT / "redaction/battement_flux_strontium.py"),
                            str(dest)], cwd=temp, env=env, check=True)
            summary["battement_flux_strontium"] = check_battement(dest)
        if args.choice in ("tout", 'substrate'):
            dest = out / 'substrate'
            subprocess.run([sys.executable, str(ROOT/"redaction/eprouver_substrat.py"), str(dest)],
                           cwd=temp, env=env, check=True)
            summary['substrate'] = check_substrate(dest)
            subprocess.run([sys.executable, str(ROOT/"redaction/rejouer_horloges_R411.py"), str(dest)],
                           cwd=temp, env=env, check=True)
            summary['substrate']["R411"] = read_json(dest/"Rejeu_horloges_R411.json")['summary']
        if args.choice in ("tout","reduction"):
            import numpy as np
            dest = out / "Comparaison_S14c.json"
            subprocess.run([sys.executable, str(ROOT / "redaction/tester_dynamique_reduite.py"), str(dest)],
                           cwd=temp, env=env, check=True)
            ref = read_json(ROOT / "redaction/donnees/Comparaison_S14c.json")["series"]
            for obtained, expected in zip(read_json(dest)["series"], ref, strict=True):
                assert (obtained['stiffness_contact'], obtained["dt"]) == (expected['stiffness_contact'], expected["dt"])
                for direction in ("entrant", "sortant"):
                    for field, value in expected['trials'][direction].items():
                        np.testing.assert_allclose(obtained['trials'][direction][field], value, rtol=1e-8, atol=2e-10)
            summary["reduction"] = {'series_contact_and_step': len(ref), 'file': dest.name}
        if args.choice in ("tout","ouverture"):
            import numpy as np
            where=out/"ouverture"
            subprocess.run([sys.executable,str(ROOT/"redaction/exemple_battement.py"),str(where)],
                           cwd=temp,env=env,check=True)
            name_field="Lecture_deux_battements.csv"
            ref=np.loadtxt(ROOT/"redaction/donnees"/name_field,delimiter=",",skiprows=1)
            calculation=np.loadtxt(where/name_field,delimiter=",",skiprows=1)
            np.testing.assert_allclose(calculation,ref,rtol=1e-8,atol=2e-10)
            summary["ouverture"]=read_json(where/"Lecture_deux_battements.json")
            name_field = "Dispersion_R412.csv"
            subprocess.run([sys.executable, str(ROOT/"redaction/controler_dispersion.py"), str(where/name_field)],
                           cwd=temp, env=env, check=True)
            ref = np.loadtxt(ROOT/"redaction/donnees"/name_field, delimiter=",", skiprows=1)
            calculation = np.loadtxt(where/name_field, delimiter=",", skiprows=1)
            np.testing.assert_allclose(calculation, ref, rtol=1e-10, atol=2e-12)
            summary["dispersion_R412"] = {'grids': len(calculation), "directions": 2, 'file': name_field}
        if args.choice in ("tout","physique-rapide"):
            with ZipFile(REFERENCES/"BST_R573_AUDIT_ET_COURBES.zip") as z:
                z.extractall(temp)
            base=temp/"BST_R573"
            with (out/"PHYSIQUE_R573.log").open("w") as log:
                subprocess.run(["bash","REJOUER.command","--rapide"],cwd=base,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            reference=read_json(REFERENCES/"R573/resultats/REPRODUCTION_R573.json")['files']
            differences=[n for n,v in reference.items() if sha((base/n).read_bytes())!=v]
            if differences:
                raise ValueError('Outputs R573 different: '+", ".join(differences))
            summary['physics_rapide']={'outputs_identiques':len(reference)}
        if args.choice in ("tout","archives"):
            summary["lean"]=check_lean(out)
        if args.choice in ("tout","premiers-essais"):
            where=out/'initial_trials'; where.mkdir()
            with ZipFile(REFERENCES/"beating-substrate-framework.zip") as z:
                for stage in STAGES:
                    file=temp/(stage+".py")
                    file.write_bytes(z.read("beating-substrate-framework/scripts/"+file.name))
                    res=subprocess.run([sys.executable,str(file)],cwd=temp,env=env,check=True,stdout=subprocess.PIPE)
                    (where/(stage+".log")).write_bytes(res.stdout)
            summary['initial_trials']=len(STAGES)
            check_field = where / "Controle_plaquette_S2.json"
            subprocess.run([sys.executable, str(ROOT / "redaction/controler_plaquette.py"), str(check_field)],
                           cwd=temp, env=env, check=True)
            summary["plaquettes_S2"] = read_json(check_field)
        if args.choice in ("tout","animations"):
            with ZipFile(REFERENCES/"BST_R178_GIFS_MAC.zip") as z:
                z.extractall(temp)
            source=temp/"BST_GIFS_MAC"
            subprocess.run([sys.executable,"-m","unittest","discover","-s","tests"],cwd=source,env=env,check=True)
            subprocess.run([sys.executable,"generer_gifs.py","--sortie",str(out/"animations")],cwd=source,env=env,check=True)
            summary["gifs"]=check_gifs(out/"animations")
        if args.choice == "latent":
            subprocess.run([sys.executable, str(ROOT/"cognition/prolongement/rejouer.py"),
                            str(out/"latent")], cwd=temp, env=env, check=True)
            summary["latent"] = read_json(out/"latent/BILAN_LATENT.json")
        if args.choice in ("tout","cognition"):
            subprocess.run([sys.executable, str(ROOT/"cognition/rejouer_cognition.py"),
                            str(out/"cognition")], cwd=temp, env=env, check=True)
            summary["cognition"] = read_json(out/"cognition/BILAN_COGNITION.json")
        if args.choice in ("tout",'cosmology'):
            subprocess.run([sys.executable,str(ROOT/"calculs/projections.py"),str(REFERENCES/"R573"),
                            str(out/'cosmology')],cwd=temp,env=env,check=True)
            summary['cosmology']=read_json(out/"cosmologie/CONTROLE_PROJECTIONS.json")
    write_json(out/"BILAN.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
