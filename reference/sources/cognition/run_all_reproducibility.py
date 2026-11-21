#!/usr/bin/env python3
'Run every BST–COG A0–A9 test and regenerate deterministic result files.'

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGES = (
    (
        "A0",
        "A0_actualization",
        "test_bst_hidden_asymmetry_audit.py",
        ["bst_hidden_asymmetry_audit.py", "--max-n", "7", "--alphabet-size", "2"],
        "bst_cog_a0_results.json",
    ),
    (
        "A1",
        "A1_functional_arbiter",
        "test_bst_cog_a1_arbiter_models.py",
        ["bst_cog_a1_arbiter_models.py"],
        "bst_cog_a1_results.json",
    ),
    (
        "A2",
        "A2_authorship_criterion",
        "test_bst_cog_a2_authorship_criterion.py",
        ["bst_cog_a2_authorship_criterion.py"],
        "bst_cog_a2_results.json",
    ),
    (
        "A3",
        "A3_biface_authorship",
        "test_bst_cog_a3_biface_authorship.py",
        ["bst_cog_a3_biface_authorship.py", "--output", "bst_cog_a3_results.json"],
        None,
    ),
    (
        "A4",
        "A4_meta_revision_origin",
        "test_bst_cog_a4_meta_revision_origin.py",
        ["bst_cog_a4_meta_revision_origin.py", "--output", "bst_cog_a4_results.json"],
        None,
    ),
    (
        "A5",
        "A5_consolidation",
        "test_bst_cog_a5_theorem_stack_protocol.py",
        [
            "bst_cog_a5_theorem_stack_protocol.py",
            "--output",
            "bst_cog_a5_results.json",
            "--protocol-output",
            "bst_cog_a5_biface_protocol.json",
        ],
        None,
    ),
    (
        "A6",
        "A6_empirical_protocol",
        "test_bst_cog_a6_empirical_protocol.py",
        [
            "bst_cog_a6_empirical_protocol.py",
            "--output",
            "bst_cog_a6_results.json",
            "--design-output",
            "bst_cog_a6_design_contract.json",
        ],
        None,
    ),
    (
        "A7/A7-R1",
        "A7_instruments_pipeline",
        "test_bst_cog_a7_instrument_battery.py",
        ["bst_cog_a7_instrument_battery.py", "--output-dir", "."],
        None,
    ),
    (
        "A8",
        "A8_identity_internalization_cogito",
        "test_bst_cog_a8_identity_internalization_cogito.py",
        [
            "bst_cog_a8_identity_internalization_cogito.py",
            "--output",
            "bst_cog_a8_results.json",
        ],
        None,
    ),
    (
        "A9",
        "A9_sealed_vessel_source_attribution",
        "test_bst_cog_a9_sealed_vessel_source_attribution.py",
        [
            "bst_cog_a9_sealed_vessel_source_attribution.py",
            "--output",
            "bst_cog_a9_results.json",
        ],
        None,
    ),
)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def write_manifest() -> int:
    manifest = ROOT / "MANIFEST_SHA256_COMPLETE.txt"
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    summary = {
        "package": "BST_COG_COMPLETE_REPRODUCIBLE_A0_A9_R0",
        "python": sys.version.split()[0],
        "human_data_present": False,
        "V94_imported_or_modified": False,
        "V94_status_preserved": "C2_OPEN__C3_CLOSED",
        "stages": [],
    }
    all_pass = True
    for stage, folder, test_file, result_command, capture_name in STAGES:
        cwd = ROOT / folder
        test = run([sys.executable, "-m", "unittest", "-v", test_file], cwd)
        result = run([sys.executable, *result_command], cwd)
        if capture_name is not None and result.returncode == 0:
            # A0–A2 print their canonical JSON to stdout.
            payload = json.loads(result.stdout)
            (cwd / capture_name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        passed = test.returncode == 0 and result.returncode == 0
        all_pass = all_pass and passed
        summary["stages"].append(
            {
                "stage": stage,
                "directory": folder,
                "tests_pass": test.returncode == 0,
                "result_generation_pass": result.returncode == 0,
                "pass": passed,
                "test_log_tail": test.stdout.strip().splitlines()[-4:],
            }
        )
        print(f"{stage}: {'PASS' if passed else 'FAIL'}")

    summary["pass"] = all_pass
    summary["verdict"] = (
        "A0_A9_FULL_INTERNAL_REPRODUCIBILITY_PASS"
        if all_pass else "A0_A9_REPRODUCIBILITY_FAIL"
    )
    (ROOT / "REPRODUCTION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    count = write_manifest()
    print(f"manifest_files: {count}")
    print(summary["verdict"])
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
