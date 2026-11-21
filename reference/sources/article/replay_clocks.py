'Recalculate the radial frequencies and the twelve transverse trajectories of R411.'

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    heritage = source / "heritage_R411"
    reference = json.loads((source / "donnees/Horloges_reference_R411.json").read_text())
    for name, digest in reference['files_originaux'].items():
        if hashlib.sha256((heritage / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source R411 modifiée : {name}")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(heritage))
    import numpy as np
    from benchmark_observables import modes_rescales, test_waves
    from dynamics import compile_kernel

    profiles = np.loadtxt(heritage / "PROFILS_RADIAUX_R410.csv", delimiter=",", skiprows=1)
    modes = modes_rescales(profiles)
    with TemporaryDirectory() as temp:
        kernel = Path(temp) / "liens_R411.so"
        compile_kernel(heritage / "liens_jauge_R411.c", kernel)
        waves = [test_waves(kernel, law, s0, step) for law in ("R410", 'common')
                 for s0 in (-0.12, 0.0, 0.12) for step in (100, 200)]

    for group, measurements, keys in (
        ("modes", modes, ('degree', 'law', "elements", "delta_s")),
        ('waves', waves, ('law', "s_uniforme", 'step')),
    ):
        for calc, archive in zip(measurements, reference[group], strict=True):
            assert calc.keys() == archive.keys()
            assert all(calc[c] == archive[c] for c in keys)
            for c in calc.keys() - set(keys):
                np.testing.assert_allclose(calc[c], archive[c], rtol=5e-8, atol=5e-10)

    error_modes = max(abs(r - m['ratio_expected']) for m in modes for r in m['ratio_computed'])
    error_waves = max(m['error_relative'] for m in waves if m['step'] == 200)
    assert len(modes) == 24 and error_modes < 2e-8
    assert len(waves) == 12 and error_waves < 1.35e-7
    summary = {"preparations_radiales": len(modes), "modes_par_preparation": 2,
             "trajectoires_transverses": len(waves), 'deviation_max_ratio_radial': error_modes,
             'deviation_relative_max_transverse_step_fin': error_waves}
    args.output.mkdir(exist_ok=True, parents=True)
    results = {'summary': summary, "modes": modes, 'waves': waves}
    (args.output / "Rejeu_horloges_R411.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
