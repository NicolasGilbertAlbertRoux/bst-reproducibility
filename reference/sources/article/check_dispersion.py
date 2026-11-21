'Compare two directions of equation (15) of R412 at fixed wavelength.'
import argparse
import csv
import math
from pathlib import Path


def speed_of_phase(mode, step):
    'Apply the Laplacian to a maximum of six neighbors of a Fourier cosine.'
    norm = math.hypot(*mode)
    laplacian = sum(2 * (math.cos(component * step) - 1) for component in mode) / step**2
    # In the units of the calculation, c' is equal to 1, and therefore -laplacean - pulsation squared.
    pulsation = math.sqrt(-laplacian)
    symbol = 2 * math.hypot(*(math.sin(k * step / 2) for k in mode)) / step
    if not math.isclose(pulsation, symbol, rel_tol=2e-12):
        raise ValueError("Le laplacien et son symbole de Fourier divergent.")
    return pulsation / norm


def main(output):
    axis, oblique = (3, 0, 0), (2, 2, 1)
    assert math.hypot(*axis) == math.hypot(*oblique) == 3
    last_difference = None
    with output.open("w", newline="", encoding="utf-8") as stream:
        table = csv.writer(stream, lineterminator="\n")
        table.writerow(("sites_par_cote", 'step', 'v_phase_axis_over_c_star',
                        'v_phase_oblique_over_c_star', 'deviation_over_c_star'))
        for sites in (16, 32, 64, 128):
            step = math.tau / sites
            va, vo = (speed_of_phase(mode, step) for mode in (axis, oblique))
            difference = vo - va
            if not (0 < va < vo < 1):
                raise ValueError('Unexpected order of phase speeds.')
            if last_difference is not None and not 3.8 < last_difference / difference < 4.1:
                raise ValueError('The refinement does not return to the expected order two.')
            table.writerow((sites, *(format(v, ".12g") for v in (step, va, vo, difference))))
            last_difference = difference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='CSV file to be written')
    main(parser.parse_args().output)
