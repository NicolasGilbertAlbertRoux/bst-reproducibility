"""Build screen readouts for measured wavelengths and specified optical transmission cells."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

INPUTS = Path(__file__).resolve().parent / 'inputs' / 'optics'
XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -.4985314],
                        [-.9692660, 1.8760108, .0415560],
                        [.0556434, -.2040259, 1.0572252]])
PATHS_M = (0, .01, .1, 1, 5, 10)
VISIBLE_LINES_NM = (473.7, 516.5, 563.5, 557.02894, 587.09160)


@lru_cache(maxsize=1)
def observer():
    return np.loadtxt(INPUTS / 'daylight-observer.csv', delimiter=',', skiprows=1)


def srgb(linear):
    v = np.clip(linear, 0, 1)
    encoded = np.where(v <= .0031308, 12.92*v, 1.055*v**(1/2.4)-.055)
    return tuple(int(x) for x in np.rint(255*encoded))


@lru_cache(maxsize=None)
def spectral_colour(wavelength):
    """Relative screen hue of a narrow filter; clipping and normalisation are display choices."""
    data = observer()
    if not data[0, 0] <= wavelength <= data[-1, 0]:
        raise ValueError('Wavelength outside the tabulated display window')
    xyz = [np.interp(wavelength, data[:, 0], column) for column in data[:, 1:4].T]
    linear = np.maximum(XYZ_TO_SRGB @ xyz, 0)
    return srgb(linear/linear.max())


def absorption(material, nm):
    if material == 'water':
        wave, value, _ = np.loadtxt(INPUTS/'water-absorption.csv', delimiter=',', skiprows=1).T
        return np.interp(nm, wave, value)
    if material == 'ice':
        wave_um, k = np.loadtxt(INPUTS/'ice-extinction.csv', delimiter=',', skiprows=1).T
        # Warren and Brandt prescribe log(k) interpolation against log(wavelength).
        extinction = np.exp(np.interp(np.log(nm/1000), np.log(wave_um), np.log(k)))
        return 4*np.pi*extinction/(nm*1e-9)
    raise ValueError('Unknown optical cell')


IODINE_NUMBER_DENSITY_CM3 = 5e15


@lru_cache(maxsize=1)
def iodine_grid():
    # Resolve the structured (0.1 to 0.2 nm) part of the measured spectrum before
    # integrating colour. The original source has wider spacing in its wings.
    grid = np.linspace(380, 700, 16001)
    reference = observer()
    data = np.column_stack([grid] + [
        np.interp(grid, reference[:, 0], column) for column in reference[:, 1:].T
    ])
    wavelengths, cross_section = np.loadtxt(
        INPUTS / 'iodine-cross-section.csv', delimiter=',', skiprows=1).T
    sigma = np.interp(grid, wavelengths, cross_section)
    return data, sigma


def transmission(material, distance_m):
    if not np.isfinite(distance_m) or distance_m < 0:
        raise ValueError('Optical paths must be finite and nonnegative')
    if material == 'iodine':
        data, sigma = iodine_grid()
        column_cm2 = IODINE_NUMBER_DENSITY_CM3 * 100 * distance_m
        trans = np.exp(-sigma * column_cm2)
        # The plotted curve is averaged over a 1 nm window at each integer nm.
        kernel = np.ones(51)
        smoothed = np.convolve(trans, kernel, mode='same') / np.convolve(
            np.ones_like(trans), kernel, mode='same')
        curve = smoothed[::50]
    else:
        data = observer()
        trans = np.exp(-absorption(material, data[:, 0])*distance_m)
        curve = trans[::5]
    # CIE 2 degree observer under D65; included band 380–700 nm, with trapezoidal end weights.
    weight = data[:, 4].copy()
    weight[[0, -1]] *= .5
    weighted_xyz = data[:, 1:4]*weight[:, None]
    white, transmitted = weighted_xyz.sum(axis=0), trans @ weighted_xyz
    # White-balance the incident light in the same finite band; hold the exposure fixed over paths.
    rgb = srgb((XYZ_TO_SRGB @ transmitted)/(XYZ_TO_SRGB @ white))
    return {'path_m': distance_m, 'rgb': rgb, 'relative_Y': round(transmitted[1]/white[1], 8),
            'transmission': curve.round(8).tolist()}


def build(output):
    records = {m: [transmission(m, d) for d in PATHS_M] for m in ('water', 'ice', 'iodine')}
    records['iodine_number_density_cm3'] = IODINE_NUMBER_DENSITY_CM3
    records['lines'] = {str(w): spectral_colour(w) for w in VISIBLE_LINES_NM}
    output.write_text('window.BST_OPTICS = '+json.dumps(records,separators=(',',':'),allow_nan=False)+';\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    build(parser.parse_args().output)
