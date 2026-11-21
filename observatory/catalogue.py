"""Export the shipped atlas or test a measured neon interval against frozen bounds."""
from pathlib import Path
import csv, json, math
HERE=Path(__file__).resolve().parent
def data():
    raw=(HERE/'data.js').read_text()
    return json.loads(raw.removeprefix('window.BST_DATA = ').removesuffix(';\n'))
def export(out,element):
    d=data();out.mkdir(parents=True,exist_ok=True)
    matched=[row for row in d['elements'] if element is None or row[1].lower()==element.lower()]
    if not matched:raise ValueError(f'Unknown element: {element}')
    numbers={r[0] for r in matched}
    tables={'elements':('Z symbol period archived_neutral_count candidate_neutral_count surface_nodes positive_modes omega_min omega_max',matched),'isotopes':('Z A state stable half_life_seconds half_life_estimated mass_estimated physical_wave_count mass_neutral_u spin_parity excitation_keV',[r for r in d['isotopes'] if r[0] in numbers]),'ions':('Z charge electron_count cumulative_ionization_eV physical_wave_count',[r for r in d['ions'] if r[0] in numbers])}
    for name,(columns,rows) in tables.items():
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(columns.split());writer.writerows(rows)
    return {name:len(rows) for name,(_,rows) in tables.items()}
def neon(pressure,interval):
    k=f'{pressure:g}';d=data()['neon']
    if k not in d['aire_par_sommet']:raise ValueError('Pressure must be one of 1.9, 2.1, 2.4, 2.7 or 3.1 bar')
    if interval is not None and (not all(math.isfinite(x) for x in interval) or interval[0]<0 or interval[0]>interval[1]):raise ValueError('Supply ordered, finite nonnegative powers in nW')
    report={'pressure_bar':pressure,'band_nm':[400,600],'calibration':k=='2.4','interval_nW':interval,'families':{}}
    for key,name in [('aire_par_sommet','area_per_peak'),('forme_reduite','reduced_shape')]:
        lo,hi=d[key][k]['projection_10pct_nW']; report['families'][name]={'range_nW':[lo,hi]}
        if interval is not None:
            a,b=interval;report['families'][name]['comparison']='disjoint' if b<lo or a>hi else 'contained' if lo<=a<=b<=hi else 'overlapping'
    return report
