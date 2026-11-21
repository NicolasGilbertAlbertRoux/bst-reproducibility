"""Export the numerical trajectory used by the offline gravity experiment as CSV."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def simulate(preparation='binary', settings=None, substeps=60):
    request=dict(preparation=preparation,settings=settings or {},substeps=substeps)
    run=subprocess.run(['node',str(ROOT/'observatory/gravity-export.cjs')],input=json.dumps(request),capture_output=True,text=True,check=True)
    return json.loads(run.stdout)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('preparation',choices=['binary','triple']);p.add_argument('output',type=Path)
    p.add_argument('--settings',type=Path,help='JSON object containing the same body, radius, density, speed and separation settings as the console.')
    p.add_argument('--substeps',type=int,default=60,help='Integration steps per hour, for convergence checks.')
    a=p.parse_args();sim=simulate(a.preparation,json.loads(a.settings.read_text()) if a.settings else None,a.substeps)
    with a.output.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_s','body','species','radius_m','mass_kg','constituent_count','x_m','y_m','max_relative_energy_drift','contact'])
        for r in sim['records']:
            for i,(xy,b) in enumerate(zip(r['xy'],sim['bodies']),1):
                w.writerow([r['t'],i,b['species'],b['radius'],b['mass'],b['count'],*xy,r['drift'],r['contact']])
