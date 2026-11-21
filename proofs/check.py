"""Build the proof libraries and report the axioms of every audited declaration."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv, json, os, re, shutil, subprocess
ROOT=Path(__file__).resolve().parent
def check(output):
    version=subprocess.check_output(['lean','--version'],text=True).strip()
    if 'version 4.30.0,' not in version:
        raise RuntimeError(f'Lean 4.30.0 is required. Found {version}')
    with (ROOT/'declarations.csv').open() as f:
        expected=dict(list(csv.reader(f))[1:])
    with TemporaryDirectory() as tmp:
        root=Path(tmp)/'proofs'
        shutil.copytree(ROOT,root)
        declared={}
        def audit(where, prefix, names, lake=False):
            if not names:return
            audit_file=where/'Audit.lean'
            audit_file.write_text(prefix+'\n'+'\n'.join('#print axioms '+n for n in names)+'\n')
            cmd=['lake','env','lean','Audit.lean'] if lake else ['lean','Audit.lean']
            output=subprocess.check_output(cmd,cwd=where,stderr=subprocess.STDOUT,text=True)
            for m in re.finditer(r"'([^']+)' (?:does not depend on any axioms|depends on axioms:\s*\[([^\]]*)\])",output):
                declared[m[1]]=' '.join((m[2] or '').split())
        for project in ('bridge27','projections'):
            cwd=root/project
            subprocess.run(['lake','build','BST'],cwd=cwd,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            names=json.loads((ROOT/'groups.json').read_text())[project]
            audit(cwd,'import BST',names,True)
        for p in sorted((root/'standalone').glob('*.lean')):
            names=json.loads((ROOT/'groups.json').read_text())[p.stem]
            audit(p.parent,p.read_text(),names)
        if declared!=expected:
            raise AssertionError(f'Declaration or axiom mismatch: {set(expected)^set(declared)}, '+str([k for k in declared if declared[k]!=expected.get(k)]))
        if any('sorryAx' in x for x in declared.values()):raise AssertionError('Incomplete proof')
        report={'compiler':version,'checked_declarations':len(declared),'no_listed_axioms':sum(not x for x in declared.values()),'declarations':declared}
        Path(output).write_text(json.dumps(report,indent=2)+'\n')
        return {k:v for k,v in report.items() if k!='declarations'}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path);print(check(p.parse_args().output))
