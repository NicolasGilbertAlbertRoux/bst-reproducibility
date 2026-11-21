'Directly check the original base of the ions and isotopes and its cross-sectional view.'
import hashlib,sqlite3
from readout_ir import I,decode,save
def main():
    path=I/'ATLAS_IONS_ISOTOPES_R557.sqlite';expected=decode('DECISION_R572.json')['atlas_original']
    sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==expected['SHA256_SQLite']
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        assert conn.execute('PRAGMA quick_check').fetchone()==('ok',)
        count={name:conn.execute('SELECT count(*) FROM '+name).fetchone()[0] for name in ('ELEMENTS','MASSES_AME2020','NUCLIDES_NUBASE2020','IONISATIONS','ETATS_DE_CHARGE','CROISEMENT')}
        ext=conn.execute('SELECT count(*) FROM NUCLIDES_NUBASE2020 WHERE state_index != 0').fetchone()[0]
    assert count['CROISEMENT']==expected['positions_isotope_state_charge'] and ext==expected['inputs_excitation']
    save('ATLAS_VERIFIE_R573.json',dict(sha256=sha,rows=count,states_excites=ext,reference=expected['lien'],file='entrees/'+path.name))
if __name__=='__main__':main()
