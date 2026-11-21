"""Extract input scalars for the multiscale views from the distributed benchmark."""
import json
import numpy as np


def additional_inputs(db, archive, rows):
    nuclides = {'H': (1, 1), 'C': (6, 12), 'O': (8, 16), 'F': (9, 19), 'S': (16, 32), 'Kr': (36, 84)}
    masses = {}
    for atom, (z, a) in nuclides.items():
        row = db.execute("SELECT mass_neutral_u FROM NUCLIDES_NUBASE2020 WHERE Z=? AND A=? AND state_index=0", (z, a)).fetchone()
        masses[atom] = row[0]
    names = db.execute("SELECT symbol, reference_core, count_status FROM ELEMENTS ORDER BY Z").fetchall()
    prefix = 'BST_R573/'
    read = lambda name: json.loads(archive.read(prefix+name+'.json'))
    source = read('resultats/AIRES_IR_ET_LECTEUR_R573')['candidats']
    field = read('entrees/preuves/OBSERVATEURS_CHAMPS_R446')['cas']
    sf = read('entrees/ETAT_SF6_PARENT_R480')
    s = sf['chart']
    frame = np.array(s['contact_frame'])
    pos = np.array(s['contact_base']) + np.einsum('ij,ijk->ik',np.array(sf['coordinates'][48:60]).reshape(6,2),frame)
    length = np.linalg.norm(pos,axis=1)
    unit = pos/length[:,None]
    r = np.asarray(sf['radii'])
    mean = r.mean()
    du = (frame-unit[:,None,:]*np.einsum('ij,ikj->ik',unit,frame)[:,:,None])/length[:,None,None]
    v = np.array(rows['fluoride']['vectors'])
    fits = {}
    for key, archive_key in [('modern','moderne'), ('extended','moderne_avec_1995')]:
        branch = source[archive_key]['solutions'][0]
        a,b = branch['a_b_etalons']
        # Derivative of the calibrated electric dipole along each native eigenvector.
        dp = a*np.einsum('mj,ji->mi', v[:,12:],unit)
        dp += np.einsum('mjk,jki,j->mi',v[:,:12].reshape(66,6,2),du,a*(r-mean)+b*mean)
        np.testing.assert_allclose(np.sum(dp**2,axis=1),branch['intensites'],rtol=1e-6,atol=2e-4)
        fits[key] = {'vectors':dp.round(8).tolist(),'coefficients':[a,b],'maxBandAreaErrorPct':branch['ecart_max_4_aires_pct']}
    return {'atomic':{a:{'core':b,'status':c} for a,b,c in names},
            'covalentRadiusA':{'H':.31,'C':.76,'O':.66,'F':.57,'S':1.05,'Kr':1.16,'I':1.39},
            'nuclideMassU':masses,'gravityG':6.67430e-11,'atomicMassKg':1.66053906892e-27,
            'dipoleFits':fits,'fieldRuns':[{'recipe':c['recette'][-1].upper(),'grid':c['n'],'dt':c['dt']} for c in field]}
