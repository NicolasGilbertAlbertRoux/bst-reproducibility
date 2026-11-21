"""Localize the annotations drawn into the reference GIF frames."""
import os
import re

ENGLISH = {'Les manteaux du krypton': 'The krypton mantles',
    'La couleur du dicarbone': 'The colour of dicarbon',
    'C₂ EXCITÉ': 'EXCITED C₂',
    'Deux amas de 309 sites. La forme du manteau suit leurs positions.': 'Two clusters of 309 sites. Each mantle follows the site positions.',
    'Une forme calculée sur 34 sites par atome, éclairée par la teinte du filtre choisi.': 'A shape calculated from 34 sites per atom, shown in the selected filter colour.',
    'À la résolution moléculaire': 'At molecular resolution',
    'Détail atomique': 'Atomic detail',
    'Manteaux seuls': 'Mantles only',
    'Distance entre les centres': 'Centre separation',
    'Gris · vitesse de déformation du manteau.': 'Grey shows the local mantle strain rate.',
    'Le gris garde une trace du mouvement interne. Les deux vues de droite suivent le même instant.': 'Grey traces the internal motion. Both views on the right show the same instant.',
    'Temps du dimère': 'Dimer time',
    'Filtre': 'Filter',
    'Bandes de Swan mesurées. Leur localisation sur les manteaux et leur éclat sont hypothétiques.': 'Measured Swan bands. Their position and brightness on the mantles are hypotheses.',
    'Même mouvement rejoué sous trois filtres   ·   Temps de la vibration': 'Same motion under three filters   ·   Vibrational time',
    'UV · gris': 'UV · grey',
    'À gauche, le mouvement des sites. À droite, trois raies du krypton neutre excité.': 'Left, motion of the sites. Right, three lines of excited neutral krypton.',
    'Trois filtres optiques': 'Three optical filters',
    'Source non résolue · même ouverture': 'Unresolved source · same aperture',
    'Le blanc de l’amas code le mouvement des 309 sites. Déplacements internes agrandis × 12.': 'White traces the motion of the 309 sites. Internal displacements enlarged × 12.',
    'Dans le filtre UV, le gris rend visible une raie mesurée à 123,58 nm. Éclats normalisés.': 'Grey represents the measured 123.58 nm UV line. Brightness is normalized.',
    'Le battement et le contact': 'The beat and the contact',
    'ONDES': 'WAVES',
    'Le blanc suit le mouvement local du manteau. L’ambre suit l’énergie du contact.': 'White traces local mantle motion. Amber traces stored contact energy.',
    'Une onde libre': 'A free wave',
    'Deux ondes en interaction': 'Two interacting waves',
    'Retournement · halo éteint': 'Turning point · halo dark',
    'Dilatation': 'Expansion',
    'Énergie au contact': 'Contact energy',
    'Rayon': 'Radius',
    'Puissance vers la séparation': 'Power into separation',
    'Attraction dominante': 'Net attraction',
    'Répulsion dominante': 'Net repulsion',
    'Forces équilibrées': 'Balanced forces',
    'Force nette sur la séparation': 'Net separation force',
    'unités du modèle': 'model units',
    'Retournement du battement en contact': 'Turning point of the coupled beat',
    'Le sillage gris conserve jusqu’à neuf images du trajet.': 'The grey wake retains up to nine past positions.',
    'Un atome de krypton': 'A krypton atom',
    'La géométrie calculée à gauche et deux vues filtrées d’un émetteur excité à droite.': 'Calculated geometry on the left and two filtered views of an excited emitter on the right.',
    'Détail × 1,6': 'Detail × 1.6',
    'Krypton neutre excité': 'Excited neutral krypton',
    'Deux raies de Kr I (NIST)': 'Two Kr I lines (NIST)',
    'Même ouverture optique': 'Same optical aperture',
    'Blanc : mouvement des 309 sites du calcul, huit modes. Déplacements agrandis × 12.': 'White traces the 309 sites in the eight-mode calculation. Displacements enlarged × 12.',
    'Couleurs : émetteur non résolu dans deux filtres. Profils optiques illustratifs, éclats normalisés.': 'An unresolved emitter in two filters. Illustrative profiles, normalized brightness.'}

def translate(text):
    if os.environ.get("BST_PLATE_LANGUAGE", "fr") == "fr":
        return text
    for source, target in ENGLISH.items():
        text = text.replace(source, target)
    return re.sub(r"(?<=\d),(?=\d)", ".", text)
