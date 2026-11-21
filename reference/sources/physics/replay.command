#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
mode="${1:---rapide}"
case "$mode" in
  --rapide) ;;
  --complet)
    python3 code/trajectoire.py 72
    python3 code/defaut.py 40,.09,13 48,.09,13 56,.09,13 64,.09,13 72,.09,13 40,.09,16 40,.045,13 111,.045,12 221,.0225,10 331,.015,9 441,.01125,8.75 551,.009,8.5
    for calc in kirchhoff lecteur_ir croiser_spectre charge_et_unites trois_bandes reponse_commune quantifier oscillations_cosmologie calorimetrie ajuster_chaleur viriels_etendus phase_commune gauss_commun transport_commun cisaillement; do python3 "code/$calc.py"; done
    python3 code/cisaillement.py --complet
    ;;
  *) echo 'Choose --rapide or --complet.' >&2; exit 2 ;;
esac
for calc in atlas bilan_flux geometrie_et_ports bilan_cisaillement intervalle_diffusion conditions matrice registre figures verifier; do python3 "code/$calc.py"; done
