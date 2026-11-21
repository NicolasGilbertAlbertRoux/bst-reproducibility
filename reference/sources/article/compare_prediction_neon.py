'Compare a power measurement to neon projections published with BST R573.'
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'references/R573/entrees/PREDICTIONS_GELEES_R572.json'
STANDARD = Decimal('2.4')
NAMES = {'area_par_sommet': 'Area by summit', 'shape_reduced': 'Reduced form'}

def count(texte):
    try:
        value = Decimal(texte.replace(',', '.'))
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError('Expected decimal number.') from exc
    if not value.is_finite():
        raise argparse.ArgumentTypeError('A finite value is required.')
    return value

def charger_intervals():
    document = json.loads(SOURCE.read_text(encoding='utf-8'), parse_float=Decimal)
    families = document['predictions_gelee_before_cette_revue']['families']
    return {ident: {Decimal(p): fiche['projection_10pct_nW'] for p, fiche in values_field.items()}
            for ident, values_field in families.items()}

def compare(families, pressure, minimum, maximum):
    if minimum > maximum:
        raise ValueError('The lower bound shall be less than or equal to the upper bound.')
    if any(pressure not in values_field for values_field in families.values()):
        raise ValueError('Pressures published in bars: 1.9 / 2.1 / 2.4 / 2.7 / 3.1.')
    return {ident: max(minimum, values_field[pressure][0]) <= min(maximum, values_field[pressure][1])
            for ident, values_field in families.items()}

def figure(families, output):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':'white'})
    p = Decimal('3.1')
    a, b = [families[k][p] for k in NAMES]
    fig, ax = plt.subplots(figsize=(7.65, 2.75), layout='constrained')
    low, high = max(a[0], b[0]), min(a[1], b[1])
    ax.axvspan(float(low),float(high),color='#e3ece9',zorder=0)
    ax.text((float(low)+float(high))/2,1.64,'Two compatible families',
            ha='center',color='#234c48',fontsize=9)
    for y, ident, color in [(1,'area_par_sommet','#286867'),(0,'shape_reduced','#995022')]:
        left, right = families[ident][p]
        ax.plot([float(left),float(right)],[y,y],color=color,linewidth=5,solid_capstyle='butt')
        for bound in (left,right):
            ax.plot(float(bound),y,marker='|',markersize=13,color=color,markeredgewidth=1.7)
            ax.annotate(format(bound,'.3f').replace('.',','),(float(bound),y),
                        xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=color)
    ax.set(xlim=(1.0,1.50),ylim=(-.45,1.95),yticks=[0,1],
           yticklabels=['Reduced form','Area by summit'],xlabel='Integrated light output of 400 to 600 nm (nW)')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: f'{x:.1f}'.replace('.',',')))
    for k in ('top','right','left'):
        ax.spines[k].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=9)
    ax.spines['bottom'].set_color('#8a9692')
    fig.savefig(output,dpi=230)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('compare', help='Compare a closed experimental interval.')
    c.add_argument('pressure_bar', type=count)
    c.add_argument('bound_lower_nW', type=count)
    c.add_argument('bound_upper_nW', type=count)
    f = commands.add_parser('figure', help='Reproduce the figure of chapter 7 at 3.1 bar.')
    f.add_argument('output', type=Path)
    options = parser.parse_args()
    families = charger_intervals()
    if options.command == 'figure':
        figure(families, options.output)
        return
    p = options.pressure_bar
    try:
        verdict = compare(families, p, options.bound_lower_nW, options.bound_upper_nW)
    except ValueError as exc:
        parser.error(str(exc))
    print('Calibration pressure check.' if p == STANDARD else 'Comparison with a non-calibration pressure.')
    for ident, compatible in verdict.items():
        left, right = families[ident][p]
        readout = 'intersection non vide' if compatible else 'intervalles disjoints'
        print(f'{NAMES[ident]} [{left}, {right}] nW : {readout}.')

if __name__ == '__main__':
    main()
