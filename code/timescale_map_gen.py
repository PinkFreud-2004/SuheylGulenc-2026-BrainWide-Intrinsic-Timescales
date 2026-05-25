"""
fig2_timescale_map.py
----------------------
Adapted for local execution (no Google Colab required).

Usage:
    python fig2_timescale_map.py
    python fig2_timescale_map.py --csv /path/to/good_isttc.csv
    python fig2_timescale_map.py --csv /path/to/good_isttc.csv --output /path/to/output_dir

Requirements:
    pip install numpy pandas matplotlib scipy
"""

# -- Imports ------------------------------------------------------------------
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS  —  edit these defaults to match your local setup,
#           or pass them as command-line arguments (see Usage above)
# =============================================================================
DEFAULT_CSV_PATH   = 'good_isttc.csv'
DEFAULT_OUTPUT_DIR = 'isttc_summary_plots'


def parse_args():
    parser = argparse.ArgumentParser(description='Plot iSTTC timescale map (Fig 2b-d)')
    parser.add_argument('--csv',    default=DEFAULT_CSV_PATH,
                        help=f'Path to good_isttc.csv (default: {DEFAULT_CSV_PATH})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for plots (default: {DEFAULT_OUTPUT_DIR})')
    return parser.parse_args()


# -- Beryl -> 12-structure mapping -------------------------------------------
STRUCTURE_ORDER = ['CTX', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL',
                   'TH',  'HY',  'MB',  'P',     'MY',  'CB']

STRUCTURE_LABELS = {
    'CTX':'Isocortex','OLF':'OLF','HPF':'HPF','CTXsp':'CTXsp',
    'STR':'STR','PAL':'PAL','TH':'TH','HY':'HY',
    'MB':'MB','P':'P','MY':'MY','CB':'CB',
}

STRUCTURE_COLORS = {
    'CTX':'#2CA02C','OLF':'#98DF8A','HPF':'#AEC7E8','CTXsp':'#FFBB78',
    'STR':'#1F77B4','PAL':'#9467BD','TH':'#8C564B','HY':'#E377C2',
    'MB':'#D62728','P':'#FF7F0E','MY':'#BCBD22','CB':'#556B2F',
}

FOREBRAIN_STRUCTS  = ['CTX', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY']
MIDBRAIN_STRUCTS   = ['MB']
HINDBRAIN_STRUCTS  = ['P', 'MY', 'CB']

BERYL_TO_STRUCTURE = {
    'FRP':'CTX','MOs':'CTX','MOp':'CTX','SSp-n':'CTX','SSp-bfd':'CTX',
    'SSp-ll':'CTX','SSp-m':'CTX','SSp-ul':'CTX','SSp-tr':'CTX','SSp-un':'CTX',
    'SSs':'CTX','GU':'CTX','VISC':'CTX','AUDd':'CTX','AUDp':'CTX',
    'AUDpo':'CTX','AUDv':'CTX','VISal':'CTX','VISam':'CTX','VISl':'CTX',
    'VISp':'CTX','VISpl':'CTX','VISpm':'CTX','VISli':'CTX','VISpor':'CTX',
    'VISrl':'CTX','VISa':'CTX','ACAd':'CTX','ACAv':'CTX','PL':'CTX',
    'ILA':'CTX','ORBl':'CTX','ORBm':'CTX','ORBvl':'CTX','AId':'CTX',
    'AIp':'CTX','AIv':'CTX','RSPagl':'CTX','RSPd':'CTX','RSPv':'CTX',
    'TEa':'CTX','PERI':'CTX','ECT':'CTX',
    'MOB':'OLF','AON':'OLF','TT':'OLF','DP':'OLF','PIR':'OLF',
    'COAa':'OLF','COAp':'OLF','PAA':'OLF','TR':'OLF','OT':'OLF',
    'AOB':'OLF','NLOT':'OLF',
    'CA1':'HPF','CA2':'HPF','CA3':'HPF','DG':'HPF','FC':'HPF',
    'ENTl':'HPF','ENTm':'HPF','PAR':'HPF','POST':'HPF','PRE':'HPF',
    'SUB':'HPF','ProS':'HPF','HA':'HPF','TAA':'HPF','APr':'HPF',
    'CLA':'CTXsp','EPd':'CTXsp','EPv':'CTXsp','LA':'CTXsp',
    'BLA':'CTXsp','BMA':'CTXsp','PA':'CTXsp',
    'CP':'STR','ACB':'STR','FS':'STR','LSc':'STR',
    'LSr':'STR','LSv':'STR','SF':'STR','SH':'STR','CEA':'STR',
    'MEA':'STR','IA':'STR',
    'GPe':'PAL','GPi':'PAL','SI':'PAL','MA':'PAL','MS':'PAL',
    'NDB':'PAL','TRS':'PAL','BST':'PAL','BAC':'PAL',
    'VAL':'TH','VM':'TH','VPL':'TH','VPLpc':'TH','VPM':'TH',
    'VPMpc':'TH','PoT':'TH','SPF':'TH','MG':'TH','LGd':'TH',
    'LP':'TH','PO':'TH','POL':'TH','SGN':'TH','Eth':'TH',
    'AV':'TH','AM':'TH','IAM':'TH','IAD':'TH','LD':'TH',
    'MD':'TH','SMT':'TH','PR':'TH','PVT':'TH','RE':'TH',
    'CM':'TH','PCN':'TH','CL':'TH','PF':'TH','PIL':'TH',
    'RT':'TH','IGL':'TH','IntG':'TH','LGv':'TH','MH':'TH',
    'LH':'TH','PVH':'TH','MPO':'TH','MPN':'TH','PVHd':'TH',
    'LHA':'HY','LPOA':'HY','PeF':'TH','STN':'HY','ZI':'HY',
    'NB':'HY','SAG':'HY','PBG':'HY',
    'SNr':'MB','MRN':'MB','SCm':'MB','PAG':'MB','APN':'MB',
    'NOT':'MB','NPC':'MB','SCs':'MB','IC':'MB','RN':'MB',
    'PS':'MB','VTN':'MB','AT':'MB','DT':'MB','SNc':'MB',
    'PPN':'MB','DR':'MB','NLL':'MB','PSV':'MB','PB':'MB',
    'SOC':'MB','DTN':'MB','PCG':'MB',
    'PRNc':'P','SUT':'P','TRN':'P','V':'P','P5':'P','I5':'P',
    'CS':'P','LDT':'P','NI':'P','PRNr':'P','DCO':'P','VCO':'P',
    'CU':'P',
    'ECU':'MY','NTS':'MY','SPVC':'MY','SPVI':'MY','SPVO':'MY',
    'VII':'MY','GRN':'MY','ICB':'MY','IRN':'MY','LIN':'MY',
    'LRN':'MY','MARN':'MY','MDRN':'MY','PARN':'MY','PGRN':'MY',
    'PRP':'MY','LAV':'MY','MV':'MY','SPIV':'MY','SUV':'MY',
    'x':'MY','XII':'MY',
    'AHN':'HY','AVP':'HY','DMH':'HY','LPO':'HY','MEPO':'HY','MM':'HY',
    'MT':'HY','PH':'HY','PMd':'HY','PMv':'HY','PP':'HY','PSTN':'HY',
    'SFO':'HY','SUM':'HY','VMH':'HY','ADP':'HY',
    'AD':'TH','IG':'TH','LM':'TH','PT':'TH','RH':'TH',
    'SPA':'TH','SubG':'TH','Xi':'TH',
    'CLI':'MB','CUN':'MB','IPN':'MB','PDTg':'MB','PPT':'MB','RPF':'MB',
    'RR':'MB','VTA':'MB','III':'MB','OP':'MB','PAS':'MB',
    'CUL4 5':'P','Pa4':'P','Pa5':'P','PC5':'P','PG':'P','RO':'P',
    'AMB':'MY','AP':'MY','DMX':'MY','GR':'MY','IO':'MY','TTd':'MY','TU':'MY',
    'AAA':'OLF','HATA':'OLF',
    'LING':'CB','CENT2':'CB','CENT3':'CB','CUL4,5':'CB','DEC':'CB',
    'FOTU':'CB','PYR':'CB','UVU':'CB','NOD':'CB','SIM':'CB',
    'ANcr1':'CB','ANcr2':'CB','PRM':'CB','COPY':'CB','PFL':'CB',
    'FL':'CB','FN':'CB','IP':'CB','DN':'CB','VeCB':'CB',
}

def get_structure(region):
    return BERYL_TO_STRUCTURE.get(region, None)

def get_division(s):
    if s in FOREBRAIN_STRUCTS:  return 'Forebrain'
    if s in MIDBRAIN_STRUCTS:   return 'Midbrain'
    if s in HINDBRAIN_STRUCTS:  return 'Hindbrain'
    return None


# ============================================================================
# Plotting helpers
# ============================================================================

FOREBRAIN_COSMOS = [
    ('CTX',   ['CTX'],   '#2CA02C'),
    ('OLF',   ['OLF'],   '#98DF8A'),
    ('HPF',   ['HPF'],   '#AEC7E8'),
    ('CTXsp', ['CTXsp'], '#FFBB78'),
    ('STR',   ['STR'],   '#1F77B4'),
    ('PAL',   ['PAL'],   '#9467BD'),
    ('TH',    ['TH'],    '#8C564B'),
    ('HY',    ['HY'],    '#E377C2'),
]
MIDBRAIN_COSMOS = [
    ('MB',    ['MB'],    '#D62728'),
]
HINDBRAIN_COSMOS = [
    ('P',     ['P'],     '#FF7F0E'),
    ('MY',    ['MY'],    '#FF9896'),
    ('CB',    ['CB'],    '#556B2F'),
]

DIVISIONS = [
    ('Forebrain', FOREBRAIN_COSMOS, '#2CA02C'),
    ('Midbrain',  MIDBRAIN_COSMOS,  '#D62728'),
    ('Hindbrain', HINDBRAIN_COSMOS, '#FF7F0E'),
]

ALL_COSMOS = FOREBRAIN_COSMOS + MIDBRAIN_COSMOS + HINDBRAIN_COSMOS


def build_division_bars(df_mapped, division, cosmos_list):
    sub = df_mapped[df_mapped['division'] == division].copy()
    grp = (sub.groupby(['beryl_region', 'structure'])['tau_eff_s']
               .agg(median='median',
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                    n='count')
               .reset_index())
    grp = grp[grp['n'] >= 15]

    ordered_rows, block_dividers, block_labels = [], [], []
    x = 0
    for cosmos_label, struct_keys, color in cosmos_list:
        block_sub = grp[grp['structure'].isin(struct_keys)].sort_values('beryl_region')
        if len(block_sub) == 0:
            continue
        x_start = x
        for _, row in block_sub.iterrows():
            ordered_rows.append({'x': x, 'region': row['beryl_region'],
                                  'median': row['median'], 'q25': row['q25'],
                                  'q75': row['q75'], 'color': color,
                                  'cosmos': cosmos_label})
            x += 1
        block_labels.append(((x_start + x - 1) / 2.0, cosmos_label, color))
        block_dividers.append(x - 0.5)
        x += 0.8
    bars = pd.DataFrame(ordered_rows)
    return bars, block_dividers[:-1], block_labels


def draw_row(ax, bars, block_dividers, block_labels, division, div_color,
             div_label_y=0.5, y_top=None, cosmos_label_y=-0.35):
    if bars.empty:
        ax.set_visible(False)
        return

    ax.bar(bars['x'], bars['median'], color=bars['color'],
           width=0.7, linewidth=0, zorder=2)
    for _, row in bars.iterrows():
        ax.plot([row['x'], row['x']], [row['q25'], row['q75']],
                color='#111111', linewidth=1.8, zorder=3)
    for bx in block_dividers:
        ax.axvline(bx, color='#bbbbbb', linewidth=0.8, linestyle='--', zorder=1)
    for mid_x, label, color in block_labels:
        ax.text(mid_x, cosmos_label_y, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=80,
                fontweight='bold', color=color)

    ax.set_xticks(bars['x'])
    ax.set_xticklabels(bars['region'], rotation=90, fontsize=28, ha='center')
    ax.set_xlim(bars['x'].min() - 0.8, bars['x'].max() + 0.8)

    if y_top is None:
        y_top = max(1.5, bars['q75'].max() * 1.05)
    yticks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    ax.set_ylim(0, y_top)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
    ax.set_axisbelow(True)
    for y in yticks:
        ax.axhline(y, color='#bbbbbb', linewidth=0.8, zorder=0)

    ax.set_ylabel('Effective timescale (s)', fontsize=80, labelpad=80)
    ax.tick_params(axis='y', labelsize=80)
    ax.yaxis.grid(True, linewidth=0.8, color='#bbbbbb', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(1.004, div_label_y, division,
            transform=ax.transAxes, fontsize=80, fontweight='bold',
            color=div_color, rotation=270, va='center', ha='left',
            clip_on=False)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    CSV_PATH   = args.csv
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Load data ------------------------------------------------------------
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            f"Run isttc_pipeline.py first to generate it, or pass --csv /path/to/good_isttc.csv"
        )

    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {len(df):,} rows')

    df['tau_eff_s'] = df['tau_effective_ms'] / 1000.0
    df['structure'] = df['beryl_region'].map(get_structure)

    n_nan_region = df['beryl_region'].isna().sum()
    if n_nan_region:
        print(f'Info: {n_nan_region:,} rows have no beryl_region -- excluded.')
        df = df.dropna(subset=['beryl_region']).reset_index(drop=True)

    df = df[~df['beryl_region'].isin(['root', 'void'])].reset_index(drop=True)

    unmapped = df[df['structure'].isna()]['beryl_region'].dropna().unique()
    if len(unmapped) > 0:
        print(f'Warning: {len(unmapped)} beryl regions not mapped to a structure:')
        print(sorted(unmapped))
    else:
        print('OK: all beryl regions mapped to a structure.')

    df['division'] = df['structure'].map(get_division)

    mapped = df.dropna(subset=['structure', 'tau_eff_s'])
    print(f'Mapped {len(mapped):,} neurons to structures')
    print(f'Structure counts:\n{mapped["structure"].value_counts()[STRUCTURE_ORDER]}')

    # ── Build data ────────────────────────────────────────────────────────────
    all_bars, all_divs, all_labels = {}, {}, {}
    for division, cosmos_list, div_color in DIVISIONS:
        b, d, l = build_division_bars(mapped, division, cosmos_list)
        all_bars[division]   = b
        all_divs[division]   = d
        all_labels[division] = l

    max_x   = max(b['x'].max() if not b.empty else 0 for b in all_bars.values())
    fig_w   = max(18, max_x * 0.70)

    y_top_global = max(
        all_bars[div]['q75'].max()
        for div, _, _ in DIVISIONS
        if not all_bars[div].empty
    ) * 1.05
    y_top_global = min(4.0, max(1.5, y_top_global))

    n_bars  = {div: (len(all_bars[div]) if not all_bars[div].empty else 1)
               for div, _, _ in DIVISIONS}
    n_bars['Midbrain']  = max(n_bars['Midbrain'],  n_bars['Forebrain'] * 0.75)
    n_bars['Hindbrain'] = max(n_bars['Hindbrain'], n_bars['Forebrain'] * 0.75)
    total   = sum(n_bars.values())
    heights = [n_bars[div] / total for div, _, _ in DIVISIONS]

    fig, axes = plt.subplots(
        3, 1,
        figsize=(fig_w, fig_w * 0.65),
        facecolor='white',
        gridspec_kw={'hspace': 0.45, 'height_ratios': heights}
    )

    for ax, (division, cosmos_list, div_color) in zip(axes, DIVISIONS):
        div_label_y    = 0.35 if division == 'Midbrain' else 0.5
        cosmos_label_y = -0.15 if division == 'Forebrain' else -0.25 if division == 'Midbrain' else -0.35
        draw_row(ax, all_bars[division], all_divs[division],
                 all_labels[division], division, div_color,
                 div_label_y=div_label_y, y_top=y_top_global,
                 cosmos_label_y=cosmos_label_y)
        if division == 'Forebrain':
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height])

    legend_handles = [mpatches.Patch(color=color, label=label)
                      for label, _, color in ALL_COSMOS]
    fig.legend(handles=legend_handles, fontsize=60, ncol=len(ALL_COSMOS),
               loc='upper center', bbox_to_anchor=(0.5, 0.98),
               frameon=False, handlelength=2.5, handletextpad=0.8, columnspacing=2,
               borderpad=1.2, markerscale=3)

    out_path = os.path.join(OUTPUT_DIR, 'fig_all_divisions_15neuron_600dpi.png')
    fig.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
