"""
Figure 1 Panels A and B Only
A) Effective Timescale by Brain Division (3 boxplots)
B) Timescale × Beryl Brain Subdivisions (12 boxplots)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
CSV_PATH = 'good_isttc.csv'
OUTPUT_DIR = 'figure1_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Structure definitions
STRUCTURE_COLORS = {
    'CTX':'#2CA02C', 'OLF':'#98DF8A', 'HPF':'#AEC7E8', 'CTXsp':'#FFBB78',
    'STR':'#1F77B4', 'PAL':'#9467BD', 'TH':'#8C564B', 'HY':'#E377C2',
    'MB':'#D62728', 'P':'#FF7F0E', 'MY':'#BCBD22', 'CB':'#556B2F',
}

FOREBRAIN_STRUCTS = ['CTX', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY']
MIDBRAIN_STRUCTS = ['MB']
HINDBRAIN_STRUCTS = ['P', 'MY', 'CB']

# Beryl to structure mapping
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
    'CS':'P','LDT':'P','NI':'P','PRNr':'P','DCO':'P','VCO':'P','CU':'P',
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
    if s in FOREBRAIN_STRUCTS: return 'Forebrain'
    if s in MIDBRAIN_STRUCTS: return 'Midbrain'
    if s in HINDBRAIN_STRUCTS: return 'Hindbrain'
    return None

# Load and prepare data
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df['tau_eff_ms'] = df['tau_effective_ms']
df['structure'] = df['beryl_region'].map(get_structure)
df['division'] = df['structure'].map(get_division)

df = df.dropna(subset=['beryl_region']).reset_index(drop=True)
df = df[~df['beryl_region'].isin(['root', 'void'])].reset_index(drop=True)
mapped = df.dropna(subset=['structure', 'tau_eff_ms'])

print(f'Mapped {len(mapped):,} neurons to structures\n')

# =============================================================================
# PANEL A: Effective Timescale by Brain Division
# =============================================================================

print("Generating Panel A...")

division_data = []
division_labels = []
division_colors = []
division_stats = []

for div_name, color in [('Forebrain', '#7FC97F'), 
                         ('Midbrain', '#FDC086'), 
                         ('Hindbrain', '#FFCC99')]:
    data = mapped[mapped['division'] == div_name]['tau_eff_ms'].values
    if len(data) > 0:
        division_data.append(data)
        division_labels.append(div_name)
        division_colors.append(color)
        median = np.median(data)
        n_regions = len(mapped[mapped['division'] == div_name]['beryl_region'].unique())
        division_stats.append((median, n_regions))
        print(f"  {div_name}: median={median:.1f}ms, n_regions={n_regions}")

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')

bp = ax.boxplot(division_data, 
                labels=division_labels,
                patch_artist=True,
                widths=0.5,
                showfliers=False,
                medianprops=dict(color='black', linewidth=2.5),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], division_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# Position annotations inside plot
for i, (median, n_regions) in enumerate(division_stats):
    y_pos = 1500  # Fixed position in middle-upper area
    ax.text(i + 1, y_pos, 
            f'med={int(median)} ms\nn={n_regions} regions',
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='none', alpha=0.8))

ax.set_yscale('log')
ax.set_ylabel('τ_eff median (ms, log scale)', fontsize=13, fontweight='bold')
ax.set_title('Effective Timescale by Brain Division', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.25, axis='y', linestyle='-', linewidth=0.8)

# Don't cut off bottom - show full range
ax.set_ylim(bottom=50, top=3000)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='both', labelsize=11, width=1.5, length=6)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'figure1_panelA_division_boxplot.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {out_path}\n")

# =============================================================================
# PANEL B: Timescale × Beryl Brain Subdivisions
# =============================================================================

print("Generating Panel B...")

# Exact order from the uploaded figure
PANEL_B_ORDER = ['HPF', 'CTXsp', 'OLF', 'STR', 'PAL', 'CTX', 'HY', 'TH', 'MB', 'CB', 'MY', 'P']

structure_stats = []
for struct in PANEL_B_ORDER:
    data = mapped[mapped['structure'] == struct]['tau_eff_ms'].values
    if len(data) > 0:
        median = np.median(data)
        n_regions = len(mapped[mapped['structure'] == struct]['beryl_region'].unique())
        structure_stats.append({
            'structure': struct,
            'median': median,
            'data': data,
            'n_regions': n_regions,
            'color': STRUCTURE_COLORS[struct]
        })
        print(f"  {struct}: median={median:.1f}ms, n_regions={n_regions}")

plot_data = [s['data'] for s in structure_stats]
plot_labels = [s['structure'] for s in structure_stats]
plot_colors = [s['color'] for s in structure_stats]

fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')

positions = np.arange(1, len(plot_data) + 1)
bp = ax.boxplot(plot_data,
                positions=positions,
                labels=plot_labels,
                patch_artist=True,
                widths=0.6,
                showfliers=False,
                medianprops=dict(color='black', linewidth=2.5),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

for patch, color in zip(bp['boxes'], plot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# Add annotations inside plot
for i, stat in enumerate(structure_stats):
    y_pos = 1800  # Fixed position
    ax.text(i + 1, y_pos,
            f"med={int(stat['median'])}ms\nn={stat['n_regions']}",
            ha='center', va='top', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='none', alpha=0.8))

# Left y-axis
ax.set_yscale('log')
ax.set_ylabel('τ_eff median (ms, log scale)', fontsize=13, fontweight='bold')
ax.set_xlabel('Beryl Subdivision', fontsize=13, fontweight='bold')
ax.set_title('Timescale × Beryl Brain Subdivisions', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.25, axis='y', linestyle='-', linewidth=0.8)

# Don't cut off bottom - show full range
ax.set_ylim(bottom=30, top=4000)

# Right y-axis
ax2 = ax.twinx()
ax2.set_yscale('log')
ax2.set_ylabel('Slow Timescale τ₂ (ms) [Log Scale]', fontsize=13, fontweight='bold')
ax2.set_ylim(ax.get_ylim())  # Match left axis
ax2.tick_params(labelsize=11, width=1.5, length=6)

# Styling
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='x', labelsize=12, rotation=0, width=1.5, length=6)
ax.tick_params(axis='y', labelsize=11, width=1.5, length=6)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'figure1_panelB_structure_boxplot.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: {out_path}\n")


"""
Reproduces panel C of Figure 1 - REVISED with larger fonts and fewer y-ticks
"""

"""
Reproduces panel C of Figure 1 - REVISED with larger fonts and fewer y-ticks
"""

# -- Imports ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')


CSV_PATH   = f'good_isttc.csv'
OUTPUT_DIR = f'figure1_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Load data ----------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f'Loaded {len(df):,} rows')

df['tau_eff_s'] = df['tau_effective_ms'] / 1000.0

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

def get_division(s):
    if s in FOREBRAIN_STRUCTS:  return 'Forebrain'
    if s in MIDBRAIN_STRUCTS:   return 'Midbrain'
    if s in HINDBRAIN_STRUCTS:  return 'Hindbrain'
    return None

df['division'] = df['structure'].map(get_division)

mapped = df.dropna(subset=['structure', 'tau_eff_s'])
print(f'Mapped {len(mapped):,} neurons to structures')
print(f'Structure counts:\n{mapped["structure"].value_counts()[STRUCTURE_ORDER]}')


# ============================================================================
# Plotting - REVISED FOR LARGER SIZE AND FONTS
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
                color='#111111', linewidth=2.5, zorder=3)  # Thicker error bars
    
    for bx in block_dividers:
        ax.axvline(bx, color='#bbbbbb', linewidth=1.0, linestyle='--', zorder=1)
    
    for mid_x, label, color in block_labels:
        ax.text(mid_x, cosmos_label_y, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=110,  # Increased from 80
                fontweight='bold', color=color)

    ax.set_xticks(bars['x'])
    ax.set_xticklabels(bars['region'], rotation=90, fontsize=40, ha='center')  # Increased from 28
    ax.set_xlim(bars['x'].min() - 0.8, bars['x'].max() + 0.8)

    if y_top is None:
        y_top = max(1.5, bars['q75'].max() * 1.05)
    
    # REVIEWER REQUEST: Reduce number of y-ticks to 3-4 and increase font size
    yticks = [0, 1.0, 2.0, 3.0, 4.0]  # Reduced from many ticks
    ax.set_ylim(0, y_top)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
    ax.set_axisbelow(True)
    for y in yticks:
        ax.axhline(y, color='#bbbbbb', linewidth=1.0, zorder=0)

    ax.set_ylabel('Effective timescale (s)', fontsize=110, labelpad=100)  # Increased from 80
    ax.tick_params(axis='y', labelsize=110)  # Increased from 80
    ax.yaxis.grid(True, linewidth=1.0, color='#bbbbbb', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    
    ax.text(1.004, div_label_y, division,
            transform=ax.transAxes, fontsize=110, fontweight='bold',  # Increased from 80
            color=div_color, rotation=270, va='center', ha='left',
            clip_on=False)


# ── Build data ────────────────────────────────────────────────────────────────
all_bars, all_divs, all_labels = {}, {}, {}
for division, cosmos_list, div_color in DIVISIONS:
    b, d, l = build_division_bars(mapped, division, cosmos_list)
    all_bars[division]   = b
    all_divs[division]   = d
    all_labels[division] = l

max_x   = max(b['x'].max() if not b.empty else 0 for b in all_bars.values())
fig_w   = max(24, max_x * 0.85)  # Increased from 18 and 0.70

# Shared y_top across all divisions, capped at 4.0
y_top_global = max(
    all_bars[div]['q75'].max()
    for div, _, _ in DIVISIONS
    if not all_bars[div].empty
) * 1.05
y_top_global = min(4.0, max(1.5, y_top_global))

# Row heights proportional to number of regions
n_bars  = {div: (len(all_bars[div]) if not all_bars[div].empty else 1)
           for div, _, _ in DIVISIONS}
n_bars['Midbrain']  = max(n_bars['Midbrain'],  n_bars['Forebrain'] * 0.75)
n_bars['Hindbrain'] = max(n_bars['Hindbrain'], n_bars['Forebrain'] * 0.75)
total   = sum(n_bars.values())
heights = [n_bars[div] / total for div, _, _ in DIVISIONS]

fig, axes = plt.subplots(
    3, 1,
    figsize=(fig_w, fig_w * 0.75),  # Increased height ratio from 0.65
    facecolor='white',
    gridspec_kw={'hspace': 0.50, 'height_ratios': heights}  # Increased from 0.45
)

for ax, (division, cosmos_list, div_color) in zip(axes, DIVISIONS):
    div_label_y    = 0.35 if division == 'Midbrain' else 0.5
    cosmos_label_y = -0.12 if division == 'Forebrain' else -0.20 if division == 'Midbrain' else -0.30
    draw_row(ax, all_bars[division], all_divs[division],
             all_labels[division], division, div_color,
             div_label_y=div_label_y, y_top=y_top_global,
             cosmos_label_y=cosmos_label_y)
    if division == 'Forebrain':
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + 0.04, box.width, box.height])

# Single shared legend at top - INCREASED SIZE
legend_handles = [mpatches.Patch(color=color, label=label)
                  for label, _, color in ALL_COSMOS]
fig.legend(handles=legend_handles, fontsize=85, ncol=len(ALL_COSMOS),  # Increased from 60
           loc='upper center', bbox_to_anchor=(0.5, 0.985),
           frameon=False, handlelength=2.8, handletextpad=1.0, columnspacing=2.5,
           borderpad=1.5, markerscale=4)

out_path = os.path.join(OUTPUT_DIR, 'figure1_panelC_all_divisions.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f'Saved: {out_path}')

"""
Reproduces panel D of Figure 1
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
from iblatlas.regions import BrainRegions

# =============================================================================
# PATHS  —  edit these defaults or pass as command-line arguments
# =============================================================================
DEFAULT_CSV_PATH   = 'region_timescale_summary.csv'
DEFAULT_OUTPUT_DIR = 'figure1_plots'


def parse_args():
    parser = argparse.ArgumentParser(description='Figure 1 Panel D Generator')
    parser.add_argument('--csv',    default=DEFAULT_CSV_PATH,
                        help=f'Path to region_timescale_summary.csv (default: {DEFAULT_CSV_PATH})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for plots (default: {DEFAULT_OUTPUT_DIR})')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    summary_path = args.csv
    df = pd.read_csv(summary_path)
    df = df[(df['region_acronym'].notna()) & (df['n_neurons'] >= 15)].copy()
    
    br = BrainRegions()
    
    def beryl_to_cosmos(acronym):
        try:
            rid = np.atleast_1d(br.acronym2id(acronym))[0]
            cosmos_id = br.remap(np.array([rid]), source_map='Allen', target_map='Cosmos')[0]
            return br.id2acronym(np.array([cosmos_id]))[0]
        except:
            return 'Unknown'
    
    df['cosmos'] = df['region_acronym'].apply(beryl_to_cosmos)
    df = df[~df['cosmos'].isin(['Unknown', 'void', 'root', 'grey'])].copy()
    
    forebrain_cosmos = ['HPF', 'OLF', 'CTXsp', 'CNU', 'HY', 'Isocortex', 'TH']
    midbrain_cosmos  = ['MB']
    hindbrain_cosmos = ['HB', 'CB']
    
    def assign_division(cosmos):
        if cosmos in forebrain_cosmos:  return 'Forebrain'
        if cosmos in midbrain_cosmos:   return 'Midbrain'
        if cosmos in hindbrain_cosmos:  return 'Hindbrain'
        return 'Other'
    
    df['division'] = df['cosmos'].apply(assign_division)
    df = df[df['division'] != 'Other'].copy()
    
    # Keep only regions with both tau_1 and tau_2
    plot_df = df.dropna(subset=['tau_1_ms_median', 'tau_2_ms_median']).copy()
    
    # Log-log regression for trend line
    log_tau1 = np.log10(plot_df['tau_1_ms_median'].values)
    log_tau2 = np.log10(plot_df['tau_2_ms_median'].values)
    slope, intercept, r, p, _ = stats.linregress(log_tau1, log_tau2)
    x_line = np.linspace(log_tau1.min(), log_tau1.max(), 200)
    y_line = slope * x_line + intercept
    
    # ── Plot with colorblind-friendly palette ────────────────────────────────
    division_colors = {
        'Forebrain': '#0173B2',   # blue (colorblind-safe)
        'Midbrain':  '#DE8F05',   # orange (colorblind-safe)
        'Hindbrain': '#CC78BC',   # purple/pink (colorblind-safe)
    }
    
    fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
    ax.set_facecolor('white')
    
    division_labels = {
        div: f"{div} (n={len(grp)})"
        for div, grp in plot_df.groupby('division')
    }
    
    for division, grp in plot_df.groupby('division'):
        ax.scatter(
            grp['tau_1_ms_median'],
            grp['tau_2_ms_median'],
            color=division_colors[division],
            s=70,
            alpha=0.82,
            edgecolors='white',
            linewidths=0.5,
            label=division_labels[division],
            zorder=3,
        )
    
    # Trend line
    ax.plot(
        10**x_line, 10**y_line,
        color='#555555', linewidth=1.8,
        linestyle='--', zorder=2,
    )
    
    # Stats annotation
    ax.text(
        0.97, 0.05,
        f'slope = {slope:.3f}\nr = {r:.3f}, p = {p:.2e}\nN = {len(plot_df)} regions',
        transform=ax.transAxes,
        fontsize=10, color='#333333',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.85),
    )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('Fast Timescale τ₁ (ms) [Log Scale]', fontsize=12, color='#333333', labelpad=8)
    ax.set_ylabel('Slow Timescale τ₂ (ms) [Log Scale]', fontsize=12, color='#333333', labelpad=8)
    ax.set_title('τ₁–τ₂ Coupling Across Brain Regions', fontsize=14, color='#111111', pad=14)
    
    legend = ax.legend(
        title='Major Division', title_fontsize=10,
        fontsize=9, framealpha=0.9,
        edgecolor='#cccccc', facecolor='white',
        loc='upper left',
    )
    legend.get_title().set_fontweight('bold')
    
    ax.tick_params(colors='#555555', which='both')
    ax.grid(True, which='major', alpha=0.15, color='#aaaaaa', linestyle='-')
    ax.grid(True, which='minor', alpha=0.07, color='#aaaaaa', linestyle='-')
    
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    out_path = os.path.join(args.output, 'figure1_panelD_tau1_tau2_coupling.png')
    fig.savefig(out_path, dpi=380, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f'Saved: {out_path}')
    print(f'Log-log slope = {slope:.3f}, r = {r:.3f}, p = {p:.2e}, N = {len(plot_df)}')