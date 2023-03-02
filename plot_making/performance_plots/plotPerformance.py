#!/bin/python3
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'cmr10',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False,
            'axes.labelsize': 34,
            'font.size': 12,
            'figure.dpi': 100,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'axes.axisbelow': 'line',
            'axes.formatter.use_mathtext': True,
})

EVENTS = {
    'TSStep',
    'KSPSolve',
    'SNESSolve',
    'SNESFunctionEval',
    'SNESJacobianEval',
    'PCSetUp',
    'PCApply',
    'MatMult',
    'MatSetValuesCOO',
    'KSPGMRESOrthog',
}

def parse_file_content(filename):

    record_list = []
    with open(filename, 'r') as fd:
        file_data = {}
        for line in fd:
            ll = line.strip().split()
            if line.startswith('-- Navier-Stokes solver'):
                pass
            elif line.strip().startswith('Host Name'):
                file_data['Hostname'] = ll[-1]
            elif line.strip().startswith('Number of 1D Basis Nodes'):
                file_data['Order'] = int(ll[-1]) - 1
            elif line.strip().startswith('DM MatType'):
                file_data['MatType'] = ll[-1]
            elif line.strip().startswith('-dm_plex_box_faces'):
                file_data['Extent'] = ll[1].split(',')[0]
            elif line.strip().startswith('Total ranks'):
                file_data['Ranks'] = int(ll[-1])
            elif 'Global DoFs' in line:
                file_data['Global DoFs'] = int(ll[-1])
            elif 'Owned nodes' in line:
                file_data['Nodes/GPU'] = int(ll[-1])
            elif ll and ll[0].strip() in EVENTS:
                event = ll[0].strip()
                #print(line)
                #event = line[:16]
                count = int(line[17:24])
                #count_balance = float(line[24:28])
                time = float(line[29:39])
                #time_balance = float(line[39:43])
                #print(event, count, count_balance, time, time_balance)
                file_data[event] = time
                file_data[f"{event} count"] = count
                file_data[f"{event} t/c"] = time/count
            if line.startswith("#End of PETSc Option Table entries"):
                record_list.append(file_data)
                file_data = {}

    if file_data:
        print(f"Incomplete records in {filename}; discarding incompletes: {file_data}")

    nsteps = record_list[0]['TSStep count']
    for key in (record_list[0].keys() & EVENTS):
        record_list[0][key + ' t/s'] = record_list[0][key] / nsteps
    return record_list[0]

#%% Get file data
paths = {
    # 'Q1_A': Path("../../data/12-30_p1_ROPI/compFlatPlate.o370627"),
    # 'Q1_B': Path("../../data/12-30_p1_ROPI/compFlatPlate.o386591"),
    # 'Q2_A': Path("../../data/24-60_p2_ROPI/compFlatPlate.o371738"),
    # 'Q2_B': Path("../../data/24-60_p2_ROPI/compFlatPlate.o371599"),
    # 'Q3_A': Path("../../data/36-90_p3_ROPI/compFlatPlate.o371808"),
    # 'Q3_B': Path("../../data/36-90_p3_ROPI/compFlatPlate.o386583"),

    # 'Q1': Path("../../data/8partP1/statsTestPer52.log"),
    # 'Q2': Path("../../data/8partP2/compFlatPlate.o371600"),
    # 'Q3': Path("../../data/8partP3/GPUAM_NoMAGMA_GLog.log"),

    'Q1': Path("../../data/8partP1lag9/compFlatPlate.o440064"),
    'Q2': Path("../../data/8partP2lag9/compFlatPlate.o440091"),
    'Q3': Path("../../data/8partP3lag9/compFlatPlate.o440114"),

    'Q1_sparse': Path("../../data/8partP1lag9_sparse/compFlatPlate.o440083"),
    'Q2_sparse': Path("../../data/8partP2lag9_sparse/compFlatPlate.o440098"),
    'Q3_sparse': Path("../../data/8partP3lag9_sparse/compFlatPlate.o440122"),
}

data = {}

for key, path in paths.items():
    data[key] = parse_file_content(path)

order_cases = [
'Q1',
'Q2',
'Q3',
# 'Q1_sparse',
# 'Q2_sparse',
# 'Q3_sparse',
]

datadf = pd.DataFrame.from_records(data)
datadf = datadf[order_cases]

#%% Plot stuff

event_label_dict = {
    'SNESFunctionEval': r"$\mathcal{G} (\mathbf{Y}_{,t}, \mathbf{Y})$",
    'SNESJacobianEval': r"$\mathrm{d}\mathcal{G} / \mathrm{d} \mathbf{Y}$ Setup",
    'PCSetUp': "PreCond Setup",
    # 'PCApply': "PreCond Apply",
    'MatMult':  r"$\mathrm{d}\mathcal{G} / \mathrm{d} \mathbf{Y} \ \Delta \mathbf{Y}$ "}
for key in list(event_label_dict.keys()):
    event_label_dict[key + ' t/s'] = event_label_dict.pop(key)

generic_plot_kwargs = {'width': 0.5}

plot_data = {}
bottom = np.zeros(len(order_cases))

for event in event_label_dict.keys():
    plot_data[event] = {}
    plot_data[event]['height'] = datadf.loc[event].to_numpy().astype(float)
    plot_data[event]['bottom'] = np.copy(bottom)
    plot_data[event]['label'] = event_label_dict[event]
    bottom += plot_data[event]['height']

# plot_data['PCTotal'] = {}
# plot_data['PCTotal']['height'] = datadf.loc['PCSetUp t/s'].to_numpy().astype(float) + datadf.loc['PCApply t/s'].to_numpy().astype(float)
# plot_data['PCTotal']['bottom'] = np.copy(bottom)
# plot_data['PCTotal']['label'] = "Preconditioning"
# bottom += plot_data['PCTotal']['height']

# plot_data['KSPSolve Other'] = {}
# plot_data['KSPSolve Other']['height'] = datadf.loc['KSPSolve t/s'].to_numpy().astype(float) - datadf.loc['MatMult t/s'].to_numpy().astype(float)
# plot_data['KSPSolve Other']['bottom'] = np.copy(bottom)
# plot_data['KSPSolve Other']['label'] = "LinSolve Misc."
# bottom += plot_data['KSPSolve Other']['height']

plot_data['Misc.'] = {}
plot_data['Misc.']['height'] = datadf.loc['TSStep t/s'].to_numpy().astype(float) - bottom
plot_data['Misc.']['bottom'] = np.copy(bottom)
plot_data['Misc.']['label'] = "Misc."
bottom += plot_data['Misc.']['height']

plot_kwargs = {}
order_events = ['Misc.', 'PCSetUp t/s', 'SNESFunctionEval t/s', 'KSPSolve Other', 'SNESJacobianEval t/s', 'MatMult t/s']
order_events = ['Misc.', 'PCSetUp t/s', 'SNESFunctionEval t/s', 'SNESJacobianEval t/s', 'MatMult t/s']
# order_events = ['Misc.', 'PCSetUp t/s', 'PCApply t/s', 'SNESFunctionEval t/s', 'SNESJacobianEval t/s', 'MatMult t/s']
bottom = np.zeros(len(order_cases))
for key in order_events:
    plot_kwargs[key] = plot_data[key].copy()
    plot_kwargs[key]['bottom'] = np.copy(bottom)
    bottom += plot_kwargs[key]['height']

xaxis_labels = [r'$Q_1$', r'$Q_2$', r'$Q_3$', r'$Q_1$ sparse', r'$Q_2$ sparse', r'$Q_3$ sparse']
fig, ax = plt.subplots(figsize=(6.4,5.6))

bar_labels = ['{:.2f}'.format(num) for num in datadf.loc['TSStep t/s'].to_numpy().astype(float)]

for key, plot_kwarg in plot_kwargs.items():
    p = ax.bar(xaxis_labels[:len(order_cases)], **plot_kwarg, **generic_plot_kwargs)

    if key == order_events[-1]:
        ax.bar_label(p, bar_labels)

ax.set_ylabel('Time per Step (s)')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
ax.grid(False)
plt.tight_layout()
