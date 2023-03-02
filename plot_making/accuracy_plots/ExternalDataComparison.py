import numpy as np
import vtk
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
import re
import sys

sys.path.append('../../data/externalData')
from externalDataModule import *

import vtkpytools as vpt

plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'cmr10',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False,
            'axes.labelsize': 24,
            'font.size': 12,
            'figure.dpi': 100,
            'lines.linewidth': 1.5,
            'axes.grid': True,
            'axes.axisbelow': 'line',
            'axes.formatter.use_mathtext': True,
})

Uref = 1.0
rho = 1
mu = 1.2504E-5
nu = mu/rho
delta = 0.113
L = 3.1

#%% Load external data

exdata = {}
exdata['sch'] = loadSchlatterData()
exdata['jim'] = loadJimenezData()
exdata['wu'] = loadWuData()
exdata['spa'] = loadSpalartData()
jhtdb, jhtdb_wall_dists = loadJHTDB()


#%% Load data
paths = {
   'Q1': Path('../../data/12-30_p1_ROPI/stats.vtm'),
   'Q2': Path('../../data/24-60_p2_ROPI/stats.vtm'),
   'Q3': Path('../../data/36-90_p3_ROPI/stats.vtm'),
}

dataBlocks = {'jhtdb': jhtdb}
for key, path in paths.items():
    dataBlocks[key] = pv.MultiBlock(path.as_posix())

    grid = dataBlocks[key]['grid']
    grid = grid.compute_derivative(scalars='Mean Velocity ', gradient='gradient', vorticity='vorticity')
    ReyStress = np.empty((grid.n_points, 6))
    ReyStress[:,0] = grid['Mean Momentum Flux X'][:,0] - grid['Mean Momentum '][:,0]**2/grid['Mean Density']
    ReyStress[:,1] = grid['Mean Momentum Flux YY'][:] - grid['Mean Momentum '][:,1]**2/grid['Mean Density']
    ReyStress[:,2] = grid['Mean Momentum Flux ZZ'][:] - grid['Mean Momentum '][:,2]**2/grid['Mean Density']
    ReyStress[:,3] = grid['Mean Momentum Flux X'][:,1] - (grid['Mean Momentum '][:,0]*grid['Mean Momentum '][:,1]) /grid['Mean Density']
    ReyStress[:,4] = grid['Mean Momentum Flux X'][:,2] - (grid['Mean Momentum '][:,0]*grid['Mean Momentum '][:,2]) /grid['Mean Density']
    ReyStress[:,5] = grid['Mean Momentum Flux YZ'][:] - (grid['Mean Momentum '][:,0]*grid['Mean Momentum '][:,2]) /grid['Mean Density']
    grid['ReynoldsStress'] = ReyStress

    dataBlocks[key]['wall'] = dataBlocks[key]['wall'].sample(grid)
    dataBlocks[key]['grid'] = grid

#%% Get Celltypes for MTW

for key, dataBlock in dataBlocks.items():
    if key == 'jhtdb': continue
    grid = dataBlock['grid']
    grid['CellTypes'] = grid.celltypes.astype(float)
    dataBlocks[key]['grid'] = grid.cell_data_to_point_data()

#%% Get Boundary layer heights
wall_dists = vpt.getGeometricSeries(0.2, 5E-6, 1.02)
for key, dataBlock in dataBlocks.items():
    if key == 'jhtdb': continue
    wall = dataBlock['wall']
    points = wall.points
    points[-1,0] = points[-1,0]*1.00001
    sample = vpt.sampleAlongVectors(dataBlock, wall_dists, wall['Normals'], wall.points)
    velInt = vpt.delta_velInt(sample['Mean Velocity '][:,0],
                              sample['WallDistance'], wall.points.shape[0], momentum=True, displace=True)
    wall['delta_momentum_vel'] = velInt['delta_momentum']
    wall['delta_displace_vel'] = velInt['delta_displace']

    vortInt = vpt.delta_vortInt(sample['vorticity'][:,2],
                              sample['WallDistance'], wall.points.shape[0], momentum=True, displace=True, returnUvort=True)
    wall['delta_momentum_vort'] = vortInt['delta_momentum']
    wall['delta_displace_vort'] = vortInt['delta_displace']
    wall['delta_percent'] = vpt.delta_percent(sample['Mean Velocity '][:,0], sample['WallDistance'], wall.points.shape[0], percent=0.995)
    wall['Re_theta'] = wall['delta_momentum_vel']*Uref/nu


#%% Get Cf and UTau
for key, dataBlock in dataBlocks.items():
    if key == 'jhtdb': continue
    wall_dists = np.linspace(0, 4.194e-4, 5)
    wall = dataBlock['wall']
    points = wall.points
    nwallpnts = points.shape[0]
    points[-1,0] = points[-1,0]*1.00001
    sample = vpt.sampleAlongVectors(dataBlock, wall_dists, wall['Normals'], wall.points)

    U = sample['Mean Velocity '][:,0].reshape(nwallpnts, -1)
    # stencil = np.array([-11/7, 3, -3/2, 1/3]) / np.diff(wall_dists)[0] # 3rd order accurate FD stencil
    stencil = np.array([-25/12, 4, -3, 4/3, -1/4]) / np.diff(wall_dists)[0] # 4rd order accurate FD stencil

    dudn = np.einsum('ij,j->i', U, stencil)
    tau_w = dudn*nu*rho
    # dataBlocks[key]['wall']['Cf'] = tau_w / (0.5*rho*Uref**2) #Finite Difference way
    Smits = lambda x: 0.024*x**(-1/4)
    # dataBlocks[key]['wall']['Cf'] = Smits(dataBlocks[key]['wall']['Re_theta'])
    dataBlocks[key]['wall']['Cf'] = vpt.calcCf(dataBlocks[key]['wall'], Uref, nu, rho)
    dataBlocks[key]['wall']['Utau'] = Uref*np.sqrt(dataBlocks[key]['wall']['Cf']/2)
    dataBlocks[key]['wall']['delta_nu'] = nu/dataBlocks[key]['wall']['Utau']

    # dataBlocks[key]['wall']['Cf_orig'] = vpt.calcCf(dataBlocks[key]['wall'], Uref, nu, rho)

#%% Get profiles

wall_dists = vpt.getGeometricSeries(0.2, 5E-6, 1.02)

def getProfiles(dataBlocksDict, location, profile_wall_dists, Normal=None):
    profiles = {}
    plane = vtk.vtkPlane()
    plane.SetNormal((1, 0, 0))
    plane.SetOrigin((location, 0, 0))
    for key, dataBlock in dataBlocksDict.items():
        profiles[key] = vpt.sampleDataBlockProfile(dataBlock, profile_wall_dists, cutterobj=plane, normal=Normal)

    return profiles

def getProfiles_ReTheta(dataBlocksDict, location, profile_wall_dists, Normal=None):
    profiles = {}
    plane = vtk.vtkPlane()
    plane.SetNormal((1, 0, 0))
    for key, dataBlock in dataBlocksDict.items():
        wall_dists = profile_wall_dists if key!='jhtdb' else jhtdb_wall_dists
        wall = dataBlock['wall']
        xlocation = vpt.pwlinRoots(wall.points[:,0], wall['Re_theta'] - location)

        plane.SetOrigin((xlocation, 0, 0))
        profiles[key] = vpt.sampleDataBlockProfile(dataBlock, wall_dists, cutterobj=plane, normal=Normal)

    return profiles

profilesAtLocs = {}
profilesAtLocs[r'$Re_\theta$ = 1000'] = getProfiles_ReTheta(dataBlocks, 1000, wall_dists)
profilesAtLocs[r'$Re_\theta$ = 1100'] = getProfiles_ReTheta(dataBlocks, 1100, wall_dists)
profilesAtLocs[r'$Re_\theta$ = 1200'] = getProfiles_ReTheta(dataBlocks, 1200, wall_dists)
profilesAtLocs[r'$Re_\theta$ = 1410'] = getProfiles_ReTheta(dataBlocks, 1410, wall_dists)

for profkey, profiles in profilesAtLocs.items():
    for key, profile in profiles.items():
        rotation_tensor = vpt.wallAlignRotationTensor(profile.walldata['Normals'], np.array([0,1,0]))
        if key == 'jhtdb':
            profile['Velocity_Wall'] = vpt.rotateTensor(profile['Velocity'], rotation_tensor)
        else:
            profile['Velocity_Wall'] = vpt.rotateTensor(profile['Mean Velocity '], rotation_tensor)
        profile['ReynoldsStress_Wall'] = vpt.rotateTensor(profile['ReynoldsStress'], rotation_tensor)

profilesAtLocs[r'$Re_\theta$ = 1000']['sch'] = exdata['sch']['1000']
profilesAtLocs[r'$Re_\theta$ = 1000']['wu'] = exdata['wu']['1000']
profilesAtLocs[r'$Re_\theta$ = 1100']['jim'] = exdata['jim']['1100']
profilesAtLocs[r'$Re_\theta$ = 1200']['wu'] = exdata['wu']['1200']
profilesAtLocs[r'$Re_\theta$ = 1410']['sch'] = exdata['sch']['1410']
profilesAtLocs[r'$Re_\theta$ = 1410']['wu'] = exdata['wu']['1410']
# profilesAtLocs[r'$Re_\theta$ = 1410']['spa'] = exdata['spa']['1410']

#%% Plot profiles

dataBlockPlotKwargs = {
    'Q1':{'linestyle':'solid' ,'color':'red', 'label':r'$Q_1$'},
    'Q2': {'linestyle':'solid' ,'color':'green', 'label':r'$Q_2$'},
    'Q3': {'linestyle':'solid' ,'color':'blue', 'label':r'$Q_3$'},
    'jhtdb': {'linestyle': (0,(1,4)) ,'color':'magenta', 'label':r'Zaki 2013', 'zorder':1, 'linewidth':4.0},
    'jim':   {'linestyle': (0,(1,4)), 'color':'darkorange', 'label':'Jimenez 2010', 'zorder':1, 'linewidth':4.0},
    'sch':   {'linestyle': (0,(1,4)), 'color':'cyan', 'label':'Schlatter 2010', 'zorder':1, 'linewidth':4.0},
    'wu':    {'linestyle': (0,(1,4)), 'color':'grey', 'label':'Wu 2017', 'zorder':1, 'linewidth':4.0},
}

LogLawPlotKwargs = {'linestyle': 'dotted', 'color':'k', 'label':r'$\ln (y^+)/0.41 + 5$'}
STGInflowPlotKwargs = {'linestyle': 'dotted', 'color':'k', 'label':'STG Inflow'}

exdataScatterKwargs = {
    'jhtdb': {'marker':'d', 'facecolor':'None', 'linewidth':1.5},
    'jim':   {'marker':'s', 'facecolor':'None', 'linewidth':1.5},
    'sch':   {'marker':'^', 'facecolor':'None', 'linewidth':1.5},
    'wu':    {'marker':'o', 'facecolor':'None', 'linewidth':1.5},
}

for key in exdataScatterKwargs:
    exdataScatterKwargs[key].update(
        {'edgecolor': dataBlockPlotKwargs[key]['color'],
         'label': dataBlockPlotKwargs[key]['label'],
         'zorder': dataBlockPlotKwargs[key]['zorder'],
         }
    )


#%% Plot Velocity
Re_Theta_regex = r'Re_\\theta'

fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1

    for key in dataBlockPlotKwargs:

        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance+']
            y = profile['Velocity'][:,0]/profile.walldata['Utau']
        elif key == 'wu':
            x = profile['y+']['U+'][:,0]
            y = profile['y+']['U+'][:,1]
        # elif key == 'spa':
        #     x = profile['mean']['y+']
        #     y = profile['mean']['U+']
        elif key == 'jhtdb':
            x = profile['WallDistance']/profile.walldata['delta_nu']
            y = profile['Velocity'][:,0]/profile.walldata['Utau']
        else:
            x = profile['WallDistance']/profile.walldata['delta_nu']
            y = profile['Mean Velocity '][:,0]/profile.walldata['Utau']

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    yplus = np.geomspace(3, 300, 10)
    ax.plot(yplus, np.log(yplus)/0.41 + 5, **LogLawPlotKwargs)

    ax.legend()
    ax.set_xlabel(r'$n^+$')
    ax.set_ylabel(r"$u^+$")
    ax.set_title(profilekey)
    ax.set_xlim((0.3, 10**3))
    ax.set_xscale('log')
    plt.tight_layout()

#%% Plot stresses - u'v'

fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1

    # cellvert = None
    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance+']
            y = -profile['ReynoldsStress'][:,3]/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y/delta']['uv+'][:,0]
            y = profile['y/delta']['uv+'][:,1]
        # elif key == 'spa':
        #     x = profile['mean']['y/delta']
        #     y = profile['mean']['uv+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance']/profile.walldata['delta_nu']
            y = -profile['ReynoldsStress_Wall'][:,3]/profile.walldata['Utau']**2
            # if re.search('.*MTW_6-15.*', key):
            #     cellvert = vpt.pwlinRoots(x, profile['CellTypes']-5.1)
            #     ax.axvline(cellvert, label='MTW 6-15 Element Transition')
            # elif re.search('.*MTW.*3-5.*', key):
            #     cellvert = vpt.pwlinRoots(x, profile['CellTypes']-5.1)
            #     ax.axvline(cellvert, label='MTW 3-5 Element Transition', linestyle='-.')

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    # ax.axvline(cellvert, label='MTW 6-15 Element Transition')
    ax.legend()
    # ax.set_xlabel(r'$n^+$')
    ax.set_xlabel(r'$n/\delta$')
    # ax.set_ylabel(r"$\langle u'v' \rangle$")
    ax.set_ylabel(r"$-\langle u'v' \rangle / u_\tau^2$")
    # ax.set_xscale('log')
    ax.set_xlim((0,1.2))
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot stresses - u'u'
fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1

    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance+']
            y = profile['ReynoldsStress'][:,0]/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y/delta']['uu+'][:,0]
            y = profile['y/delta']['uu+'][:,1]
        elif key == 'spa':
            x = profile['mean']['y/delta']
            y = profile['mean']['uu+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance']/profile.walldata['delta_nu']
            y = profile['ReynoldsStress_Wall'][:,0]/profile.walldata['Utau']**2
            # if re.search('.*MTW.*6-15.*', key):
            #     cellvert = vpt.pwlinRoots(x, profile['CellTypes']-6)

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    # ax.axvline(cellvert, label='MTW 6-15 Element Transition')
    ax.legend()
    # ax.set_xlabel(r'$n^+$')
    ax.set_xlabel(r'$n/\delta$')
   # ax.set_ylabel(r"$\langle u'u' \rangle$")
    ax.set_ylabel(r"$\langle u'u' \rangle / u_\tau^2$")
    # ax.set_xscale('log')
    ax.set_xlim((0,1.2))
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot stresses - u'u' Re 1410 only
fig, ax = plt.subplots()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    if profilekey != r'$Re_\theta$ = 1410': continue
    # ax = axs[i]
    # i+=1

    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance+']
            y = profile['ReynoldsStress'][:,0]/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y/delta']['uu+'][:,0]
            y = profile['y/delta']['uu+'][:,1]
        elif key == 'spa':
            x = profile['mean']['y/delta']
            y = profile['mean']['uu+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance']/profile.walldata['delta_nu']
            y = profile['ReynoldsStress_Wall'][:,0]/profile.walldata['Utau']**2
            # if re.search('.*MTW.*6-15.*', key):
            #     cellvert = vpt.pwlinRoots(x, profile['CellTypes']-6)

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    # ax.axvline(cellvert, label='MTW 6-15 Element Transition')
    ax.legend()
    # ax.set_xlabel(r'$n^+$')
    ax.set_xlabel(r'$n/\delta$')
   # ax.set_ylabel(r"$\langle u'u' \rangle$")
    ax.set_ylabel(r"$\langle u'u' \rangle / u_\tau^2$")
    # ax.set_xscale('log')
    ax.set_xlim((0,1.2))
    ax.set_title(profilekey)
    plt.tight_layout()



#%% Plot stresses - u'u' Log
fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1


    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance+']
            y = profile['ReynoldsStress'][:,0]/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y+']['uu+'][:,0]
            y = profile['y+']['uu+'][:,1]
        elif key == 'spa':
            x = profile['mean']['y+']
            y = profile['mean']['uu+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_nu']
            y = profile['ReynoldsStress_Wall'][:,0]/profile.walldata['Utau']**2

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    ax.legend()
    ax.set_xlabel(r'$n^+$')
    # ax.set_xlabel(r'$n/\delta$')
   # ax.set_ylabel(r"$\langle u'u' \rangle$")
    ax.set_ylabel(r"$\langle u'u' \rangle / u_\tau^2$")
    ax.set_xscale('log')
    ax.set_xlim((0.3,10**2.8))
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot stresses - v'v'
fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1

    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance+']
            y = profile['ReynoldsStress'][:,1]/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y/delta']['vv+'][:,0]
            y = profile['y/delta']['vv+'][:,1]
        elif key == 'spa':
            x = profile['mean']['y/delta']
            y = profile['mean']['vv+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            y = profile['ReynoldsStress_Wall'][:,1]/profile.walldata['Utau']**2

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    # ax.axvline(cellvert, label='MTW 6-15 Element Transition')
    ax.legend()
    # ax.set_xlabel(r'$n^+$')
    ax.set_xlabel(r'$n/\delta$')
   # ax.set_ylabel(r"$\langle u'u' \rangle$")
    ax.set_ylabel(r"$\langle v'v' \rangle / u_\tau^2$")
    # ax.set_xscale('log')
    ax.set_xlim((0,1.2))
    ax.set_title(profilekey)
    plt.tight_layout()


#%% Plot stresses - w'w'
fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1

    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance+']
            y = profile['ReynoldsStress'][:,2]/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y/delta']['ww+'][:,0]
            y = profile['y/delta']['ww+'][:,1]
        elif key == 'spa':
            x = profile['mean']['y/delta']
            y = profile['mean']['ww+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            y = profile['ReynoldsStress_Wall'][:,2]/profile.walldata['Utau']**2

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    # ax.axvline(cellvert, label='MTW 6-15 Element Transition')
    ax.legend()
    # ax.set_xlabel(r'$n^+$')
    ax.set_xlabel(r'$n/\delta$')
   # ax.set_ylabel(r"$\langle u'u' \rangle$")
    ax.set_ylabel(r"$\langle w'w' \rangle / u_\tau^2$")
    # ax.set_xscale('log')
    ax.set_xlim((0,1.2))
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot stresses - TKE
fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1

    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            # x = profile['WallDistance+']
            y = 0.5*profile['ReynoldsStress'][:,:3].sum(axis=1)/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y/delta']['vv+'][:,0]
            y = 0.5*(profile['y/delta']['uu+'][:,1] + profile['y/delta']['vv+'][:,1] + \
                     profile['y/delta']['ww+'][:,1])
        elif key == 'spa':
            x = profile['mean']['y/delta']
            y = profile['mean']['vv+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_percent']
            y = 0.5*profile['ReynoldsStress'][:,:3].sum(axis=1)/profile.walldata['Utau']**2
            # y = profile['ReynoldsStress_Wall'][:,1]/profile.walldata['Utau']**2

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    # ax.axvline(cellvert, label='MTW 6-15 Element Transition')
    ax.legend()
    # ax.set_xlabel(r'$n^+$')
    ax.set_xlabel(r'$n/\delta$')
   # ax.set_ylabel(r"$\langle u'u' \rangle$")
    ax.set_ylabel(r"$\langle k \rangle / u_\tau^2$")
    # ax.set_xscale('log')
    ax.set_xlim((0,1.2))
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot stresses - TKE Log
fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs = axs.flatten()
i=0
for profilekey, profiles in profilesAtLocs.items():
    # fig, ax = plt.subplots()
    ax = axs[i]
    i+=1

    for key in dataBlockPlotKwargs:
        if key not in profiles.keys(): continue

        profile = profiles[key]
        if key in ['jim', 'sch']:
            x = profile['WallDistance+']
            y = 0.5*profile['ReynoldsStress'][:,:3].sum(axis=1)/profile.walldata['Utau']**2
        elif key == 'wu':
            x = profile['y+']['vv+'][:,0]
            y = 0.5*(profile['y+']['uu+'][:,1] + profile['y+']['vv+'][:,1] + \
                     profile['y+']['ww+'][:,1])
        elif key == 'spa':
            x = profile['mean']['y/delta']
            y = profile['mean']['vv+']
        else:
            x = profile['WallDistance']/profile.walldata['delta_nu']
            y = 0.5*profile['ReynoldsStress'][:,:3].sum(axis=1)/profile.walldata['Utau']**2
            # y = profile['ReynoldsStress_Wall'][:,1]/profile.walldata['Utau']**2

        ax.plot(x, y, **dataBlockPlotKwargs[key])

    # ax.axvline(cellvert, label='MTW 6-15 Element Transition')
    ax.legend()
    ax.set_xlabel(r'$n^+$')
    ax.set_ylabel(r"$\langle k \rangle / u_\tau^2$")
    ax.set_xscale('log')
    ax.set_xlim((0.3,10**2.8))
    ax.set_title(profilekey)
    plt.tight_layout()

#%% Plot Boundary Layers vs x
fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys() or key=='jhtdb': continue
    ax.plot(dataBlocks[key]['wall'].points[:removeindex,0], dataBlocks[key]['wall']['delta_percent'][:removeindex], **dataBlockPlotKwargs[key])

ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\delta$')
fig.tight_layout()

fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys() or key=='jhtdb': continue
    ax.plot(dataBlocks[key]['wall'].points[:removeindex,0], dataBlocks[key]['wall']['delta_displace_vel'][:removeindex], **dataBlockPlotKwargs[key])

ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\delta^*$')
fig.tight_layout()

fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys() or key=='jhtdb': continue
    ax.plot(dataBlocks[key]['wall'].points[:removeindex,0], dataBlocks[key]['wall']['delta_momentum_vel'][:removeindex], **dataBlockPlotKwargs[key])

ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\delta_\theta$')
fig.tight_layout()

fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys() or key=='jhtdb': continue
    ax.plot(dataBlocks[key]['wall'].points[:removeindex,0], dataBlocks[key]['wall']['Re_theta'][:removeindex], **dataBlockPlotKwargs[key])

ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r"$Re_\theta$")
fig.tight_layout()


#%% Plot Boundary Layers vs Re_theta
fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys() or key=='jhtdb': continue
    ax.plot(dataBlocks[key]['wall']['Re_theta'][:removeindex], dataBlocks[key]['wall']['delta_percent'][:removeindex], **dataBlockPlotKwargs[key])

ax.legend()
ax.set_xlabel(r"$Re_\theta$")
ax.set_ylabel(r'$\delta$')
fig.tight_layout()

fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys() or key=='jhtdb': continue
    ax.plot(dataBlocks[key]['wall']['Re_theta'][:removeindex], dataBlocks[key]['wall']['delta_displace_vel'][:removeindex], **dataBlockPlotKwargs[key])

ax.legend()
ax.set_xlabel(r"$Re_\theta$")
ax.set_ylabel(r'$\delta^*$')
fig.tight_layout()

fig, ax = plt.subplots()
removeindex = -1
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys(): continue
    if key == 'jhtdb':
        H = dataBlocks[key]['wall']['delta_displace'][:removeindex] / dataBlocks[key]['wall']['delta_momentum'][:removeindex]
    else:
        H = dataBlocks[key]['wall']['delta_displace_vel'][:removeindex] / dataBlocks[key]['wall']['delta_momentum_vel'][:removeindex]
    ax.plot(dataBlocks[key]['wall']['Re_theta'][:removeindex], H, **dataBlockPlotKwargs[key])

for key, data in exdata.items():
    if key in ['jim', 'sch']:
        re_thetas = []
        Hs = []
        for re_theta in exdata[key].keys():
            if re_theta == 'global': continue
            Hs.append(data[re_theta].walldata['delta_displace'] / data[re_theta].walldata['delta_momentum'])
            re_thetas.append(int(re_theta))

        ax.scatter(re_thetas, Hs,**exdataScatterKwargs[key])

ax.legend()
ax.set_xlabel(r"$Re_\theta$")
ax.set_ylabel(r'$H_{12}$')
fig.tight_layout()

#%% Calc Cf

fig, ax = plt.subplots()
removeindex = -10
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys(): continue
    ax.plot(dataBlocks[key]['wall']['Re_theta'][:removeindex], dataBlocks[key]['wall']['Cf'][:removeindex], **{**dataBlockPlotKwargs[key], 'label': dataBlockPlotKwargs[key]['label'] + ' FD'})

# for key in dataBlockPlotKwargs.keys():
#     if key not in dataBlocks.keys(): continue
#     if key == 'jhtdb': continue
#     ax.plot(dataBlocks[key]['wall']['Re_theta'][:removeindex], dataBlocks[key]['wall']['Cf_orig'][:removeindex], **{**dataBlockPlotKwargs[key], 'linestyle':'dashed', 'label': dataBlockPlotKwargs[key]['label'] + ' VTK'})

key = list(dataBlocks.keys())[0]
Retheta_example = np.linspace(dataBlocks[key]['wall']['Re_theta'].min()*0.95,
                              dataBlocks[key]['wall']['Re_theta'].max()*1.05, 1000)
Smits = lambda _: 0.024*Retheta_example**(-1/4)
ax.plot(Retheta_example, Smits(Retheta_example), linestyle='solid', color='black', label='Smits et. al. 1983')
ax.plot(Retheta_example, Smits(Retheta_example)*1.05, linestyle='dashed', color='black', label=r'_Smits $\pm$5%')
ax.plot(Retheta_example, Smits(Retheta_example)*0.95, linestyle='dashed', color='black', label='_Smits')

# ax.scatter([855, 1150, 1450], [0.00464, 0.00426, 0.00398], marker='s',
           # facecolor='None', edgecolor='black', linewidth=0.7, label='Coles 1962')
ax.scatter([1150, 1450], [0.00426, 0.00398], marker='s',
           facecolor='None', edgecolor='black', linewidth=0.7, label='Coles 1962')
ax.scatter([935, 1421], [0.00426, 0.00388], marker='d',
           facecolor='None', edgecolor='black', linewidth=0.7, label='Schlatter 2011')

ax.legend()
ax.set_xlabel(r'$Re_\theta$')
ax.set_ylabel(r"$C_f$")
fig.tight_layout()

fig, ax = plt.subplots()
removeindex = -10
for key in dataBlockPlotKwargs.keys():
    if key not in dataBlocks.keys() or key=='jhtdb': continue
    ax.plot(dataBlocks[key]['wall'].points[:removeindex,0], dataBlocks[key]['wall']['Cf'][:removeindex], **dataBlockPlotKwargs[key])

ax.legend()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r"$C_f$")
fig.tight_layout()
