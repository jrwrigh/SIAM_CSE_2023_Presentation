
import numpy as np
import vtk
from pathlib import Path
import pyvista as pv
import vtkpytools as vpt

#%% ---- File inputs ----
files = list(Path('.').glob('stats*.cgns'))

vtmPath = Path('./stats.vtm')

grids = []

for file in files:
    reader = pv.CGNSReader(file)
    output = reader.read()
    grids.append(output['Spanwise_Stats']['Zone'])

grid = grids[0]

for name in grid.point_data.keys():
    grid[name] += grids[1][name]
    grid[name] *= 0.5

def actuallyOrderPolyData(poly: vtk.vtkPolyData, func) -> pv.PolyData:
    sort_order = func(poly)
    new_poly = pv.lines_from_points(poly.points[np.argsort(sort_order),:])
    new_poly = new_poly.sample(poly)
    return new_poly

    # Get wall edge from grid
featedges = grid.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                       feature_edges=False, manifold_edges=False)

    # Second input is point internal to domain, helping ensure that wall normals point inward
featedges = vpt.computeEdgeNormals(featedges, np.array([-0.42, 0.2, 0]))

    # Extract out cells based on their normals, may need adjustment per case
wall = featedges.extract_cells(np.arange(featedges.n_cells)[featedges['Normals'][:,1] > 0.8])
wall = vpt.unstructuredToPoly(wall.cell_data_to_point_data())
wall = actuallyOrderPolyData(wall, lambda poly: poly.points[:,0])
wall = wall.strip()

# At this point you have two objects:
#   - grid: 2D grid object for the mesh
#   - wall: A line representing the wall

dataBlock = pv.MultiBlock()
dataBlock['grid'] = grid
dataBlock['wall'] = wall
dataBlock.save(vtmPath)
