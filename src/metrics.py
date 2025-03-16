import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def calculate_pairwise_distances(data, entry_list, ligands):
    distances = []
    for df, name in zip(data, entry_list):
        distance_matrix = pd.DataFrame(squareform(pdist(df, metric='euclidean')))
        selected_rows = [distance_matrix.iloc[i, i+1:].values for i in range(len(distance_matrix)-1)]
        distance_df = pd.DataFrame(selected_rows).stack()
        distances.append(pd.DataFrame([distance_df.values], index=[name], columns=ligands))
    return pd.concat(distances)

def calculate_inertial_radii(data, entry_list):
    coords = {axis: [df[axis].values for df in data] for axis in ['x', 'y', 'z']}
    ligands = {axis: pd.DataFrame(coords[axis], index=entry_list) for axis in ['x', 'y', 'z']}
    radii = np.sqrt(
        ligands['x'].var(ddof=0, axis=1) +
        ligands['y'].var(ddof=0, axis=1) +
        ligands['z'].var(ddof=0, axis=1)
    )
    return radii

def calculate_inertial_radii_alt(data, atom_list):
    squared_sum = np.sum(data ** 2, axis=1)
    radii = np.sqrt(squared_sum) / len(atom_list)
    return radii

