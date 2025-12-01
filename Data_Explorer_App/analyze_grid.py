import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.utils.nominals import PHANTOM_CB_NOMINALS

points = []
labels = []
for k, v in PHANTOM_CB_NOMINALS.items():
    if k.startswith('CB'):
        points.append([v['x'], v['y'], v['z']])
        labels.append(k)

points = np.array(points)
n_points = len(points)

print("Nearest Neighbors Analysis:")
for i in range(n_points):
    dists = np.linalg.norm(points - points[i], axis=1)
    # Get indices of 3 nearest neighbors (excluding self)
    # argsort returns indices that sort the array. 0 is self (dist 0)
    nearest_indices = np.argsort(dists)[1:4]
    
    neighbors = []
    for idx in nearest_indices:
        neighbors.append(f"{labels[idx]} ({dists[idx]:.2f})")
    
    print(f"{labels[i]}: {', '.join(neighbors)}")
