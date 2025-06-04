import numpy as np
import re
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# --- Type Aliases ---
NodeDict = Dict[int, Tuple[float, float]]
NodeSetDict = Dict[str, List[int]]
BCDict = Dict[int, List[str]]


# ===================================
# 1. File I/O: Parsing Abaqus .inp
# ===================================

def parse_inp_file(file_path: str) -> Tuple[NodeDict, NodeSetDict, BCDict]:
    """
    Parse Abaqus .inp file to extract nodes, node sets, and boundary conditions.

    Args:
        file_path: Path to the .inp file.

    Returns:
        node_dict: {node_id: (x, y)}
        nsets: {set_name: [node_ids]}
        bc_by_dof: {dof_number: [set_names]}
    """
    with open(file_path, 'r') as file:
        data = file.read()

    node_dict = {}
    in_node_block = False

    # Extract nodes
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("*node"):
            in_node_block = True
            continue
        if in_node_block and line.startswith("*"):
            break
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                node_id = int(parts[0].strip())
                x, y = float(parts[1]), float(parts[2])
                node_dict[node_id] = (x, y)
            except ValueError:
                continue

    # Extract node sets
    nsets = {}
    nset_pattern = re.compile(r"(?i)\*Nset.*(?:nset|name)=(\S+)[^\n]*\n((?:[\d,\s]+)+)")
    for match in nset_pattern.finditer(data):
        header, node_block = match.group(0).split('\n', 1)
        set_name = match.group(1).strip(',').strip()
        node_ids = []
        for line in node_block.splitlines():
            tokens = [t.strip() for t in line.split(',') if t.strip()]
            if "generate" in header.lower() and len(tokens) == 3 and all(tok.isdigit() for tok in tokens):
                start, end, step = map(int, tokens)
                node_ids.extend(range(start, end + 1, step))
            else:
                node_ids.extend(map(int, filter(str.isdigit, tokens)))
        nsets[set_name] = node_ids

    # Extract boundary conditions
    bc_by_dof = {1: [], 2: []}
    lines = data.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.lower().startswith("*boundary"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("*"):
                parts = lines[i].strip().split(',')
                if len(parts) >= 2:
                    set_name = parts[0].strip()
                    try:
                        dof = int(parts[1])
                        if dof in bc_by_dof:
                            bc_by_dof[dof].append(set_name)
                    except ValueError:
                        pass
                i += 1
        else:
            i += 1

    return node_dict, nsets, bc_by_dof



# ===================================
# 4. Stacking and Visualization
# ===================================


def staking_geometry(domain_points: np.ndarray, nset_coords: dict) -> Tuple[np.ndarray, dict]:
    """
    Stack domain points and boundary set points, remove duplicates,
    and create index arrays for each set.

    Args:
        domain_points (np.ndarray): Array of domain coordinates.
        nset_coords (dict): Dictionary of set names to arrays of coordinates.

    Returns:
        stacked_coords (np.ndarray): Stacked array of all unique coordinates.
        index_arrays (dict): Dictionary of set names (safe variable style) to their index arrays.
                             For example: idx_domain, idx_Set_1, idx_External, etc.
    """
    all_data = []
    set_indices = {}
    start_idx = 0

    # Prepare all data in a single dictionary
    input_dict = {'domain': domain_points}
    input_dict.update(nset_coords)  # Add all boundary sets like External, Internal, etc.

    # 1. Stack all data first
    for name, coords in input_dict.items():
        all_data.append(coords)

    stacked_coords = np.vstack(all_data)

    # 2. Remove duplicate points
    # View points as structured array to enable row-wise uniqueness
    stacked_coords_view = stacked_coords.view([('x', stacked_coords.dtype), ('y', stacked_coords.dtype)])
    unique_coords, unique_indices = np.unique(stacked_coords_view, return_index=True)

    # Get the cleaned stacked_coords
    stacked_coords = stacked_coords[unique_indices]

    # 3. Now rebuild index arrays for each set
    current_idx = 0
    index_arrays = {}

    for name, coords in input_dict.items():
        safe_name = name.replace('-', '_').replace(' ', '_')

        if coords.size == 0:
            index_arrays[f'idx_{safe_name}'] = np.array([], dtype=int)
            continue

        coords_view = coords.view([('x', coords.dtype), ('y', coords.dtype)])
        unique_view = stacked_coords.view([('x', stacked_coords.dtype), ('y', stacked_coords.dtype)])

        # For each point in coords, find its index in the unique stacked coords
        idx_array = np.nonzero(np.isin(unique_view, coords_view))[0]

        index_arrays[f'idx_{safe_name}'] = idx_array

    return stacked_coords, index_arrays




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_geometry_with_normals(domain_points: np.ndarray,
                                nset_coords: dict,
                                outer_normals: np.ndarray = None,
                                inner_normals: np.ndarray = None,
                                figsize: tuple = (12, 10)):
    """
    Plot domain points, boundary sets, and boundary normals.

    Args:
        domain_points (np.ndarray): (N, 2) array of domain points.
        nset_coords (dict): Dictionary of boundary sets {set_name: (N_i, 2) arrays}.
        outer_normals (np.ndarray): (M, 4) array [x, y, nx, ny] for outer boundary normals.
        inner_normals (np.ndarray): (P, 4) array [x, y, nx, ny] for inner boundary normals.
        figsize (tuple): Figure size.
    """
    palette = sns.color_palette('tab20', len(nset_coords))
    
    plt.figure(figsize=figsize)

    # Plot domain points
    if domain_points is not None and len(domain_points) > 0:
        plt.scatter(domain_points[:, 0], domain_points[:, 1],
                    color='black', s=10, marker='o', label='Domain Points')

    # Plot nset_coords (Set-1, Set-2, etc.) except External and Internal
    for i, (name, coords) in enumerate(nset_coords.items()):
        if name not in ['External', 'Internal']:  # Skip External and Internal
            plt.scatter(coords[:, 0], coords[:, 1],
                        s=30, marker='x',
                        color=palette[i % len(palette)],
                        label=name)

    # Helper to plot normals
    def _plot_normals(points_normals, normal_color, normal_label):
        x, y = points_normals[:, 0], points_normals[:, 1]
        nx, ny = points_normals[:, 2], points_normals[:, 3]
        plt.quiver(x, y, nx, ny,
                   color=normal_color, scale=30, width=0.0025,
                   headwidth=3, headlength=4, alpha=0.8,
                   label=normal_label)

    # Plot outer boundary normals (No scatter points, only quiver)
    if outer_normals is not None:
        _plot_normals(outer_normals, normal_color='darkgreen', normal_label='Outer Normals')

    # Plot inner boundary normals
    if inner_normals is not None:
        _plot_normals(inner_normals, normal_color='darkorange', normal_label='Inner Normals')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Domain, Boundary Sets, and Boundary Normals')
    plt.legend(fontsize='small', markerscale=1.2, loc='best')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()





# ===================================
# 5. Data Augmentation
# ===================================

def repeat_points(coords: np.ndarray, repeat_factor: int) -> np.ndarray:
    """
    Repeat points for data augmentation.
    """
    if coords is None or coords.size == 0:
        return np.empty((0, coords.shape[1]))
    return np.repeat(coords, repeat_factor, axis=0)


def augment_boundary_nodes(bc_u_coords_sets: Dict[str, np.ndarray],
                            bc_v_coords_sets: Dict[str, np.ndarray],
                            outer_points_and_normals: np.ndarray = None,
                            inner_points_and_normals: np.ndarray = None,
                            repeat_factor: int = 10) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Repeat boundary condition nodes and free traction nodes for training augmentation.

    Args:
        bc_u_coords_sets: BC nodes for DOF 1 (u).
        bc_v_coords_sets: BC nodes for DOF 2 (v).
        outer_points_and_normals: Outer boundary points and normals (N, 4).
        inner_points_and_normals: Inner boundary points and normals (N, 4).
        repeat_factor: Number of times to repeat.

    Returns:
        repeated_bc_sets: Dictionary of repeated BC node sets.
        repeated_outer_boundary: Repeated outer boundary nodes (free traction).
        repeated_inner_boundary: Repeated inner boundary nodes (free traction).
    """
    unique_bc_sets = set(bc_u_coords_sets) | set(bc_v_coords_sets)
    repeated_bc_sets = {}

    for set_name in sorted(unique_bc_sets):
        # Safer fetching:
        coords_set = bc_u_coords_sets.get(set_name, None)
        if coords_set is None:
            coords_set = bc_v_coords_sets.get(set_name, None)

        safe_set_name = set_name.replace('-', '_').replace(' ', '_')
        general_name = f"Repeated_BC_{safe_set_name}"
        repeated_bc_sets[general_name] = repeat_points(coords_set, repeat_factor)

    repeated_outer_boundary = repeat_points(outer_points_and_normals[:, :2] if outer_points_and_normals is not None else None, repeat_factor)
    repeated_inner_boundary = repeat_points(inner_points_and_normals[:, :2] if inner_points_and_normals is not None else None, repeat_factor)

    return repeated_bc_sets, repeated_outer_boundary, repeated_inner_boundary



# ===================================
# 6. Indices Assignment and Data Stacking
# ===================================

def stack_input_coords(domain_coords: np.ndarray,
                       repeated_bc_sets: Dict[str, np.ndarray],
                       repeated_inner_boundary: np.ndarray,
                       repeated_outer_boundary: np.ndarray) -> np.ndarray:
    """
    Stack domain and boundary coordinates into a single input array.
    """
    bc_stack_list = [repeated_bc_sets[name] for name in sorted(repeated_bc_sets)]
    input_coords = np.vstack((domain_coords, *bc_stack_list, repeated_inner_boundary, repeated_outer_boundary))
    return input_coords


def assign_indices(domain_coords: np.ndarray,
                   repeated_bc_sets: Dict[str, np.ndarray],
                   repeated_inner_boundary: np.ndarray,
                   repeated_outer_boundary: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Assign index arrays to domain, BCs, and boundaries.
    """
    idx_domain_coords = np.arange(domain_coords.shape[0])
    idx_bc_sets = {}
    start = idx_domain_coords[-1] + 1

    for name in sorted(repeated_bc_sets):
        end = start + repeated_bc_sets[name].shape[0]
        idx_bc_sets[name] = np.arange(start, end)
        start = end

    end_inner = start + repeated_inner_boundary.shape[0]
    idx_inner_boundary = np.arange(start, end_inner)
    start_outer = end_inner
    end_outer = start_outer + repeated_outer_boundary.shape[0]
    idx_outer_boundary = np.arange(start_outer, end_outer)

    return idx_domain_coords, idx_bc_sets, idx_inner_boundary, idx_outer_boundary


def prepare_normals(domain_coords: np.ndarray,
                    repeated_bc_sets: Dict[str, np.ndarray],
                    inner_points_and_normals: np.ndarray,
                    outer_points_and_normals: np.ndarray,
                    repeat_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare normal vectors for domain and boundary points.
    """
    zeros_domain = np.zeros((domain_coords.shape[0], 1))
    nx_input_bc_sets = [np.zeros((repeated_bc_sets[name].shape[0], 1)) for name in sorted(repeated_bc_sets)]
    ny_input_bc_sets = [np.zeros((repeated_bc_sets[name].shape[0], 1)) for name in sorted(repeated_bc_sets)]

    nx_input_inner = np.repeat(inner_points_and_normals[:, 2:3], repeat_factor, axis=0)
    ny_input_inner = np.repeat(inner_points_and_normals[:, 3:4], repeat_factor, axis=0)

    nx_input_outer = np.repeat(outer_points_and_normals[:, 2:3], repeat_factor, axis=0)
    ny_input_outer = np.repeat(outer_points_and_normals[:, 3:4], repeat_factor, axis=0)

    nx_input = np.vstack((zeros_domain, *nx_input_bc_sets, nx_input_inner, nx_input_outer))
    ny_input = np.vstack((zeros_domain, *ny_input_bc_sets, ny_input_inner, ny_input_outer))

    return nx_input, ny_input


def prepare_training_data(domain_coords: np.ndarray,
                           repeated_bc_sets: Dict[str, np.ndarray],
                           repeated_inner_boundary: np.ndarray,
                           repeated_outer_boundary: np.ndarray,
                           inner_points_and_normals: np.ndarray,
                           outer_points_and_normals: np.ndarray,
                           repeat_factor: int = 10):
    """
    Prepare the full training dataset for SciANN model.
    """
    input_coords = stack_input_coords(domain_coords, repeated_bc_sets, repeated_inner_boundary, repeated_outer_boundary)
    idx_domain_coords, idx_bc_sets, idx_inner_boundary, idx_outer_boundary = assign_indices(
        domain_coords, repeated_bc_sets, repeated_inner_boundary, repeated_outer_boundary
    )

    nx_input, ny_input = prepare_normals(
        domain_coords, repeated_bc_sets,
        inner_points_and_normals, outer_points_and_normals,
        repeat_factor
    )

    x_input = input_coords[:, 0:1]
    y_input = input_coords[:, 1:2]

    return x_input, y_input, nx_input, ny_input, idx_domain_coords, idx_bc_sets, idx_inner_boundary, idx_outer_boundary


# ===================================
# 8. Visualization: Training Loss and Field Predictions
# ===================================

def plot_training_loss(history, target_labels):
    """
    Plot training loss components from SciANN training history.
    """
    component_loss_keys = [k for k in history.history if k.endswith('_loss') and k != 'loss']

    if len(component_loss_keys) != len(target_labels):
        raise ValueError(f"Mismatch: {len(target_labels)} targets vs {len(component_loss_keys)} losses.")

    ordered_loss_keys = [k for k in history.history.keys() if k in component_loss_keys]

    plt.figure(figsize=(12, 6))
    if 'loss' in history.history:
        plt.semilogy(history.history['loss'], label='Total Loss', linewidth=2.5, color='black')

    colors = plt.cm.tab20.colors

    for i, (key, label) in enumerate(zip(ordered_loss_keys, target_labels)):
        plt.semilogy(history.history[key], label=f'{label}_loss', linewidth=2, color=colors[i % len(colors)])

    plt.title('Training Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend(ncol=3, fontsize='small')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def evaluate_and_plot_fields(x_input, y_input, functionals, titles, nx_input=None, ny_input=None,
                             figsize=(20, 4), cmap='jet'):
    """Evaluate SciANN functionals and plot field predictions."""
    inputs = [x_input, y_input]
    if nx_input is not None and ny_input is not None:
        inputs = [x_input, y_input, nx_input, ny_input]

    predictions = [func.eval(inputs) for func in functionals]
    fig, axs = plt.subplots(1, len(functionals), figsize=figsize)

    for ax, values, title in zip(axs, predictions, titles):
        sc = ax.scatter(x_input, y_input, c=values.flatten(), cmap=cmap, s=20, edgecolor='none')
        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.colorbar(sc, ax=ax, shrink=0.85)

    plt.tight_layout()
    plt.show()



# === Full Random Seed Initialization ===
import os
import numpy as np
import random
import tensorflow as tf


def set_random_seed(seed=42):
    """
    Set random seed for full reproducibility.
    Args:
        seed (int): Random seed value.
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow (SciANN is built on TF backend)
    tf.random.set_seed(seed)
    
    # OS level (hash seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Random seed set to {seed} for reproducibility.")




import numpy as np
from typing import Dict, Tuple, Any

import numpy as np
from typing import Dict, Tuple, Any

def stack_and_get_indices(merge_same_keys: bool = True, **kwargs) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    """
    Stack multiple coordinate arrays and return the stacked array along with the
    inclusive index ranges for each input.

    Accepts any number of keyword arguments, where each value can be:
    - a np.ndarray
    - a dict of np.ndarrays (multiple named sets)

    If `merge_same_keys` is True, sets with the same sub-key name are merged.

    Returns:
        stacked_coords: np.ndarray
        indices: Dict[str, Tuple[int, int]]
    """
    all_data = []
    indices = {}
    current_idx = 0

    # For merging sets with the same name
    merged_sets = {}

    for key, value in kwargs.items():
        if isinstance(value, dict):
            # It's a dict of sets
            for sub_key, arr in value.items():
                full_key = f"{key}:{sub_key}"

                if merge_same_keys:
                    # Merge sub_keys with same name
                    if sub_key not in merged_sets:
                        merged_sets[sub_key] = arr
                    else:
                        # Check if they are identical
                        if not np.allclose(merged_sets[sub_key], arr):
                            raise ValueError(f"Conflict: Sub-key '{sub_key}' appears in multiple inputs but has different data.")
                        # If identical, no action needed (already added)
                else:
                    # No merging — treat fully qualified names separately
                    merged_sets[full_key] = arr

        elif isinstance(value, np.ndarray):
            # Single array (e.g., domain, external_coords)
            merged_sets[key] = value

        else:
            raise TypeError(f"Unsupported type {type(value)} for key '{key}'. Must be np.ndarray or dict of np.ndarray.")

    # Now process the merged sets
    for set_name, arr in merged_sets.items():
        all_data.append(arr)
        start_idx = current_idx
        end_idx = current_idx + len(arr) - 1  # Inclusive end
        indices[set_name] = (start_idx, end_idx)
        current_idx = end_idx + 1

    # Stack everything vertically
    stacked_coords = np.vstack(all_data)

    return stacked_coords, indices






def get_set_coords(nodes: dict, nsets: dict) -> dict:
    """
    Match node IDs from nsets with coordinates from nodes and 
    return a dictionary of set names to 2D arrays of coordinates.

    Args:
        nodes (dict): Dictionary of node IDs to (x, y) coordinates.
        nsets (dict): Dictionary of set names to list of node IDs.

    Returns:
        dict: Dictionary mapping set names to numpy arrays of coordinates.
    """
    nset_coords = {}

    for set_name, node_ids in nsets.items():
        coords_list = [nodes[nid] for nid in node_ids if nid in nodes]
        if coords_list:  # Only add if there are matching nodes
            nset_coords[set_name] = np.array(coords_list)

    return nset_coords



def get_domain_coords(nodes: dict, nset_coords: dict) -> np.ndarray:
    """
    Subtract the coordinates in nset_coords from all nodes to get domain points.

    Args:
        nodes (dict): Dictionary of node IDs to (x, y) coordinates.
        nset_coords (dict): Dictionary of set names to arrays of coordinates.

    Returns:
        np.ndarray: Array of domain coordinates.
    """
    # Stack all boundary points from nsets
    boundary_points = np.vstack(list(nset_coords.values())) if nset_coords else np.empty((0, 2))

    # Get all node coordinates
    all_nodes = np.array(list(nodes.values()))

    if boundary_points.size > 0:
        # View for row-wise comparison
        all_nodes_view = all_nodes.view([('x', 'f8'), ('y', 'f8')])
        boundary_points_view = boundary_points.view([('x', 'f8'), ('y', 'f8')])

        # Mask: True where node is NOT in boundary points
        mask = ~np.isin(all_nodes_view, boundary_points_view)

        # Fix the shape issue
        domain_points = all_nodes[mask.ravel()]
    else:
        domain_points = all_nodes

    return domain_points



def constant_array(value: float, length: int, dtype: str = 'float32') -> np.ndarray:
    """
    Create an array of a constant value.

    Args:
        value (float): The value to fill the array with.
        length (int): Length of the array.
        dtype (str): Data type of the array.

    Returns:
        np.ndarray: Array filled with the constant value.
    """
    return np.full((length, 1), value, dtype=dtype)



def plot_stacked_coords_custom(stacked_coords: np.ndarray,
                                index_arrays: dict,
                                figsize: tuple = (10, 8)):
    """
    Plot stacked coordinates highlighting each component separately.

    Args:
        stacked_coords (np.ndarray): Stacked coordinate array (N, 2).
        index_arrays (dict): Dictionary with index arrays for each component.
        figsize (tuple): Size of the figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create color palette
    palette = sns.color_palette('tab10', n_colors=len(index_arrays))

    plt.figure(figsize=figsize)

    for i, (name, idx_array) in enumerate(index_arrays.items()):
        # Extract the points for this block
        block_coords = stacked_coords[idx_array]
        label_name = name.replace('idx_', '')  # cleaner name for plot

        plt.scatter(block_coords[:, 0], block_coords[:, 1],
                    s=20, marker='o',
                    color=palette[i % len(palette)],
                    label=label_name)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Stacked Geometry Components')
    plt.legend(fontsize='small')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



from scipy.spatial import cKDTree
import numpy as np

def traction_normals(nset_coords: dict,
                     boundary_name: str,
                     flip: bool = False,
                     reorder: bool = True,
                     tol: float = 1e-8) -> np.ndarray:
    """
    Prepare traction normals for a given boundary ('External' or 'Internal'),
    optionally reordering points and removing points overlapping with other BCs.

    Args:
        nset_coords (dict): Dictionary of set names to arrays of coordinates.
                            Must include 'External' and 'Internal' keys.
        boundary_name (str): 'External' or 'Internal' — the boundary to compute normals for.
        flip (bool): Flip normals (True for Internal boundaries).
        reorder (bool): Whether to reorder points to minimize jumps.
        tol (float): Tolerance for KDTree matching.

    Returns:
        np.ndarray: (M, 4) array of [x, y, nx, ny] after BC exclusion.
    """
    def reorder_points(coords: np.ndarray) -> np.ndarray:
        coords = coords.copy()
        n = len(coords)
        visited = np.zeros(n, dtype=bool)
        order = [0]
        visited[0] = True

        for _ in range(1, n):
            last = order[-1]
            dists = np.linalg.norm(coords - coords[last], axis=1)
            dists[visited] = np.inf
            next_idx = np.argmin(dists)
            order.append(next_idx)
            visited[next_idx] = True

        return coords[order]

    def compute_normals(coords: np.ndarray, outward: bool, closed: bool = False) -> np.ndarray:
        n = coords.shape[0]
        tangents = np.zeros_like(coords)

        for i in range(n):
            if closed:
                tangent = coords[(i + 1) % n] - coords[(i - 1) % n]
            else:
                if i == 0:
                    tangent = coords[i + 1] - coords[i]
                elif i == n - 1:
                    tangent = coords[i] - coords[i - 1]
                else:
                    tangent = coords[i + 1] - coords[i - 1]
            tangents[i] = tangent

        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

        centroid = np.mean(coords, axis=0)
        vectors_to_centroid = coords - centroid
        dot_products = np.einsum('ij,ij->i', normals, vectors_to_centroid)

        mask = dot_products > 0 if outward else dot_products < 0
        normals[mask] = -normals[mask]

        if not closed:
            coords = coords[1:-1]
            normals = normals[1:-1]

        return np.hstack((coords, normals))

    # ---- Start Process ----

    # 1. Get boundary points
    boundary_coords = nset_coords[boundary_name]

    # 2. Stack all other BC points
    bc_points = []
    for key, coords in nset_coords.items():
        if key != boundary_name:
            bc_points.append(coords)
    bc_points_array = np.vstack(bc_points) if bc_points else np.empty((0, 2))

    # 3. (Optional) Reorder boundary points
    if reorder:
        boundary_coords = reorder_points(boundary_coords)

    # 4. Compute normals
    boundary_normals = compute_normals(boundary_coords, outward=not flip, closed=False)

    boundary_coords_normals = boundary_normals[:, :2]
    normals = boundary_normals[:, 2:]

    # 5. KDTree matching to remove BC points
    if bc_points_array.size > 0:
        tree = cKDTree(bc_points_array)
        dists, _ = tree.query(boundary_coords_normals, k=1)
        mask = dists > tol
    else:
        mask = np.ones(boundary_coords_normals.shape[0], dtype=bool)

    clean_coords = boundary_coords_normals[mask]
    clean_normals = normals[mask]

    return np.hstack((clean_coords, clean_normals))
