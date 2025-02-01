import numpy as np
from typing import Optional, Dict, Tuple, List
from scipy.stats import lognorm
import pandas as pd
# from custom_tools_max.master_config import LOG_SHAPE,LOG_SCALE

def traffic_jam_p(minute):

    hour = minute / 60 + 6 # shift start at 6am
    # Average traffic jam durations (weighted average across all cities)
    avg_duration = 15.7  # Weighted mean of the 'Average duration' column from the table

    # Severity based on time of day (normalized from graphs)
    if 0 <= hour < 6:
        p_cong = 0.05  # Low traffic at night
    elif 6 <= hour < 9:
        p_cong = 0.2 / 3 # Morning peak
    elif 9 <= hour < 12:
        p_cong = 0.1/ 3# Midday decline
    elif 12 <= hour < 16:
        p_cong = 0.15/4 # Gradual rise in the afternoon
    elif 16 <= hour < 20:
        p_cong = 0.3/ 4 # Evening peak
    elif 20 <= hour <= 24:
        p_cong = 0.1  # Decline after rush hour
    else:
        raise ValueError("Hour must be between 0 and 24.")
    
    return p_cong

from scipy.stats import truncnorm   
def generate_single_traffic_jam_duration():
    """
    Generate a single traffic jam duration based on the provided distribution summary.

    Returns:
        float: A single generated traffic jam duration.
    """
    # Distribution parameters from the dataset description
    if np.random.rand() > float(1/3):
        mean_duration = 21.23
        std_duration = 18.92
        min_duration = 10
        max_duration = 74.7
        
    else:
        mean_duration = 6
        std_duration = 5
        min_duration = 1
        max_duration = 10
    
    # Define the bounds for truncation
    a, b = (min_duration - mean_duration) / std_duration, (max_duration - mean_duration) / std_duration

    # Generate a single sample
    duration = truncnorm.rvs(a, b, loc=mean_duration, scale=std_duration, size=1, random_state=None)
    
    return int(duration[0])


def update_time_matrix_dynamic(
        T: np.ndarray,
        montufar_block=None,
        modification_history: List[Tuple[np.ndarray, np.ndarray, int]] = None,
        simulation_period: int = 0,
        logshape = 0.5,
        logscale = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray, int]]]:
    """
    Update the travel time matrix while avoiding re-modification of recently increased values.

    Args:
        T (array-like): A 2D array representing the travel time matrix.
        montufar_block (array-like, optional): Indices to exclude from modification.
        modification_history (list, optional): History of modified indices and deltas with periods.
        simulation_period (int): The current simulation period.
        arc_travel_increase (int): The amount to increase travel time by.
        probability (float): Probability of modifying an arc.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray, int]]]: 
            - Updated travel time matrix.
            - Modified indices in the current cycle.
            - Updated modification history.
    """
    T = np.array(T)

    # Initialize modifiable mask
    modifiable_mask = np.ones(T.shape, dtype=bool)

    # Exclude montufar_block indices
    if montufar_block is not None:
        montufar_block = np.array(montufar_block).astype(int)
        modifiable_mask[montufar_block[:, 0], montufar_block[:, 1]] = False

    if modification_history is None:
        modification_history = []

    # Reverse changes added 3 or more periods ago
    # print("[update_time_matrix_dynamic] Reverse changes...")
    removed = []
    for i, (indices, values, last_period) in enumerate(modification_history):
        if simulation_period - last_period >= 0:
            T[indices[:, 0], indices[:, 1]] -= values
            removed.append(i)

    # Remove reversed entries from history
    modification_history = [entry for j, entry in enumerate(modification_history) if j not in removed]

    # Exclude recently modified entries from being modified again
    for indices, _, _ in modification_history:
        modifiable_mask[indices[:, 0], indices[:, 1]] = False

    # Apply random modifications
    random_mask = np.random.rand(*T.shape) > float(1.0 - traffic_jam_p(simulation_period))
    random_mask[np.arange(T.shape[0]), np.arange(T.shape[0])] = False  # Exclude diagonal
    final_mask = random_mask & modifiable_mask

    U_shape = logshape  # Controls the peak 0.5 std
    U_scale = logscale # Controls the spread 0.5 std
    U_m = lognorm.rvs(U_shape, scale=U_scale, size=T.shape) + 1
    U_m = np.clip(U_m, 1.0, 5) # experimental line ensure feasibility NOTE
    # print(pd.DataFrame(U_m))

    # Apply probabilistic updates to the time matrix
    T_delta = np.where(final_mask, np.int64(T * U_m), T)

    # Track modified indices and deltas
    modified_indices_ndarray = np.argwhere(final_mask)
    deltas_values_ndarray = np.int64((U_m[final_mask]) * T[final_mask] - T[final_mask])

    delete_in_period = simulation_period + generate_single_traffic_jam_duration()
    modification_history.append((modified_indices_ndarray, deltas_values_ndarray, delete_in_period))

    return T_delta, modified_indices_ndarray, modification_history

def Psi(
        current_node_index,
        CumDelay = np.array([0,0], dtype='int64'),
        origin_index = False,
        t_mins_vehicle_adapted: dict = {},  
        solution_vrptw: tuple = None,
        T_orig = None,
        T_delta = None,
        ):
    """
    Function to recursively check a vehicles' route for feasibility regarding
    increased arc travel times.

    Returns:
    - True, disrupted_node in case of disruption of the route
    - False, None in case of a feasible route
    """
    mgr  = solution_vrptw[0]
    rt = solution_vrptw[1]
    sol = solution_vrptw[2]
    # print("GetDimensionOrDie")
    time_dim = rt.GetDimensionOrDie("Time")
    # print("check")
    # if rt.IsStart(current_node_index):
        # print(f"[Psi] Start...{current_node_index}")
    if rt.IsEnd(current_node_index):

        return False, None, None, t_mins_vehicle_adapted

    # Get the node index from routing manager
    current_node = mgr.IndexToNode(current_node_index)
    next_node_index = sol.Value(rt.NextVar(current_node_index))
    next_node = mgr.IndexToNode(next_node_index)
    time_var_current = time_dim.CumulVar(current_node_index)
    time_var_next = time_dim.CumulVar(next_node_index)
    slack_var = time_dim.SlackVar(current_node_index)

    # Calculate travel time differences
    old_travel_time = np.int64(T_orig[current_node][next_node])
    # Montufar test
    if current_node_index != origin_index:
        new_travel_time = np.int64(T_delta[current_node][next_node])
    else:
        new_travel_time = old_travel_time
    delta_t_ij = new_travel_time - old_travel_time
    
    # Get times and slacks
    sol_min = sol.Min(time_var_current)
    sol_max = sol.Max(time_var_current)
    times = np.array([sol_min,sol_max], dtype='int64')
    # print(f"sol.Min(slack_var)...idx({current_node_index})node({current_node})")
    s_min = sol.Min(slack_var)
    # print("sol.Min(slack_var) check")
    # print(f"sol.Max(slack_var)...idx({current_node_index})node({current_node})")
    s_max = sol.Max(slack_var)
    # print("sol.Max(slack_var) check")

    slacks = np.array([s_min,s_max], dtype='int64')
    
    # NOTE time window infringements are checked by previous node
    tmin_std, tmax_std = times[0], times[1]
    tmin_new = tmin_std + np.int64(CumDelay[0])
    t_mins_vehicle_adapted[current_node] = tmin_new
    remaining_internal_buffer = tmax_std - tmin_new
    tmax_new = tmax_std - min(remaining_internal_buffer, delta_t_ij)# doesn't matter atm
    
    # influence of previous node
    slacks_1 = slacks - CumDelay
    
    # influence of the following arc travel time change
    slacks_2 = slacks_1 - np.array([delta_t_ij, delta_t_ij], dtype='int64')
    
    # Adapt transferred CumDelay 
    # = influence on next nodes tmin
    estimated_arrival = tmin_new + new_travel_time
    CumTransfer = max(0, estimated_arrival - np.int64(sol.Min(time_var_next)))
    # print("sol.Min(time_var_next)")
    
    if (slacks_2[1] < np.int64(0)): 
        return True, current_node, min(tmax_std,tmin_new), t_mins_vehicle_adapted

    # if node is second to last, consider the last node's time window end instead of local slack
    # elif rt.IsEnd(next_node_index):
    #     return False, None, None, t_mins_vehicle_adapted
    
    # Recurse to the next node
    else:
        return Psi(
            next_node_index,
            CumDelay = np.array([CumTransfer,CumTransfer], dtype='int64'),
            origin_index = origin_index,
            t_mins_vehicle_adapted = t_mins_vehicle_adapted,
            solution_vrptw = solution_vrptw,
            T_orig = T_orig,
            T_delta = T_delta,
            )