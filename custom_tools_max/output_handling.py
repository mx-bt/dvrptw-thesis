import time
import platform
# import cpuinfo
from typing import Dict, List, Tuple, Any
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes

def get_routes_dict(solution, routing, manager):
    routes = {}
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
    
        routes[route_nbr] = route
    return routes

def make_initial_routes_from_dict(routes_dict, nodes_to_exclude):
    """For initial VRPTW kickstart"""
    initial_routes = []
    exclude = nodes_to_exclude
    ir_exist = False
    for vehicle_id, route_list in routes_dict.items():

        cut_route = [s for s in route_list if s not in exclude]
        ir_exist = True
        initial_routes.append(cut_route)

    if ir_exist:
        initial_routes
    else:
        return False
    return initial_routes

def get_all_arcs_np(data, manager, routing, solution):
    """Returns all travelled arcs as a numpy array."""

    all_travelled_arcs = []

    for vehicle_id in range(data["num_vehicles"]):

        index = routing.Start(vehicle_id)
        route_arcs = []  # To store arcs for the current vehicle
        
        # Iterate over the nodes in the route
        while not routing.IsEnd(index):
            start_node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            end_node = manager.IndexToNode(next_index)

            route_arcs.append((start_node, end_node))
            index = next_index
        if (manager.IndexToNode(routing.Start(vehicle_id)) != manager.IndexToNode(routing.End(vehicle_id))) or len(route_arcs)>2:
            all_travelled_arcs.extend(route_arcs)
    
    # Convert the list of arcs to a numpy array and return
    return np.array(all_travelled_arcs)

# [START] PLOT REQS for EGADE 
def iterate_nodes_dynamically(
    current_node: int,
    start_node: int,
    result_nodes_dict: dict[int, int],
    distance_matrix: list[list[float]],
    distance_sum: float = 0,
    all_nodes: list[list[int]] = None
    ) -> list[list[int]]:
    
    """
    Iteratively traverses nodes based on the provided `result_nodes_dict` and calculates
    the cumulative distance cost between them using the `distance_matrix`. The function 
    prints the current node, the next node, the distance cost between them, and the cumulative 
    total cost dynamically. This process continues until it returns to the `start_node`.

    Args:
        current_node (int): The current node being evaluated.
        start_node (int): The node to which the traversal should eventually return.
        result_nodes_dict (dict[int, int]): A mapping of each node to its corresponding next node.
        distance_matrix (list[list[float]]): A 2D matrix representing the distances between nodes.
        distance_sum (float, optional): The cumulative distance cost. Defaults to 0.
        all_nodes (list[list[int]], optional): A list storing the traversed node pairs. Defaults to None.

    Returns:
        list[list[int]]: A list of pairs representing the traversed nodes and their next nodes.
    """
    
    all_nodes = [] if all_nodes == None else all_nodes
    next_node = result_nodes_dict[current_node]
    distance_cost = distance_matrix[current_node][next_node]
    
    print_output = str(current_node).ljust(5)
    print_output += " -> "
    print_output += str(next_node).rjust(5)

    table_width = 15

    print_header = True if distance_sum == 0 else False

    if print_header:
        header_output = 'Node'.ljust(len(print_output))

    
    all_nodes.append([current_node,next_node])

    print_output += str(round(distance_cost/10)).rjust(table_width)

    if print_header:
        header_output += 'Arc Cost'.rjust(table_width)

    distance_sum += distance_cost/10
    print_output += str(round(distance_sum)).rjust(table_width)

    if print_header:
        header_output += 'Total Cost'.rjust(table_width)

    # Exclude "empty routes"
    if start_node != next_node:
        print(header_output) if print_header else None
        print(print_output)

        return iterate_nodes_dynamically(
            next_node,
            start_node,
            result_nodes_dict,
            distance_matrix,
            distance_sum = distance_sum,
            all_nodes=all_nodes)
    # elif len(all_nodes) < 2:
    #     return None
    else:
        print(header_output) if print_header else None
        print(print_output)      
        return all_nodes

def print_nodes_nicely(
        route,
        distance_matrix,
        transform_route_to_nodes = False
        ):
    """
    Use `transform_route_to_nodes` to handle different route data structures

    Example 1:
        >>> routing = [0,5,8,2,4,...]

        >>> print_nodes_nicely(..., transform_route_to_nodes = True)

    Example 2:
        >>> routing = [(0,5),(5,8),(8,2),(2,4)...]
        >>> print_nodes_nicely(..., transform_route_to_nodes = False)

    Returns:
        A list which can be used for CSV printing.
        >>> result = print_nodes_nicely(...)
        >>> result
        []
    """

    if transform_route_to_nodes:
        result_nodes_list = [
            (route[i],route[i+1]) for i in range(len(route[:-1]))
            ]
    else:
        result_nodes_list = route
    
    result_nodes_dict = {node[0]:node[1] for node in result_nodes_list}
    # print(result_nodes_dict)   
 
    # TODO if nodes do not need to be transformed (e.g. MIP result) then this step 
    # is FUCKING extremely redundant. Dynamic route traverse only needed for plain
    # unordered route lists ! Will scale horribly for larger problem instances
    start_node_s = 0
    current_node_s = start_node_s
    n = iterate_nodes_dynamically(
        current_node_s,
        start_node_s,
        result_nodes_dict,
        distance_matrix
        )
    
    return n

def measure_time(func):
    """
    Wrapper function to measure another functions' execution time

    Example:
        >>> @measure_time
        >>> def another_function(*args, **kwargs):
        >>> ...
        >>> another_function()
        "Some return of another_function"
        "Time taken: 1 hour(s), 2 minute(s), 3 second(s)"

    Returns:
        - regular function outputs
        - printout of the functions kernel time
    """

    def format_size(size_in_bytes):
    # Convert sizes to more readable formats (KB, MB, etc.)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024
    

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_delta = end_time - start_time

        hours, rem = divmod(time_delta, 3600)
        minutes, seconds = divmod(rem, 60)



        print(f"\nSolver time: {int(hours)} hour(s), {int(minutes)} minute(s), {seconds:.2f} second(s)")
        # cpu_info = cpuinfo.get_cpu_info()
        # print(f"Processor: {cpu_info['brand_raw']}")
        # print(f"Architecture: {platform.machine()}")
        # print(f"Cores: {cpu_info['count']}")
        # print(f"Vendor: {cpu_info['vendor_id_raw']}")
        # print(f"L2 Cache Size: {format_size(cpu_info['l2_cache_size'])}")
        # print(f"L3 Cache Size: {format_size(cpu_info.get('l3_cache_size', 'N/A'))}")
        return result
    
    return wrapper

def save_nodes_OD(header_dict, all_routes_raw, start_node = False, end_node = False):
    """
    Inputs
        - `header_dict` {0: "depot", 1: "Customer X", 2: "Customer Y"...]
        - `all_routes_raw` [[0,1],[1,2]...]

    # TODO currently only configured for lucila data!
    # """
    # useCustom = False
    # if start_node or end_node:
    #     sn = start_node
    #     en = end_node
    #     useCustom = True # TODO end this ...

    all_routes_OD_form = []
    for route in all_routes_raw:
        if route:
            for node in route:
                if (node[0] == 0) and (node[1] == 0):
                    all_routes_OD_form.append([f'O . D'])
                elif node[0] == 0:
                    all_routes_OD_form.append([f'O . {header_dict[node[1]]}'])
                elif node[1] == 0:
                    all_routes_OD_form.append([f'{header_dict[node[0]]} . D'])
                else:
                    all_routes_OD_form.append([f'{header_dict[node[0]]} . {header_dict[node[1]]}'])

    for od in all_routes_OD_form:
        # print(od)
        pass
    
    return all_routes_OD_form

# [END] PLOT REQS for EGADE 

def get_first_solution_strategy(ref: int):
    """From https://developers.google.com/optimization/routing/routing_options"""
    solution_description = {
        0: {
            "name": "UNSET",
            "description": "See the homonymous value in LocalSearchMetaheuristic."
        },
        15: {
            "name": "AUTOMATIC",
            "description": """Lets the solver detect which strategy to use according to the model being solved."""
        },
        3: {
            "name": "PATH_CHEAPEST_ARC",
            "description": """--- Path addition heuristics --- 
            Starting from a route "start" node, connect it to the node which produces
            the cheapest route segment, then extend the route by iterating on the
            last node added to the route."""
        },
        4: {
            "name": "PATH_MOST_CONSTRAINED_ARC",
            "description": """Same as PATH_CHEAPEST_ARC, but arcs are evaluated with a comparison-based
            selector which will favor the most constrained arc first."""
        },
        5: {
            "name": "EVALUATOR_STRATEGY",
            "description": """Same as PATH_CHEAPEST_ARC, except that arc costs are evaluated using the
            function passed to RoutingModel::SetFirstSolutionEvaluator()."""
        },
        10: {
            "name": "SAVINGS",
            "description": """Savings algorithm (Clarke & Wright).
            Reference: Clarke, G. & Wright, J.W.:
            "Scheduling of Vehicles from a Central Depot to a Number of Delivery
            Points", Operations Research, Vol. 12, 1964, pp. 568-581"""
        },
        11: {
            "name": "SWEEP",
            "description": """Sweep algorithm (Wren & Holliday).
            Reference: Anthony Wren & Alan Holliday: Computer Scheduling of Vehicles
            from One or More Depots to a Number of Delivery Points Operational
            Research Quarterly (1970-1977), Vol. 23, No. 3 (Sep., 1972), pp. 333-344"""
        },
        13: {
            "name": "CHRISTOFIDES",
            "description": """Christofides algorithm (actually a variant of the Christofides algorithm
            using a maximal matching instead of a maximum matching, which does
            not guarantee the 3/2 factor of the approximation on a metric travelling
            salesman)."""
        },
        6: {
            "name": "ALL_UNPERFORMED",
            "description": """--- Path insertion heuristics ---
            Make all nodes inactive. Only finds a solution if nodes are optional (are
            element of a disjunction constraint with a finite penalty cost)."""
        },
        7: {
            "name": "BEST_INSERTION",
            "description": """Iteratively build a solution by inserting the cheapest node at its
            cheapest position; the cost of insertion is based on the global cost
            function of the routing model."""
        },
        8: {
            "name": "PARALLEL_CHEAPEST_INSERTION",
            "description": """Iteratively build a solution by inserting the cheapest node at its
            cheapest position; the cost of insertion is based on the arc cost
            function. Is faster than BEST_INSERTION."""
        },
        14: {
            "name": "SEQUENTIAL_CHEAPEST_INSERTION",
            "description": """Iteratively build a solution by constructing routes sequentially, for
            each route inserting the cheapest node at its cheapest position until the
            route is completed; the cost of insertion is based on the arc cost
            function. Is faster than PARALLEL_CHEAPEST_INSERTION."""
        },
        9: {
            "name": "LOCAL_CHEAPEST_INSERTION",
            "description": """Iteratively build a solution by inserting each node at its cheapest
            position; the cost of insertion is based on the arc cost function.
            Differs from PARALLEL_CHEAPEST_INSERTION by the node selected for
            insertion; here nodes are considered in decreasing order of distance to
            the start/ends of the routes, i.e. farthest nodes are inserted first."""
        },
        16: {
            "name": "LOCAL_CHEAPEST_COST_INSERTION",
            "description": """Same as LOCAL_CHEAPEST_INSERTION except that the cost of insertion is
            based on the routing model cost function instead of arc costs only."""
        },
        1: {
            "name": "GLOBAL_CHEAPEST_ARC",
            "description": """--- Variable-based heuristics ---
            Iteratively connect two nodes which produce the cheapest route segment."""
        },
        2: {
            "name": "LOCAL_CHEAPEST_ARC",
            "description": """Select the first node with an unbound successor and connect it to the
            node which produces the cheapest route segment."""
        },
        12: {
            "name": "FIRST_UNBOUND_MIN_VALUE",
            "description": """Select the first node with an unbound successor and connect it to the
            first available node."""
        }
    }
    return solution_description[ref]

def get_local_search_metaheuristic(ref: int):
    metaheuristic_description = {
        0: {
            "name": "UNSET",
            "description": """Means "not set". If the solver sees that, it'll behave like for AUTOMATIC. 
            But this value won't override others upon a proto MergeFrom(), whereas "AUTOMATIC" will."""
        },
        6: {
            "name": "AUTOMATIC",
            "description": "Lets the solver select the metaheuristic."
        },
        1: {
            "name": "GREEDY_DESCENT",
            "description": """Accepts improving (cost-reducing) local search neighbors until a local
            minimum is reached."""
        },
        2: {
            "name": "GUIDED_LOCAL_SEARCH",
            "description": """Uses guided local search to escape local minima
            (cf. http://en.wikipedia.org/wiki/Guided_Local_Search); this is generally
            the most efficient metaheuristic for vehicle routing."""
        },
        3: {
            "name": "SIMULATED_ANNEALING",
            "description": """Uses simulated annealing to escape local minima
            (cf. http://en.wikipedia.org/wiki/Simulated_annealing)."""
        },
        4: {
            "name": "TABU_SEARCH",
            "description": """Uses tabu search to escape local minima
            (cf. http://en.wikipedia.org/wiki/Tabu_search)."""
        },
        5: {
            "name": "GENERIC_TABU_SEARCH",
            "description": """Uses tabu search on a list of variables to escape local minima. 
            The list of variables to use must be provided via the SetTabuVarsCallback callback."""
        }
    }
    return metaheuristic_description[ref]

def get_routes_and_times(data: Dict[str, Any], manager: Any, routing: Any, solution: Any) -> Dict[int, List[Dict[str, int]]]:
    """
    Extracts the valid routes and their associated time windows for each vehicle from the routing solution.

    Args:
        data (Dict[str, Any]): A dictionary containing the problem data, including the number of vehicles and time windows.
            - "num_vehicles" (int): The number of vehicles in the problem.
            - "time_windows" (List[Tuple[int, int]]): A list of time windows for each node, where each tuple represents 
              the opening and closing time of a node.
        manager (Any): The index manager from OR-Tools, which maps between node indices and node IDs.
        routing (Any): The routing model from OR-Tools, which handles the constraints and routes.
        solution (Any): The solution object from OR-Tools, which contains the routes and values (like time windows) for each vehicle.

    Returns:
        Dict[int, List[Dict[str, int]]]: A dictionary where the keys are vehicle IDs, and the values are lists of dictionaries
        representing each node in the route along with its time window details:
            - "node_id" (int): The ID of the node visited by the vehicle.
            - "time_min" (int): The earliest time the vehicle can arrive at the node (min time in the solution).
            - "time_max" (int): The latest time the vehicle can leave the node (max time in the solution).
            - "time_window_open" (int): The opening time of the time window for this node.
            - "time_window_close" (int): The closing time of the time window for this node.


    This function only includes routes where a vehicle is actually moving from one node to another. If a vehicle's route starts and ends at the same node without visiting any others, it is excluded from the output.
    """
    all_routes_and_times = {}

    time_dimension = routing.GetDimensionOrDie("Time")
    for vehicle_id in range(data["num_vehicles"]):
        vehicle_route_and_times = []

        index = routing.Start(vehicle_id)
        
        while not routing.IsEnd(index):
            
            # Access variables
            
            time_var = time_dimension.CumulVar(index)
            node_id = manager.IndexToNode(index)

            if node_id in data["depot_node_ids"]:
                slack_min = 0
                slack_max = data["shift_duration"]
            else:
                slack_var = time_dimension.SlackVar(index)
                slack_min = solution.Min(slack_var)
                slack_max = solution.Max(slack_var)

            # Next node considerations
            next_node_id = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            actual_travel_time = data['time_matrix'][node_id][next_node_id]
            # estimated_departure = solution.Min(time_dimension.CumulVar(solution.Value(routing.NextVar(index))))-actual_travel_time
            estimated_departure = solution.Min(time_var) # NOTE due to updated policy: immediate departures
            
            vehicle_route_and_times.append(
                {
                    "node_id": node_id,
                    # "node_index": index, NOTE don't that
                    "time_min": solution.Min(time_var),
                    "time_max": solution.Max(time_var),
                    "time_window_open": data["time_windows"][node_id][0],
                    "time_window_close": data["time_windows"][node_id][1],
                    "travel_time_to_next_node": actual_travel_time,
                    "slack_min": slack_min,
                    "slack_max": slack_max,
                    "estimated_departure_time": estimated_departure,
                }
            )
            # Move to the next node
            index = solution.Value(routing.NextVar(index))

        # Process the last node (depot)
        time_var = time_dimension.CumulVar(index)
        slack_var = time_dimension.SlackVar(index)
        node_id = manager.IndexToNode(index)
    
        vehicle_route_and_times.append({
            "node_id": node_id,
            # "node_index": index, NOTE don't that
            "time_min": solution.Min(time_var),
            "time_max": solution.Max(time_var),
            "time_window_open": data["time_windows"][node_id][0],
            "time_window_close": data["time_windows"][node_id][1],
            "travel_time_to_next_node": None,    
            "slack_min": None,
            "slack_max": None, 
            "estimated_departure_time": None,    
        })
        
        # Exclude empty routes
        if (manager.IndexToNode(routing.Start(vehicle_id)) != manager.IndexToNode(routing.End(vehicle_id))) or len(vehicle_route_and_times) > 2:
            all_routes_and_times[vehicle_id] = vehicle_route_and_times

    return all_routes_and_times

def get_next_node_GAMSPy(x_r, n):
    """
    Retrieves the next node given the current node 'n'.
    Returns None if no such node exists.
    """
    matching = x_r[(x_r["level"] == 1.0) & (x_r["i0"] == n)]
    if not matching.empty:
        return matching["j0"].iloc[0]
    return None

def extract_routes_GAMSPy(x_records, starts: List[str] = ["0","0","0","0"],ends=["0","0","0","0"]) -> List[List[int]]:
    """
    Extracts routes starting from nodes specified in the 'starts' list.
    Handles cases where 'starts' may contain 0 (or None) values.
    """
    routes = []
    
    
    # Filter rows where level == 1.0 and i0 is in starts

    filtered_records = x_records[(x_records["level"] == 1.0) & (x_records["i0"].isin(starts))]
    print(filtered_records)
    for _, row in filtered_records.iterrows():
        start_node = row["i0"]
        current_node = row["j0"]
        
        route = [int(start_node)]
        
        sec = len(x_records)
        iteration = 0
        while current_node not in ends and iteration <= sec:
            print(current_node)
            route.append(int(current_node))
            current_node = get_next_node_GAMSPy(x_records, current_node)
            iteration += 1
        if iteration == sec:
            print("Problem with extract_routes_GAMSPy")
            return None
        route.append(int(current_node))

        routes.append(route)

    return routes

def plot_multiple_cycles(all_routes_per_cycle, data_per_cycle, time_of_disruption_per_cycle=None):
    """
    Plot the vehicle routes for multiple cycles in one figure.
    
    Args:
        all_routes_per_cycle (list): A list of all_routes_and_times dictionaries, each representing a cycle.
        data_per_cycle (list): A list of data dictionaries, each representing the data for a cycle.
        time_of_disruption_per_cycle (list): A list of time_of_disruption values, one for each cycle (optional).
    """
    from matplotlib import gridspec

    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    plt.rc('font', family='serif')

    num_cycles = len(all_routes_per_cycle)

    if time_of_disruption_per_cycle:
        time_of_disruption_per_cycle.append(None)

    # Before creating subplots
    height_ratios = [len(routes) for routes in all_routes_per_cycle]
    # Optional: Normalize or set a minimum height to avoid extremely small subplots
    height_ratios = [max(hr, 1) for hr in height_ratios]
        
    # Create a figure with subplots for each cycle
    # fig_height = 6 * num_cycles if num_cycles > 1 else 6
    # fig, axs = plt.subplots(num_cycles, figsize=(12, fig_height), sharex=True, sharey=False)
    fig = plt.figure(figsize=(12, 6 * sum(height_ratios)))
    gs = gridspec.GridSpec(num_cycles, 1, height_ratios=height_ratios)
    axs = [fig.add_subplot(gs[i]) for i in range(num_cycles)]

    
    # Handle the case where there is only one cycle (i.e., axs is not a list)
    # print(axs)
    # if num_cycles == 1:
    #     axs = [axs]
    # print(axs)

    for cycle_idx, (all_routes_and_times, data) in enumerate(zip(all_routes_per_cycle, data_per_cycle)):

        ax = axs[cycle_idx]
        plot_offset = 1
        route_pos = 0

        # Calculate number of vehicles for current cycle
        num_vehicles = len(all_routes_and_times)
        
        # Set y-limits based on number of vehicles
        ax.set_ylim(-plot_offset, num_vehicles)

        for vehicle_id, route in all_routes_and_times.items():
            y = route_pos  # Vehicle ID determines the y position of the graph
            route_pos += 1

            for i, node in enumerate(route):
                # last node remaining time window
                if i + 1 == len(route):
                    adapted_rectangle = plt.Rectangle(
                        (node['time_min'], y - 0.2),  # (x, y)
                        data["shift_duration"] - node['time_min'],  # width
                        0.4,  # height
                        edgecolor=(0.5, 0.5, 0.5, 0.2),
                        facecolor=(0.5, 0.5, 0.5, 0.2)
                    )
                    ax.add_patch(adapted_rectangle)
                    ax.vlines(node['time_min'], ymin=y-0.1, ymax=y+0.1, color='black', linewidth=1)
                    ax.text(node['time_min']+0.3, y + 0.4, f"{node['node_id']}", 
                            ha='center', va='center', color='black', fontsize=7)
                    
                # Depot time windows will not be displayed
                elif node['node_id'] in data["depot_node_ids"]:
                    adapted_rectangle = plt.Rectangle(
                        (0, y - 0.2),  # (x, y)
                        0,  # width
                        0.4,  # height
                        edgecolor=(0.5, 0.5, 0.5, 0.2),
                        facecolor=(0.5, 0.5, 0.5, 0.2)
                    )
                    ax.add_patch(adapted_rectangle)
                    ax.vlines(node['time_min'], ymin=y-0.1, ymax=y+0.1, color='black', linewidth=1)
                    ax.text(node['time_min'], y + 0.4, f"{node['node_id']}", 
                            ha='center', va='center', color='black', fontsize=7)
                else:
                    rect = plt.Rectangle(
                        (node['time_window_open'], y - 0.2),
                        node['time_window_close'] - node['time_window_open'],  # width
                        0.4,
                        edgecolor=(0.5, 0.5, 0.5, 0.2),
                        facecolor=(0.5, 0.5, 0.5, 0.2)
                    )
                    ax.add_patch(rect)
                    ax.vlines(node['time_min'], ymin=y-0.1, ymax=y+0.1, color='black', linewidth=1)
                    
                    ax.text(node['time_min']+0.3, y + 0.4, f"{node['node_id']}",
                            ha='center', va='center', color='black', fontsize=7)

                if i < len(route) - 1:
                    next_node = route[i + 1]

                    # a red arrow means something very, very bad
                    arrow_color = 'red' if int(node['travel_time_to_next_node']) != int(data['time_matrix'][int(node['node_id'])][int(next_node['node_id'])]) else 'black'
                    # print(f"{node['node_id']}->{next_node['node_id']} ", int(node['travel_time_to_next_node']), int(data['time_matrix'][int(node['node_id'])][int(next_node['node_id'])]), " ", arrow_color)
                    ax.annotate("",
                                # xy=(next_node['time_min'], y), xytext=(next_node['time_min']-node['travel_time_to_next_node'], y),
                                xy=(node['time_min'] + node['travel_time_to_next_node'], y), xytext=(node['time_min'], y),
                                arrowprops=dict(arrowstyle="->", lw=1.0, color=arrow_color))

                    ax.vlines(node['time_min'] + node['travel_time_to_next_node'], ymin=y-0.05, ymax=y+0.05, color='black', linewidth=1)

        vehicle_indices = [i for i in all_routes_and_times.keys()]

        if time_of_disruption_per_cycle:
            time_of_disruption = time_of_disruption_per_cycle[cycle_idx+1]

        if time_of_disruption:
            ax.axvline(x=time_of_disruption, color='r', linestyle='-.', linewidth=1)

        ax.set_xlim(-plot_offset, data["shift_duration"] + plot_offset)
        # ax.set_ylim(-plot_offset, len(vehicle_indices))
        ax.set_xlabel(" ")
        if time_of_disruption:
            ax.set_ylabel(
                f"Cycle {cycle_idx+1}\n$t_D$={time_of_disruption}",
                fontsize=10,
                rotation=0,
                ha='left',
                va='center',
                labelpad=50
            )
        else:
            ax.set_ylabel(
                f"Cycle {cycle_idx+1}",
                fontsize=10,
                rotation=0,
                ha='left',
                va='center',
                labelpad=50
            )

        ax.set_yticks(range(len(vehicle_indices)))
        
        ax.set_yticklabels([f"V. {v_id}" for v_id in vehicle_indices]) # True labels
        # ax.set_yticklabels([f"V. {i+1}" for i, v_id in enumerate(vehicle_indices)])

        if cycle_idx == len(all_routes_per_cycle)-1:
            ax.set_xlabel("Time units")
            ax.tick_params(axis='x', which='both', labelbottom=True)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='both', labelbottom=False)


        ax.set_title("")
        ax.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.7)

    # grey_box_patch = mpatches.Patch(
    #     facecolor=(0.5, 0.5, 0.5, 0.2),
    #     edgecolor='none',
    #     label='Time windows'
    # )
    # disruption_line_legend = mlines.Line2D(
    #     [], [], color='red', linestyle='-.', label=r'$t_D$'
    # )
    # driving_arrows_legend = mlines.Line2D(
    #     [], [], color='black', marker=r'$\rightarrow$', label='Driving period'
    # )

    # # Display a single legend for the entire figure
    # fig.legend(
    # handles=[grey_box_patch, disruption_line_legend, driving_arrows_legend],
    # loc='upper right',       # Place legend in the top-right corner
    # bbox_to_anchor=(0.9,1.0),
    # fancybox=True,
    # shadow=False,
    # ncol=1                   # Controls how many columns the legend has
    # )

    # plt.tight_layout(pad=-1.0, w_pad=-1.0, h_pad=-1.0)
    # plt.tight_layout()
    plt.show()

def print_solution_VRPTW(data, manager, routing, solution, t_D = None):
    """Prints solution on console."""
    print("[print_solution_(D)VRPTW] ")
    # Display dropped nodes.

    # due to the montufar assumption (if a vehicle once started
    # a arc traverse, the travel time won't change), it doesn't make
    # sense to continue the simulation, since there CANNOT be any
    # infeasibility anymore TODO must be checked in vehicle status!
    dropped_nodes = ""
    dn = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            dropped_nodes += f" {manager.IndexToNode(node)}"
            if manager.IndexToNode(node) not in data["visited_nodes"]:

                dn.append(manager.IndexToNode(node))
    print("Neglected customer nodes:",dn)
    # print("Dropped valid nodes:",set(dn))
    # print("visited_nodes",data["visited_nodes"])
    # Assuming data["time_matrix"] is a list of lists or a 2D array
    time_matrix_df = pd.DataFrame(data["time_matrix"])
    customers_to_serve = [n for n in range(len(data["time_matrix"])) if n not in data["visited_nodes"]]
    filtered_df = time_matrix_df.iloc[customers_to_serve]
    # print("customers_to_serve",customers_to_serve)
    # print(filtered_df) if dn else None
    # print("")
    
    correct_t_D = t_D if t_D else 0

    time_dimension = routing.GetDimensionOrDie("Time")
    total_time_from_t_D = 0
    total_time = 0
    travel_time_true = 0
    ov_global_ortools = 0
    
    shift_end = data["shift_duration"]
    overtime_count = 0
    overtime_ort = 0
    es, ls = 0,0
    active_vehicles = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_length = 1
        travel_time_vehicle = 0
        while not routing.IsEnd(index):

            time_var = time_dimension.CumulVar(index)
            slack_var = time_dimension.SlackVar(index)

            actual_travel_time = data["time_matrix"][manager.IndexToNode(index)][manager.IndexToNode(solution.Value(routing.NextVar(index)))]
            travel_time_vehicle += actual_travel_time
            plan_output += (
                f"{manager.IndexToNode(index)}({index})"
                f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                f"hrd[{time_var.Min()},{time_var.Max()}]"
                f"sft[{time_dimension.GetCumulVarSoftLowerBound(index)},{time_dimension.GetCumulVarSoftUpperBound(index)}]"
                # f" Slack({solution.Min(slack_var)}, {solution.Max(slack_var)})"
                f" --({actual_travel_time})--> "
            )
            es += max(0,time_dimension.GetCumulVarSoftLowerBound(index)-solution.Min(time_var))
            ls += max(0,solution.Min(time_var)-time_dimension.GetCumulVarSoftUpperBound(index))
            index = solution.Value(routing.NextVar(index))
            route_length += 1
        time_var = time_dimension.CumulVar(index)
        plan_output += (
            f"{manager.IndexToNode(index)}"
            f" Last Node Time({solution.Min(time_var)},{solution.Max(time_var)})"
            f"hrd[{time_var.Min()},{time_var.Max()}]"
            f"sft[{time_dimension.GetCumulVarSoftLowerBound(index)},{time_dimension.GetCumulVarSoftUpperBound(index)}]"
        )
        plan_output += f"\nTime of the route: {solution.Min(time_var)}min\n"
        
        # Exclude empty routes
        if (manager.IndexToNode(routing.Start(vehicle_id)) != manager.IndexToNode(routing.End(vehicle_id))) or route_length > 2:
            # print(plan_output)
        
            # The total route calc must be here:
            # for DVRPTW instances the sol.Min for active vehicles would be added
            # since they are Time(t_D,shift_duration) !
            # for VRPTW not a problem because they are simply Time(0,30) at depot
            # NOTE migh be good for overtime determination
            sol_Min = solution.Min(time_var)
            total_time_from_t_D += sol_Min - correct_t_D
            total_time += sol_Min 
            travel_time_true += travel_time_vehicle
            overtime_count += max(0,sol_Min-shift_end)
            active_vehicles += 1
        else:
            pass
        if route_length > 2:
            ov_global_ortools += travel_time_vehicle
            overtime_ort += max(0,solution.Min(time_var)-time_dimension.GetCumulVarSoftUpperBound(index))

    obj_std = solution.ObjectiveValue()

    if t_D:
        print(f"[Heuristics] Sum of arrival times 0>=t>=t_max: {total_time}min")
        print(f"[Heuristics] Total time of all routes t_D<=t<=t_max: {total_time_from_t_D}min")
        print(f"[Heuristics] Overtime considering shift ({shift_end}): {overtime_count}")

        OV_early_service = es * data["penalties"]["early_service"] * data["lambdas"]["LAMBDA_early_service"]
        OV_late_service = ls * data["penalties"]["late_service"] * data["lambdas"]["LAMBDA_late_service"]
        OV_outsourcing = len(dn) * data["penalties"]["customer_drop"] * data["lambdas"]["LAMBDA_outsourcing_deliveries"]
        added_vehicles = max(0,active_vehicles - len(data["considered_vehicles"]))
        OV_add_vehicle = added_vehicles * data["penalties"]["add_vehicle"] * data["lambdas"]["LAMBDA_utilisation_of_backup_trucks"]
        OV_overtime = overtime_count * data["penalties"]["overtime"] * data["lambdas"]["LAMBDA_overt_time_for_drivers"]
        ov_true = travel_time_true + OV_early_service + OV_late_service + OV_outsourcing + OV_add_vehicle + OV_overtime
        overtime_ort *= data["lambdas"]["LAMBDA_overt_time_for_drivers"]
        ov_global_ortools += overtime_ort * data["penalties"]["overtime"] + OV_early_service + OV_late_service + OV_outsourcing + OV_add_vehicle
        
    else:
        print(f"[Heuristics] Total time of all routes {total_time}min")
        ov_true = travel_time_true

    if t_D:
        obj_adapted = obj_std + (ov_true-ov_global_ortools)

    else:
        obj_adapted = obj_std
        
    # print(f"[Heuristics] Total arc travels: {travel_time_true}min")
    # print(f"[Heuristics] total ES = {es}")
    # print(f"[Heuristics] total LS = {ls}")
    # print(f"[Heuristics] Standard OV: ", obj_std)
    # print(f"[Heuristics] Predicted OV: ", ov_global_ortools)
    print(f"[Heuristics] Adapted Objective Value: {obj_adapted} (=default_OV({obj_std})")
    print(f"[Heuristics] Calculated True Objective = {ov_true}")
    print("[Heuristics] active_vehicles=",active_vehicles)
    print("[Heuristics] dropped customers", dn,"\n")

    ov_breakdown = {
        "count_overtimes" : overtime_count,
        "count_early_service" : es,
        "count_late_service": ls,
        "count_outsource_customer": len(dn),
        "count_additional_vehicles": added_vehicles,
        "ov_operating_costs" : travel_time_true,
        "ov_overtimes" : OV_overtime,
        "ov_early_service" : OV_early_service,
        "ov_late_service": OV_late_service,
        "ov_outsource_customer": OV_outsourcing,
        "ov_additional_vehicles": OV_add_vehicle,
    } if t_D else {
        "count_overtimes" :0,
        "count_early_service" :0,
        "count_late_service": 0,
        "count_outsource_customer":0,
        "count_additional_vehicles":0,
        "ov_operating_costs" : travel_time_true,
        "ov_overtimes" :0,
        "ov_early_service" :0,
        "ov_late_service": 0,
        "ov_outsource_customer":0,
        "ov_additional_vehicles":0,
    }

    # pprint(ov_breakdown)
    # print("Adaption correct = ", ov_true == obj_adapted)
    # print("Breakdown correct =",obj_adapted==sum([o for o in ov_breakdown.values()]))
    # print(f"OV Calculation ({ov_true}) correct = ",ov_true ==sum([o for o in ov_breakdown.values()]))
    return obj_adapted, total_time, ov_breakdown
        

def plot_nodes_and_paths(xy_data, routes, depot: list = None):
    """
    Plots the nodes (depot and customers) and the routes connecting the nodes.
    
    Parameters:
    - xy_data: A list of tuples (x, y) coordinates of the nodes.
    - routes: A list of routes, where each route is a list of node indices (from the `get_routes` function).
    - depot: A list of indices representing depot locations. If None, the first node is assumed to be the depot.
    """
    # Unzip the list of tuples into x and y coordinates
    x, y = zip(*xy_data)

    # Create a scatter plot for all points
    plt.figure(figsize=(8, 6))  # Set the figure size
    
    

    

    # Plot routes: for each vehicle route, connect the points with lines
    for route in routes:
        route_x = [x[i] for i in route]
        route_y = [y[i] for i in route]
        
        # Plot the path of the route
        # plt.plot(route_x, route_y, linestyle='-', color='black', linewidth=1, label=f'Route {routes.index(route)+1}')
        plt.plot(route_x, route_y, linestyle='-', linewidth=1, label=f'Route {routes.index(route)+1}')#, alpha=0.5, color="black")
        
        # Optionally, you can annotate the route with the node indices.
        # for i, (x_coord, y_coord) in enumerate(zip(route_x, route_y)):
        #     plt.text(x_coord+5, y_coord+5, str(route[i]), fontsize=9, verticalalignment='bottom', horizontalalignment='right')

    # Highlight depot(s)
    # Plot all points as customers
    plt.scatter(x, y, facecolor='white', edgecolor='black', marker='o', alpha=1, label='Customers')
    if depot:
        for d in depot:
            plt.scatter(x[d], y[d], color='black', marker='o', alpha=1, label='Depot(s)')
    else:
        plt.scatter(x[0], y[0], color='black', marker='^', s=2,alpha=1, label='Depot')

    # Title and labels
    # plt.title('VRP Solution')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Legend configuration
    # plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))  # Adjust legend position
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

        # Create custom legend handles
    customer_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='white',
                                    markeredgecolor='black', markersize=8, label='Customer')
    depot_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='black',
                                  markeredgecolor='black', markersize=8, label='Depot')
    route_handle = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label='Route')

    # Add the legend to the plot
    plt.legend(handles=[customer_handle, depot_handle, route_handle],
               loc='upper center',  # Place it above the bottom of the plot
               bbox_to_anchor=(0.5, -0.1),  # Align centrally below the plot
               ncol=3,  # Arrange items in a single row
               frameon=False)  # Optional: Remove the frame around the legend

    # Show the plot
    plt.tight_layout()
    plt.show()

class ProgressCallback:
    def __init__(self, routing, start_time):
        self.routing = routing
        self.start_time = start_time
        self.last_elapsed = 0
        self.last_objective = float('inf')  # Initialize with infinity to track improvements
        self.improvements = []  # List to store (elapsed_time, improved_objective)
        self.history = []

    def __call__(self):
        elapsed_time = time.time() - self.start_time
        current_objective = self.routing.CostVar().Max()  # Get the current best objective
        self.history.append((elapsed_time, current_objective))
        # Update progress bar based on time elapsed
        # Only update progress bar and print if the objective has improved.
        if current_objective < self.last_objective:
            self.last_objective = current_objective
            self.improvements.append((elapsed_time, current_objective))
            # print("-"*80)
            # print(f"At second {elapsed_time:.2f}, Improved Objective = {current_objective}")
            # print("-"*80)
        # Update progress bar based on time elapsed.
        if elapsed_time - self.last_elapsed >= 1:  # Update every second.
            self.last_elapsed = elapsed_time

    def get_improvements(self):
        """Return the list of improvements (elapsed_time, improved_objective)."""
        return self.improvements
    
    def get_history(self):
        """Return the list of improvements (elapsed_time, improved_objective)."""
        return self.history

def plot_improvements_and_history(callback):
    # Get the data from the callback
    improvements = callback.get_improvements()  # Improvements (elapsed_time, objective)
    history = callback.get_history()  # History (elapsed_time, objective)
    
    # Extract data for plotting
    history_time, history_objective = zip(*history)  # Unpack history into separate lists
    improvement_time, improvement_objective = zip(*improvements)  # Unpack improvements into separate lists

    # Plot the history as a black dash-dot line
    plt.plot(history_time, history_objective, linestyle='-.', alpha=0.5, color='black', label='History')

    # Plot the improvements as a black stair plot
    plt.step(improvement_time, improvement_objective, where='post', color='black', label='Improvements')

    # Set plot properties
    # plt.title("Heuristic Improvements Over Time")
    plt.xlabel("Solver Time [s]")
    plt.ylabel("Objective Value")
    plt.legend(loc="upper right")  # Add legend to upper right
    
    plt.gca().spines['top'].set_visible(False)  # Remove top spine
    plt.gca().spines['right'].set_visible(False)  # Remove right spine
    plt.gca().spines['left'].set_position('zero')  # Adjust left spine position
    plt.tight_layout()
    # Display the plot
    plt.show()