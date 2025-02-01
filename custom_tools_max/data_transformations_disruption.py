from typing import Dict, List, Tuple, Union, Optional

# NumPy for performance reasons
import numpy as np

import pandas as pd

def adapt_parameters_at_disruption(
        status_report: Dict[str, Dict[str, List[int]]],
        data: Dict[str, List],
        disruption_time: int,
    ) -> Dict[str, List]:
    """
    Adjusts the routing parameters and time matrix based on vehicle disruption data.

    This function updates the list of visited nodes, modifies vehicle start positions,
    and adapts the time matrix to reflect remaining travel times for vehicles still moving.

    Args:
        status_report (Dict[str, Dict[str, List[int]]]): Contains the disruption status of each vehicle, 
            including visited nodes, last node, target node, and time left to target node.
        data (Dict[str, List]): Routing data.
    
    Returns:
        Dict[str, List]: Updated time matrix, vehicle starts, and the visited nodes list.
    """

    # print("visited_nodes: ",data['visited_nodes'])
    # Determine visited nodes
    nodes_to_skip = [visited_node for vs in status_report.values() for visited_node in vs["visited_nodes"] if visited_node not in data['depot_node_ids']]
    data['visited_nodes'] = list(set(nodes_to_skip + data['visited_nodes']))
    # Create the list of remaining nodes after popping
    # Excluding already popped elements?

    # print("Orig time_matrix")
    # print(pd.DataFrame(data["time_matrix"]))

    # # Modify origins
    # print("origins", data["starts"])

    # Adapt time matrix to imaginary node travel times
    data["immediate_destinations"] = {}
    data['considered_vehicles'] = []
    data['lazy_finisher'] = []
    data["starts_ids"] = {}
    for vehicle_id, report in status_report.items():
        data["vehicle_is_moving"][vehicle_id] = 0
        data['considered_vehicles'].append(int(vehicle_id))
        data['lazy_finisher'].append(report["pro_lazy_finish"])
        data["starts"][vehicle_id] = report['last_node']
        # data["starts_ids"][vehicle_id] = report["last_node_index"]
        data["immediate_destinations"][vehicle_id] = report['target_node']
        
        if report['moving']:
            data["vehicle_is_moving"][vehicle_id] = 1
            data["time_matrix"][report['last_node']][report['target_node']] = report['time_left']

            # Even though not required for general route planning one must edit
            # the time windows so enforced routes stay feasible

        elif not report['moving'] and (report['last_node'] != report['target_node']):
            data["time_matrix"][report['last_node']][report['target_node']] = 0
        
        new_tw_open = disruption_time # always
        old_tw_close = data["time_windows"][report['last_node']][1]
        new_tw_close = disruption_time if disruption_time >= old_tw_close else old_tw_close
        data["time_windows"][report['last_node']] = (new_tw_open, new_tw_close)
                
            
    for i,time_window in enumerate(data["time_windows"]):
        if (time_window[1] > disruption_time) and (time_window[0] < disruption_time):
            data["time_windows"][i] = (disruption_time, time_window[1])
    
    # print("new_origins", data["starts"])
    # print("After Imaginary Node Adapt")
    # print(pd.DataFrame(data["time_matrix"]))
    # print("data['visited_nodes']: ",data['visited_nodes'])

    # Adapt time matrix
    # delta_ij = {
    #         (14,16): 1
    #     }
    
    # data["time_matrix"], _ = update_time_matrix(data["time_matrix"]) # ,delta_ij)

    results_tuple = tuple(
        (
            data["time_matrix"],
            data["starts"],
            data['visited_nodes'],
            list(set(data['considered_vehicles'])),
            data["immediate_destinations"],
            data["time_windows"],
            data["vehicle_is_moving"],
            data['lazy_finisher']
        )
    )

    return results_tuple

def get_vehicle_status(tsr: int, vehicle_id: int, all_routes_and_times: Dict[int, List[Dict[str, Union[int, bool]]]]) -> Union[Dict[str, Union[bool, int, List[int]]], bool]:
    """
    Determines the status of a vehicle at a specific time point during its route.

    Args:
        tsr (int): The time for which the status of the vehicle is requested.
        vehicle_id (int): The ID of the vehicle whose status is requested.
        all_routes_and_times (Dict[int, List[Dict[str, Union[int, bool]]]]): A dictionary containing routes and times for each vehicle. 
            - Each route is a list of dictionaries with the following keys:
            - "node_id": node_id,
            - "time_min": solution.Min(time_var),
            - "time_max": solution.Max(time_var),
            - "time_window_open": data["time_windows"][node_id][0],
            - "time_window_close": data["time_windows"][node_id][1],
            - "travel_time_to_next_node": actual_travel_time,
            - "slack_min": solution.Min(slack_var),
            - "slack_max": solution.Max(slack_var),
            - "estimated_departure_time": estimated_departure,

    Returns:
        Union[Dict[str, Union[bool, int, List[int]]], bool]: A dictionary containing the vehicle's status, or `False` if no route is found:
            - "moving" (bool): Whether the vehicle is moving.
            - "last_node" (int or bool): The last node the vehicle has visited, or `False` if none.
            - "target_node" (int or None): The node the vehicle is heading towards, or `None` if it is not moving.
            - "time_left" (int): Time left to reach the next node, or 0 if the vehicle is not moving.
            - "visited_nodes" (List[int] or None): A list of visited node IDs or `None` if none were visited.

    Example:
        Given a route for `vehicle_id=1`:
        
        ```python
        all_routes_and_times = {
            1: [
                {"node_id": 1, "time_min": 0, "time_max": 5},
                {"node_id": 2, "time_min": 10, "time_max": 15},
                {"node_id": 3, "time_min": 20, "time_max": 25}
            ]
        }
        ```

        Calling the function:

        ```python
        get_vehicle_status(7, 1, all_routes_and_times)
        ```

        Output:
        ```python
        {
            "moving": True,
            "last_node": 1,
            "target_node": 2,
            "time_left": 3,
            "visited_nodes": []
        }
        ```
    """
    route = all_routes_and_times.get(vehicle_id, [])

    if not route:
        # Return empty status if no route is found for this vehicle
        return False
    
    # print(pd.DataFrame(route))

    status = {
        "moving": False,  # Initially assume the vehicle is not moving
        "last_node": False,
        # "last_node_index": False,
        "target_node": None,
        "time_left": 0,
        "visited_nodes": None,
        "visited_nodes_indices": None,
    }

    # Iterate through each node in the route
    node_tracking = []
    # node_index_tracking = []
    i = 0
    while i < len(route) - 1:
        time_min = route[i]["time_min"]
        # print(vehicle_id, node)
        edt = route[i]["estimated_departure_time"] # NOTE might need to replace time_min when disruptions+PSI occur
        node_tracking.append(route[i]["node_id"])
        # node_index_tracking.append(route[i]["node_index"])
        edt_nn = route[i+1]["estimated_departure_time"] if route[i+1]["estimated_departure_time"] else route[i+1]["time_min"]

        # Check if the vehicle is departing from a customer
        if (tsr <= edt):
            status["moving"] = False # True
            status["last_node"] = node_tracking.pop()
            # status["last_node_index"] = node_index_tracking.pop()
            status["target_node"] = route[i]["node_id"]
            # status['target_node_index'] = route[i]["node_index"]
            status["next_node"] = route[i+1]["node_id"]
            status["time_left"] = 0 # route[i]['travel_time_to_next_node']
            status["visited_nodes"] = node_tracking
            # status["visited_nodes_indices"] = node_index_tracking
            status["pro_lazy_finish"] = True if status["next_node"] == route[-1]["node_id"] else False
            return status
        
        # Vehicle on the move
        elif (tsr > edt) and (tsr <= edt + route[i]['travel_time_to_next_node']) and (tsr != edt_nn):
            status["moving"] = True
            status["last_node"] = node_tracking.pop()
            # status["last_node_index"] = node_index_tracking.pop()
            status["target_node"] = route[i+1]["node_id"]
            # status['target_node_index'] = route[i+1]["node_index"]
            status["next_node"] = route[i+1]["node_id"]
            status["time_left"] = edt + route[i]['travel_time_to_next_node'] - tsr
            status["visited_nodes"] = node_tracking
            # status["visited_nodes_indices"] = node_index_tracking
            status["pro_lazy_finish"] = True if status["next_node"] == route[-1]["node_id"] else False
            return status
        
        # Vehicle waiting at next customer
        elif (tsr > edt + route[i]['travel_time_to_next_node']) and (tsr < edt_nn):
            status["moving"] = False
            status["last_node"] = node_tracking.pop()
            # status["last_node_index"] = node_index_tracking.pop()
            status["target_node"] = route[i+1]["node_id"]
            # status['target_node_index'] = route[i+1]["node_index"]
            status["next_node"] = route[i+1]["node_id"]
            status["time_left"] = 0
            status["visited_nodes"] = node_tracking
            # status["visited_nodes_indices"] = node_index_tracking
            status["pro_lazy_finish"] = True if status["next_node"] == route[-1]["node_id"] else False
            return status
        
        else:
            pass

        i += 1
    
    # Vehicle chilling at last node (depot)
    status["moving"] = False
    status["target_node"] = route[-1]["node_id"]
    # status['target_node_index'] = route[-1]["node_index"]
    status["next_node"] = route[-1]["node_id"]
    status["time_left"] = 0
    status["last_node"] = route[-1]["node_id"]
    # status["last_node_index"] = route[-1]["node_index"]
    status["visited_nodes"] = node_tracking
    # status["visited_nodes_indices"] = node_index_tracking
    status["pro_lazy_finish"] = True
    return status