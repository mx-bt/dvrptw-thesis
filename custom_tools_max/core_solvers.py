from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from custom_tools_max.output_handling import print_solution_VRPTW
from custom_tools_max.output_handling import get_routes_and_times
from custom_tools_max.output_handling import measure_time
from custom_tools_max.output_handling import plot_nodes_and_paths, ProgressCallback, plot_improvements_and_history
import time as time_1

def get_solver_status_detail(r):
        status = r.status()
        status_dict = {
            pywrapcp.RoutingModel.ROUTING_NOT_SOLVED: "Not Solved - Solver not run yet",
            pywrapcp.RoutingModel.ROUTING_SUCCESS: "Success - Feasible solution found",
            pywrapcp.RoutingModel.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED:
                "Partial Success - Feasible solution found, but not a local optimum",
            pywrapcp.RoutingModel.ROUTING_FAIL: "Fail - No feasible solution found",
            pywrapcp.RoutingModel.ROUTING_FAIL_TIMEOUT: "Fail - Timeout before finding a solution",
            pywrapcp.RoutingModel.ROUTING_INVALID: "Invalid Problem - Model or parameters are invalid",
            pywrapcp.RoutingModel.ROUTING_INFEASIBLE: "Infeasible - Problem proven to be infeasible",
            pywrapcp.RoutingModel.ROUTING_OPTIMAL: "Optimal - Problem solved to optimality",
        }
        code_dict = {
            pywrapcp.RoutingModel.ROUTING_NOT_SOLVED: 0,
            pywrapcp.RoutingModel.ROUTING_SUCCESS: 1,
            pywrapcp.RoutingModel.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED: 2,
            pywrapcp.RoutingModel.ROUTING_FAIL: 3,
            pywrapcp.RoutingModel.ROUTING_FAIL_TIMEOUT: 4,
            pywrapcp.RoutingModel.ROUTING_INVALID: 5,
            pywrapcp.RoutingModel.ROUTING_INFEASIBLE: 6,
            pywrapcp.RoutingModel.ROUTING_OPTIMAL: 7,
        }


        print(f"[Solve_(D)VRPTW_Heuristics] Solver status: {status_dict.get(status, 'Unknown Status')}")
        return code_dict.get(status, None)

@measure_time
def Solve_VRPTW_Heuristics(data_model, solver_time_limit = None):
    print("\n[Solve_VRPTW_Heuristics] Executed")
    """
    VRPTW
    - soft time windows
    - drops visits at a cost ()
    """
    # Instantiate the data problem.
    data = data_model
    from pprint import pprint
    import pandas as pd
    # for key in data.keys():
    #     print("\n",key)
    #     try:
    #         print(pd.DataFrame(data[key]))
    #     except Exception as e:
    #         pprint(data[key])

    print("[Solve_VRPTW_Heuristics] Solving instance...")

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]),
        data["num_vehicles"],
        data["starts"],
        data["depot_node_ids"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        return data["time_matrix"][from_node][to_node]# * std_op_cost

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        data["shift_duration"],  # allow waiting time
        data["shift_duration"],  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)
    # time_dimension.SetGlobalSpanCostCoefficient(100)

    # Add time window constraints for each location except starts
    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx in data['starts']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(
            int(time_window[0]),  int(time_window[1])
        )
        routing.AddToAssignment(time_dimension.SlackVar(index))

    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data["num_vehicles"]):
        depot_idx = data['starts'][vehicle_id]
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            int(data["time_windows"][depot_idx][0]), int(data["time_windows"][depot_idx][1])
        )
        routing.AddToAssignment(time_dimension.SlackVar(index))

    # Instantiate route start and end times to produce feasible times.
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i))
        )
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i))
        )

    # Exclude specific nodes (visited customers)
    if data['visited_nodes']:
        for node_index in data['visited_nodes']:
            index = manager.NodeToIndex(node_index)
            routing.ActiveVar(index).SetValue(0)

    if data['considered_vehicles']:
        # active = the route of the vehicle is not empty, aka there's at least one node
        # on the route other than the first and last nodes.
        # NOTE this is bad, because start node can be a customer in DVRPTW!!!
        ortools_active_vehicles = 0
        assert len(data['considered_vehicles']) == len(data['lazy_finisher'])
        for lz in data['lazy_finisher']:
            if not lz:
                ortools_active_vehicles += 1
        routing.SetMaximumNumberOfActiveVehicles(ortools_active_vehicles)

    
    if data["immediate_destinations"]:
        # print("\n[Solve_VRPTW_Heuristics] Forced arc travels: ")
        assert len(data["immediate_destinations"]) == len(data['considered_vehicles'])
        # the problem is that we also consider resting vehicles, there previous node should be forbidden
        for vehicle_id in data['considered_vehicles']:
            # Enforce that Node 1 must be visited immediately after Node 0
            if data["vehicle_is_moving"][vehicle_id]:

                # print(
                #         vehicle_id,": ",
                #         data["starts"][vehicle_id],"(",manager.NodeToIndex(data["starts"][vehicle_id]),") (",
                #         data["time_windows"][data["starts"][vehicle_id]][0],
                #         data["time_windows"][data["starts"][vehicle_id]][1],
                #         f") --/{ data["time_matrix"][data["starts"][vehicle_id]][data["immediate_destinations"][vehicle_id]]}/--> ",
                #         data["immediate_destinations"][vehicle_id],"(",manager.NodeToIndex(data["immediate_destinations"][vehicle_id]),") (",
                #         data["time_windows"][data["immediate_destinations"][vehicle_id]][0],
                #         data["time_windows"][data["immediate_destinations"][vehicle_id]][1],")"
                #         )
                
                try:
                    if data["starts"][vehicle_id] != 0:
                        v_start = manager.NodeToIndex(data["starts"][vehicle_id])
                    else:
                        v_start = manager.GetStartIndex(vehicle_id)
                    
                    if data["immediate_destinations"][vehicle_id] != 0:
                        v_end =  manager.NodeToIndex(data["immediate_destinations"][vehicle_id])
                    else:
                        v_end = manager.GetEndIndex(vehicle_id)

                    routing.NextVar(v_start).SetValue(v_end)

                except Exception as e:
                    print(e)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    if solver_time_limit:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = solver_time_limit
        search_parameters.log_search = False
        
    use_init_sol = False
    if data["initial_solution_heur"]:
        routing.CloseModelWithParameters(search_parameters)
        print("\n[Solve_VRPTW_Heuristics] Trying initial solution input:")
        

        init_sol_raw = data["initial_solution_heur"]
        exclude = data["starts"]+data["ends"]+data["depot_node_ids"]
        valid_initial_routes_exist = False
        use_init_sol = False
        initial_routes_clean = []

        for route_raw in init_sol_raw:
            cut_route = [s for s in route_raw if s not in exclude]
            valid_initial_routes_exist = True
            initial_routes_clean.append(cut_route)
        
        # print(initial_routes_clean)

        if valid_initial_routes_exist:
            initial_solution = routing.ReadAssignmentFromRoutes(initial_routes_clean, True)

            if initial_solution:
                # print(initial_solution," type= ", type(initial_solution))# a wild printout ngl
                # print_solution_VRPTW(data, manager, routing, initial_solution)
                print(f"\n[Solve_VRPTW_Heuristics] Accepted Initial Solution w/ ObjVal = {initial_solution.ObjectiveValue()}\n")
                use_init_sol = True
            else:
                print("\n[Solve_VRPTW_Heuristics] CP Rejected Initial Solution\n")
                # Solve the problem.w
        else:
            print("\n[Solve_VRPTW_Heuristics] Initial Routes Invalid!\n")
                
    if use_init_sol:
        print("\n[Solve_VRPTW_Heuristics] SolveFromAssignmentWithParameters...\n")
        solution = routing.SolveFromAssignmentWithParameters(
                    initial_solution, search_parameters
        )
    else:
        print("\n[Solve_VRPTW_Heuristics] SolveWithParameters...\n")
        solution = routing.SolveWithParameters(search_parameters)

    

    # Print solution on console.
    if solution:
        print("\n[Solve_VRPTW_Heuristics] Solved successfully")
        # print(f"[Solve_VRPTW_Heuristics] Objective value: {solution.ObjectiveValue()}")
        return manager, routing, solution, get_solver_status_detail(routing)
    
    else:
        print(f"\n[Solve_VRPTW_Heuristics] No solution found for {ortools_active_vehicles} vehicles") # TODO replace with nicer printout (already exists TTBOMK)
        return manager, routing, None, get_solver_status_detail(routing)

@measure_time
def Solve_DVRPTW_Heuristics(data_model, solver_time_limit = None, stakeholder_input = False):
    print("\n[Solve_DVRPTW_Heuristics] Executed")
    """
    VRPTW with hard time windows, starts and depots flexible.
    """

    lambda_0 = stakeholder_input["LAMBDA_overt_time_for_drivers"]
    lambda_1 = stakeholder_input["LAMBDA_utilisation_of_backup_trucks"]
    lambda_2 = stakeholder_input["LAMBDA_outsourcing_deliveries"]
    lambda_3 = stakeholder_input["LAMBDA_early_service"]
    lambda_4 = stakeholder_input["LAMBDA_late_service"]
        

    OT_PEN = max(1,int(data_model["penalties"]["overtime"]*lambda_0))
    COST_ADD_VEHICLE = max(1,int(data_model["penalties"]["add_vehicle"]*lambda_1))
    CUS_DROP_PEN = max(1,int(data_model["penalties"]["customer_drop"]*lambda_2))
    EARLY_SERV_PEN =  max(1,int(data_model["penalties"]["early_service"]*lambda_3))
    LATE_SERV_PEN = max(1,int(data_model["penalties"]["late_service"]*lambda_4))
    

    print("[Solve_DVRPTW_Heuristics] Solving instance...")

    # Create the routing index manager.
    data = data_model
    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]),
        data["num_vehicles"],
        data["starts"],
        data["depot_node_ids"]
    )
    # print("\n[Solve_DVRPTW_Heuristics] pywrapcp.RoutingIndexManager done")

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    # 
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # cost = data["time_matrix"][from_node][to_node]
        # print(f"time_callback {from_node}.{to_node}={cost}")
        return data["time_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    # print("\n[Solve_DVRPTW_Heuristics] routing.RegisterTransitCallback done")
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # print("\n[Solve_DVRPTW_Heuristics] SetArcCostEvaluatorOfAllVehicle done")

    # Add Time Windows constraint.
    time = "Time"
    routing.AddDimension(
        transit_callback_index,
        data["shift_duration"],  # allow waiting time NOTE Pertubation
        int(data["shift_duration"]*1.5),  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time,
    )
    time_dimension = routing.GetDimensionOrDie(time)
    # time_dimension.SetGlobalSpanCostCoefficient(100)
    # print("\n[Solve_DVRPTW_Heuristics] GetDimensionOrDie done")
    # Add time window constraints for each location except starts and depots
    real_starts = [data["starts"][cv] for cv in data["considered_vehicles"]]
    # print("real_starts=",real_starts)

    for location_nd, time_window in enumerate(data["time_windows"]):
        if location_nd in real_starts:
            continue
        if location_nd in data["depot_node_ids"]:
            continue

        index = manager.NodeToIndex(location_nd)
        time_dimension.SetCumulVarSoftLowerBound(index,int(time_window[0]),EARLY_SERV_PEN)
        time_dimension.SetCumulVarSoftUpperBound(index,int(time_window[1]),LATE_SERV_PEN)

        routing.AddToAssignment(time_dimension.SlackVar(index))
    # print("\n[Solve_DVRPTW_Heuristics] AddToAssignment done")

    # vehicle_start_depot_nodes = {}
    for vehicle_id in range(data["num_vehicles"]):
        depot_nd = data['depot_node_ids'][vehicle_id]
        index = routing.End(vehicle_id)
        # vehicle_start_depot_nodes[vehicle_id] = {"depot_id":index,"depot_node":depot_nd}
        # print("index end:",index)
        # print("index start:",routing.Start(vehicle_id))
        # print("indextonode",manager.IndexToNode(index))
        # print(f"Set lower bound veh.{vehicle_id}, depot {depot_nd}=",data["time_windows"][depot_nd][0])
        # print(f"Set upper bound veh.{vehicle_id}, depot {depot_nd}=",data["time_windows"][depot_nd][1])

        # time_dimension.CumulVar(index).SetMin(int(data["time_windows"][depot_nd][0]))
        time_dimension.SetCumulVarSoftLowerBound(index,int(data["time_windows"][depot_nd][0]),0)
        time_dimension.SetCumulVarSoftUpperBound(index,int(data["time_windows"][depot_nd][1]),OT_PEN)

        # print(f"{index} HasCumulVarSoftUpperBound = {time_dimension.HasCumulVarSoftUpperBound(index)} = {time_dimension.GetCumulVarSoftUpperBound(index)}")
        # print(f"{index} HasCumulVarSoftLowerBound = {time_dimension.HasCumulVarSoftLowerBound(index)} = {time_dimension.GetCumulVarSoftLowerBound(index)}")

        # NOTE maybe the depots dont need assignment bc the slack was already granted in AddDimension()
        # routing.AddToAssignment(time_dimension.SlackVar(index))
        # idx_with_soft_tw.append(index)
    # print("idx_with_soft_tw",idx_with_soft_tw)
    # print("\n[Solve_DVRPTW_Heuristics] SetCumulVarSoftUpperBound done")
    #Add time window constraints for each vehicle start nodes
    idx_with_start_tw = {}
    for vehicle_id in data["considered_vehicles"]:
        start_nd = data['starts'][vehicle_id]
        index = routing.Start(vehicle_id)

        # vehicle_start_depot_nodes[vehicle_id]["start_id"] = index
        # vehicle_start_depot_nodes[vehicle_id]["start_node"] = start_nd

        time_dimension.CumulVar(index).SetRange(
            int(data["time_windows"][start_nd][0]), int(data["time_windows"][start_nd][1])
        )
        routing.AddToAssignment(time_dimension.SlackVar(index))
        idx_with_start_tw[vehicle_id] = {"start_id":index,"start_node":start_nd}
    # print("\n[Solve_DVRPTW_Heuristics] SetRange done")
    # print(data["considered_vehicles"])
    # print(data['starts'])
    # print(data["starts_ids"])
    # print("vehicle_start_depot_nodes",vehicle_start_depot_nodes)
    # print("idx_with_start_tw",idx_with_start_tw)

    assert len(data["considered_vehicles"]) == len(idx_with_start_tw)

    # Instantiate route start and end times to produce feasible times.
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i))
        )
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i))
        )
    # print("\n[Solve_DVRPTW_Heuristics] AddVariableMinimizedByFinalizer done")
    # Exclude specific nodes (visited customers)
    if data['visited_nodes']:
        for node_index in data['visited_nodes']:
            index = manager.NodeToIndex(node_index)
            routing.ActiveVar(index).SetValue(0)
    # print("\n[Solve_DVRPTW_Heuristics] ActiveVar(index).SetValue(0) done")

    if data['considered_vehicles']:
        for vi in range(data["num_vehicles"]):
            if vi not in data['considered_vehicles']:
                routing.SetFixedCostOfVehicle(COST_ADD_VEHICLE, vi) # pesos mexicanos

    # print("\n[Solve_DVRPTW_Heuristics] SetFixedCostOfVehicle done")
    visited_nodes_set: set = set(data["visited_nodes"])
    if data["immediate_destinations"]:
        # print("\n[Solve_DVRPTW_Heuristics] Forced arc travels: ")
        assert len(data["immediate_destinations"]) == len(data['considered_vehicles'])
        # the problem is that we also consider resting vehicles, there previous node should be forbidden
        for vehicle_id in data['considered_vehicles']:
            # Enforce that Node 1 must be visited immediately after Node 0
            if data["vehicle_is_moving"][vehicle_id] or (not data["vehicle_is_moving"][vehicle_id] and (data["immediate_destinations"][vehicle_id] != data["starts"][vehicle_id])):
                try:
                    start_node = data["starts"][vehicle_id]
                    dest_node = data["immediate_destinations"][vehicle_id]
                    
                    if data["starts"][vehicle_id] != 0:
                        v_start = manager.NodeToIndex(data["starts"][vehicle_id])
                        
                    else:
                        v_start = manager.GetStartIndex(vehicle_id)
                        
                    if data["immediate_destinations"][vehicle_id] != 0:
                        v_end =  manager.NodeToIndex(data["immediate_destinations"][vehicle_id])
                        
                    else:
                        v_end = manager.GetEndIndex(vehicle_id)

                    
                    
                    # print("__routing.NextVar(v_start).SetValue(v_end)...")
                    # routing.NextVar(v_start).SetValue(v_end)
                    # print("__routing.NextVar(v_start).SetValue(v_end) done\n")
                    

                except Exception as e:
                    print("__Exception")
                    print(e)
                    print(
                        f"\nVehicle {vehicle_id}: ",
                        start_node,"(",manager.NodeToIndex(start_node),") [",
                        data["time_windows"][start_node][0],
                        data["time_windows"][start_node][1],
                        f"] --/{ data["time_matrix"][start_node][dest_node]}/--> ",
                        data["immediate_destinations"][vehicle_id],"(",manager.NodeToIndex(dest_node),") [",
                        data["time_windows"][dest_node][0],
                        data["time_windows"][dest_node][1],"]"
                        )
                    print("start_node in visited nodes? ", bool(start_node in visited_nodes_set))
                    print("dest_node in visited nodes? ", bool(dest_node in visited_nodes_set))
                    print("Vehicle moving`? ",data["vehicle_is_moving"][vehicle_id])
                    
                    print("routing.VehicleVar(v_start): ",routing.VehicleVar(v_start))
                    print("routing.VehicleVar(v_end): ",routing.VehicleVar(v_end))
                    print("")
            # if data["vehicle_is_moving"][vehicle_id] or (not data["vehicle_is_moving"][vehicle_id] and (data["immediate_destinations"][vehicle_id] != data["starts"][vehicle_id])):
            #     try:
            #         if data["starts"][vehicle_id] != 0:
            #             v_start = manager.NodeToIndex(data["starts"][vehicle_id])
            #         else:
            #             v_start = manager.GetStartIndex(vehicle_id)
                    
            #         if data["immediate_destinations"][vehicle_id] != 0:
            #             v_end =  manager.NodeToIndex(data["immediate_destinations"][vehicle_id])
            #         else:
            #             v_end = manager.GetEndIndex(vehicle_id)

            #         routing.NextVar(v_start).SetValue(v_end)

            #     except Exception as e:
            #         print(e)
            
    # print("\n[Solve_DVRPTW_Heuristics] immediate_destinations done")
    # Allow to drop nodes.
    for node in range(1, len(data["time_matrix"])):
        if node not in data['visited_nodes'] and node not in data["immediate_destinations"] and node not in data['starts']:
            routing.AddDisjunction([manager.NodeToIndex(node)], CUS_DROP_PEN)
    # print("\n[Solve_DVRPTW_Heuristics] AddDisjunction done")
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # print("\n[Solve_DVRPTW_Heuristics] Set first sol heur done")
    if solver_time_limit:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = solver_time_limit
        search_parameters.log_search = False
    # print("\n[Solve_DVRPTW_Heuristics] Set local serach heur done")
    use_init_sol = False
    
    # if False:
    if data["initial_solution_heur"]:
        routing.CloseModelWithParameters(search_parameters)
        print("\n[Solve_DVRPTW_Heuristics] Trying initial solution input:")
        # print(data["initial_solution_heur"])

        init_sol_raw = data["initial_solution_heur"]
        exclude = data["starts"]+data["ends"]+data["depot_node_ids"]
        valid_initial_routes_exist = False
        
        initial_routes_clean = []
        for route_raw in init_sol_raw:
            cut_route = [s for s in route_raw if s not in exclude]
            valid_initial_routes_exist = True
            initial_routes_clean.append(cut_route)

        # print(initial_routes_clean)

        if valid_initial_routes_exist:

            initial_solution = routing.ReadAssignmentFromRoutes(initial_routes_clean, True)

            if initial_solution:
                # print(initial_solution," type= ", type(initial_solution))# a wild printout ngl
                # print_solution_VRPTW(data, manager, routing, initial_solution)
                print(f"\n[Solve_DVRPTW_Heuristics] Accepted Initial Solution w/ ObjVal = {initial_solution.ObjectiveValue()}\n")
                use_init_sol = True
            else:
                print("\n[Solve_DVRPTW_Heuristics] CP Rejected Initial Solution\n")
                return manager, routing, None, get_solver_status_detail(routing)
                
        else:
            print("\n[Solve_DVRPTW_Heuristics] Initial Routes Invalid!\n")
        
    progress_callback = ProgressCallback(routing, time_1.time())
    routing.AddAtSolutionCallback(progress_callback)
    if use_init_sol:
        print("\n[Solve_DVRPTW_Heuristics] SolveFromAssignmentWithParameters...\n")
        solution = routing.SolveFromAssignmentWithParameters(
                    initial_solution, search_parameters
        )
    else:
        print("\n[Solve_DVRPTW_Heuristics] SolveWithParameters...\n")
        solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print("\n[Solve_DVRPTW_Heuristics] Solved successfully")
        # plot_improvements_and_history(progress_callback, info_string)
        # print(f"[Solve_DVRPTW_Heuristics] Objective value: {solution.ObjectiveValue()}")
        return manager, routing, solution, get_solver_status_detail(routing)
    
    else:
        print("\n[Solve_DVRPTW_Heuristics] No solution found") # TODO replace with nicer printout (already exists TTBOMK)
        return manager, routing, None, get_solver_status_detail(routing)

@measure_time
def Solve_VRPTW_Exact(data_model, solver_time_limit=None, initial_solution=None):
    """This actually only works for a disrupted state of the VRPTW but without
    any DVRPTW policies. Hard contraints! ... only works for multiple starts (must be customers!) + one depot"""
    print("\n[Solve_VRPTW_Exact] Executed")
    import gamspy as gp
    import pandas as pd
    import sys
    from custom_tools_max.data_input import get_google_vrptw_sample
    from custom_tools_max.output_handling import extract_routes_GAMSPy
    from custom_tools_max.output_handling import plot_nodes_and_paths
    from pprint import pprint
    from gamspy import Ord
    # from gams import GamsWorkspace

    time_matrix = data_model["time_matrix"]
    time_windows = data_model["time_windows"]
    starts, follow_up_nodes = [], []
    depots = []
    for v in data_model["considered_vehicles"]:
        starts.append(data_model["starts"][v])
        follow_up_nodes.append(data_model["immediate_destinations"][v])
    depots = data_model["depot_node_ids"]

    assert 0 not in starts
    assert set(depots) == set([0])

    # new_start_node_idcs = len(time_matrix)
    # for idx, start_node in enumerate(starts):
    #     if start_node in depots:
    #         starts[idx] = new_start_node_idcs
    #         new_start_node_idcs += 1
    #         time_matrix.append(time_matrix[start_node])
    #         time_windows.append(time_windows[start_node])

    # print(starts)
    # print(pd.DataFrame(time_matrix))
    # print(pd.DataFrame(time_windows))

    C = gp.Container()

    i0 = gp.Set(
        C,
        name = "i0", description="all nodes of the Graph",
        records=[f"{i0}" for i0 in range(len(time_matrix))]
    ) # depot + customers

    i = gp.Set(
        C,
        name = "i", description="only customer nodes",
        domain=i0,
        records=[f"{i}" for i in range(len(time_matrix)) if i not in starts+depots]
    )# customers only
    j0 = gp.Alias(
        C,
        name="j0",
        alias_with=i0,
    )
    j = gp.Alias(
        C,
        name="j",
        alias_with=i
    )
    if data_model["visited_nodes"]:
        vs = gp.Set(
            C,
            name="visited",
            description="already visited nodes, exclude from visits",
            domain=i0,
            records=[f"{vn}" for vn in data_model["visited_nodes"]]
        )
    s = gp.Set(
        C,
        name="starts",
        description="only start nodes",
        domain=i0,
        records=[f"{start}" for start in set(starts)]
    )
    d = gp.Set(
        C,
        name="depots",
        description="only depot nodes",
        domain=i0,
        records=[f"{depot}" for depot in set(depots)]
    )

    # Set up distances
    distances_list = []
    for row in range(len(time_matrix)):
        for col in range(len(time_matrix[row])):
            distances_list.append([row,col,time_matrix[row][col]])
    distances = pd.DataFrame(
        distances_list,columns=["from", "to", "distance"]
    ).set_index(["from", "to"])

    tau = gp.Parameter(
        container=C,
        name="time_matrix",
        description="travel time from i0 to j0",
        domain=[i0,j0],
        records=distances.reset_index()
    )
    

    # Set up time windows
    tw_open_list = []
    tw_close_list = []
    for tw in range(len(time_windows)):
        tw_open_list.append([tw,time_windows[tw][0]])
        tw_close_list.append([tw,time_windows[tw][1]])
    tw_open = pd.DataFrame(
        tw_open_list,
        columns=["node","tw_open"]
    ).set_index(["node","tw_open"])
    tw_close = pd.DataFrame(
        tw_close_list,
        columns=["node","tw_close"]
    ).set_index(["node","tw_close"])
    a = gp.Parameter(
        container=C,
        name="time_window_open",
        domain=i0,
        records=tw_open.reset_index()
    )
    b = gp.Parameter(
        container=C,
        name="time_window_close",
        domain=i0,
        records=tw_close.reset_index()
    )

    # print("\n",a.name, "\n",a.records)
    # print("\n",b.name, "\n",b.records)

    # VARIABLES
    x = gp.Variable(
        container=C,
        name="x_ij",
        domain=[i0,j0],
        type="binary",
        description="Arc i0->j0 is chosen",
    )
    

    if data_model["immediate_destinations"]:
        for from_i,to_j in zip(starts,follow_up_nodes):
            if from_i != to_j:
                # print(f"[Solve_VRPTW_Exact] Fixed x[{from_i},{to_j}] = 1")
                x.fx[str(from_i), str(to_j)] = 1
    
    if initial_solution is not None:
        for node_ij in initial_solution:
            x.l[str(node_ij[0]),str(node_ij[1])] = 1

    for node in i0.records.values.tolist():
        x.fx[node[0],node[0]] = 0

    t = gp.Variable(
        container=C,
        name="t_i",
        domain=i0,
        type="positive",
        description="time the vehicle leaves customer i",
    )

    customer_visits = gp.Equation(
        container=C,
        name="customer_visits",
        domain=i0,
    )
    dont_enter_starts = gp.Equation(
        container=C,
        name="dont_enter_starts",
        domain=s,
    )
    start_leaves = gp.Equation(
        container=C,
        name="start_leaves",
        domain=s,
    )
    # block_starts = gp.Equation(
    #     container=C,
    #     name="block_starts",
    #     domain=s,
    # )
    return_to_depot = gp.Equation(
        container=C,
        name="leaves_equals_depot_arrivals",
        domain=d,
    )
    dont_leave_depot = gp.Equation(
        container=C,
        name="dont_leave_depot",
        domain=d,
    )
    flow_conservation = gp.Equation(
        container=C,
        name="flow_conservation",
        domain=i,
    )

    departure_times = gp.Equation(
        container=C,
        name="departure_times",
        domain=[i,j],
    )
    time_windows_lower_limit = gp.Equation(
        container=C,
        name="time_windows_lower_limit",
        domain=i0,
    )
    time_windows_upper_limit = gp.Equation(
        container=C,
        name="time_windows_upper_limit",
        domain=i0,
    )

    # OBJECTIVE FUNCTION
    objective_function = gp.Sum([i0,j0], tau[i0,j0]*x[i0,j0])

    # CONTRAINTS
    dont_enter_starts[s] = gp.Sum(i0,x[i0,s]) == 0
    start_leaves[s] = gp.Sum(j0,x[s,j0]) == 1

    if data_model["visited_nodes"]:
        customer_visits[i0].where[~d[i0] & ~vs[i0]] = gp.Sum(j0,x[i0,j0]) == 1
    else:
        customer_visits[i0].where[~d[i0]] = gp.Sum(j0,x[i0,j0]) == 1

    flow_conservation[i] = gp.Sum(j0, x[i,j0]) - gp.Sum(j0, x[j0,i]) == 0

    return_to_depot[d] = gp.Sum(i0, x[i0,d]) == len(data_model["considered_vehicles"])
    dont_leave_depot[d] = gp.Sum(j0, x[d,j0]) == 0
    # flow_conservation[j0] = gp.Sum(i0, x[i0,j0]) - gp.Sum(i0, x[j0,i0]) == 0
    departure_times[i,j] = t[j] >= t[i] + tau[i,j]*x[i,j] - (b[i]-a[j])*(1-x[i,j])
    time_windows_lower_limit[i0] = a[i0] <= t[i0]
    time_windows_upper_limit[i0] = t[i0] <= b[i0]

    VRPTW = gp.Model(
        container=C,
        name="BasicVRPTW",
        equations=C.getEquations(),
        problem="MIP",
        sense=gp.Sense.MIN,
        objective=objective_function,
    )

    if solver_time_limit:
        VRPTW.solve(
            # output=sys.stdout,
            options=gp.Options(time_limit=float(solver_time_limit))
        )
    else:
        VRPTW.solve(
            # output=sys.stdout
        )

    model_status = gp.ModelStatus(VRPTW.status)
    print("")
    print("[Solve_VRPTW_Exact]",gp.SolveStatus(VRPTW.solve_status))
    print("[Solve_VRPTW_Exact]",model_status)
    print("[Solve_VRPTW_Exact] Objective Value:",VRPTW.objective_value,"\n")

    # print("[Solve_VRPTW_Exact] Infeasibilities: ")
    # pprint(VRPTW.computeInfeasibilities())

    # print("\n[Solve_VRPTW_Exact] Sets : ")
    # print("\n",j.name,"\n",j.records)
    # print("\n",d.name,"\n",d.records)
    # print("\n",s.name,"\n",s.records)
    # print("\n",vs.name,"\n",vs.records)
    # print("\n[Solve_VRPTW_Exact] Time Windows: ")

    # Ensure both dataframes have matching types for comparison
    # time_matrix_df = tau.records.copy()
    # solution_df = x.records[x.records["level"]==1.0].copy()
    # print("\n[Solve_VRPTW_Exact] GAMSPy Solution: ")
    # print(solution_df)
    # time_matrix_df["from"] = time_matrix_df["from"].astype(int)
    # time_matrix_df["to"] = time_matrix_df["to"].astype(int)
    # solution_df["i0"] = solution_df["i0"].astype(int)
    # solution_df["j0"] = solution_df["j0"].astype(int)

    # # Create a set of (i0, j0) pairs from the solution dataframe
    # solution_pairs = set(zip(solution_df["i0"], solution_df["j0"]))

    # # Filter the time matrix dataframe to include only rows where (from, to) is in the solution pairs
    # filtered_df = time_matrix_df[
    #     time_matrix_df.apply(lambda row: (row["from"], row["to"]) in solution_pairs, axis=1)
    # ]
    # print("\n[Solve_VRPTW_Exact] Corrresponding time matrix values:")
    # print(filtered_df)
    # print("\n[Solve_VRPTW_Exact] Corrresponding time windows:")
    # tw_opens = a.records.copy()
    # tw_closes =  b.records.copy()
    # tw_merged = pd.merge(tw_opens,tw_closes,on=["node"], how="inner")
    # print(tw_merged)
    # print("\n[Solve_VRPTW_Exact] t values:")
    # print(t.records)
    # print("[Solve_VRPTW_Exact] Variables : ")
    # pprint(x.records.values.tolist())
    # else:
    #     print("[Solve_VRPTW_Exact] Routes: ")
    #     extract_routes_GAMSPy(
    #         x.records,
    #         starts=[sn[0] for sn in s.records.values.tolist()]
    #     )
    return VRPTW.objective_value, x.records

@measure_time
def Solve_DVRPTW_Exact(data_model, solver_time_limit=None, initial_solution=None):
    """works for multiple starts (must be customers!) + one depot"""
    print("\n[Solve_DVRPTW_Exact] Executed")
    import gamspy as gp
    import pandas as pd
    import sys
    from custom_tools_max.data_input import get_google_vrptw_sample
    from custom_tools_max.output_handling import extract_routes_GAMSPy
    from custom_tools_max.output_handling import plot_nodes_and_paths
    from pprint import pprint
    from gamspy import Ord
    # from gams import GamsWorkspace

    time_matrix = data_model["time_matrix"]
    time_windows = data_model["time_windows"]
    starts, follow_up_nodes = [], []
    depots = []
    for v in data_model["considered_vehicles"]:
        starts.append(data_model["starts"][v])
        follow_up_nodes.append(data_model["immediate_destinations"][v])
    depots = data_model["depot_node_ids"]

    assert 0 not in starts
    assert set(depots) == set([0])

    # new_start_node_idcs = len(time_matrix)
    # for idx, start_node in enumerate(starts):
    #     if start_node in depots:
    #         starts[idx] = new_start_node_idcs
    #         new_start_node_idcs += 1
    #         time_matrix.append(time_matrix[start_node])
    #         time_windows.append(time_windows[start_node])

    # print(starts)
    # print(pd.DataFrame(time_matrix))
    # print(pd.DataFrame(time_windows))

    C = gp.Container()

    i0 = gp.Set(C, name = "i0", description="all nodes of the Graph", records=[f"{i0}" for i0 in range(len(time_matrix))]) # depot + customers
    i = gp.Set(C, name = "i", description="only customer nodes", domain=i0,records=[f"{i}" for i in range(len(time_matrix)) if i not in starts+depots])# customers only
    j0 = gp.Alias(C, name="j0", alias_with=i0,)
    j = gp.Alias(C, name="j", alias_with=i)
    if data_model["visited_nodes"]:
        vs = gp.Set(C, name="visited",description="already visited nodes, exclude from visits",domain=i0,records=[f"{vn}" for vn in data_model["visited_nodes"]])
    s = gp.Set(C, name="starts", description="only start nodes", domain=i0,records=[f"{start}" for start in set(starts)])
    d = gp.Set(C, name="depots", description="only depot nodes", domain=i0,records=[f"{depot}" for depot in set(depots)])

    # Set up distances
    distances_list = []
    for row in range(len(time_matrix)):
        for col in range(len(time_matrix[row])):
            distances_list.append([row,col,time_matrix[row][col]])
    distances = pd.DataFrame(
        distances_list,columns=["from", "to", "distance"]
    ).set_index(["from", "to"])

    tau = gp.Parameter(C, name="time_matrix",description="travel time from i0 to j0",domain=[i0,j0],records=distances.reset_index())
    
    # Set up time windows
    tw_open_list = []
    tw_close_list = []
    for tw in range(len(time_windows)):
        tw_open_list.append([tw,time_windows[tw][0]])
        tw_close_list.append([tw,time_windows[tw][1]])
    tw_open = pd.DataFrame(
        tw_open_list,
        columns=["node","tw_open"]
    ).set_index(["node","tw_open"])
    tw_close = pd.DataFrame(
        tw_close_list,
        columns=["node","tw_close"]
    ).set_index(["node","tw_close"])
    e = gp.Parameter(container=C, name="time_window_open", domain=i0, records=tw_open.reset_index())
    l = gp.Parameter(container=C, name="time_window_close", domain=i0, records=tw_close.reset_index())
    t_D = gp.Parameter(container=C, name="t_D",records=data_model["latest_t_D"])

    # Penalties
    c_t = gp.Parameter(C, name="c_t", description="Travel cost per unit time",records=data_model["standard_operating_costs"])
    c_ot = gp.Parameter(C, name="c_ot", description="Cost of driver overtime",records=data_model["penalties"]["overtime"])
    c_oc = gp.Parameter(C, name="c_oc", description="Cost of operating an additional vehicle",records=data_model["penalties"]["add_vehicle"])
    c_s = gp.Parameter(C, name="c_s", description="Setup cost for additional vehicle",records=data_model["standard_operating_costs"])
    c_os = gp.Parameter(C, name="c_os", description="Cost of outsourcing a customer",records=data_model["penalties"]["customer_drop"])
    c_es = gp.Parameter(C, name="c_es", description="Cost of early service",records=data_model["penalties"]["early_service"])
    c_ls = gp.Parameter(C, name="c_ls", description="Cost of late service",records=data_model["penalties"]["late_service"])

    # Priorities
    lambda_0 = gp.Parameter(C, name="lambda_0", description="Weight for overtime cost",records=1)
    lambda_1 = gp.Parameter(C, name="lambda_1", description="Weight for additional vehicle cost",records=1)
    lambda_2 = gp.Parameter(C, name="lambda_2", description="Weight for outsourcing cost",records=1)
    lambda_3 = gp.Parameter(C, name="lambda_3", description="Weight for early service cost",records=1)
    lambda_4 = gp.Parameter(C, name="lambda_4", description="Weight for late service cost",records=1)
   
    # VARIABLES
    x = gp.Variable(container=C,name="x_ij",domain=[i0,j0], type="binary",description="Arc i0->j0 is chosen",)

    if data_model["immediate_destinations"]:
        for from_i,to_j in zip(starts,follow_up_nodes):
            if from_i != to_j:
                # print(f"[Solve_VRPTW_Exact] Fixed x[{from_i},{to_j}] = 1")
                x.fx[str(from_i), str(to_j)] = 1
    
    if initial_solution is not None:
        for node_ij in initial_solution:
            x.l[str(node_ij[0]),str(node_ij[1])] = 1

    for node in i0.records.values.tolist():
        x.fx[node[0],node[0]] = 0

    a = gp.Variable(container=C, name="a_i", domain=i0,type="positive",description="time the vehicle leaves customer i")

    # holgura / slack variables
    h_ot_plus = gp.Variable(C, name="h_ot_plus", domain=i0, type="positive", description="Positive overtime slack")
    h_ot_minus = gp.Variable(C, name="h_ot_minus", domain=i0, type="positive", description="Negative overtime slack")
    h_es_plus = gp.Variable(C, name="h_es_plus", domain=i0, type="positive", description="Positive early service slack")
    h_es_minus = gp.Variable(C, name="h_es_minus", domain=i0, type="positive", description="Negative early service slack")
    h_ls_plus = gp.Variable(C, name="h_ls_plus", domain=i0, type="positive", description="Positive late service slack")
    h_ls_minus = gp.Variable(C, name="h_ls_minus", domain=i0, type="positive", description="Negative late service slack")

    # EQUATIONS
    customer_visits = gp.Equation(C, name="customer_visits", domain=i0)
    dont_enter_starts = gp.Equation(C, name="dont_enter_starts", domain=s)
    start_leaves = gp.Equation(C, name="start_leaves", domain=s)
    return_to_depot = gp.Equation(C, name="leaves_equals_depot_arrivals", domain=d)
    dont_leave_depot = gp.Equation(C, name="dont_leave_depot", domain=d)
    flow_conservation = gp.Equation(C, name="flow_conservation", domain=i)
    departure_times = gp.Equation(C, name="departure_times", domain= [i0,j0])#[i,j])
    time_windows_lower_SOFTlimit = gp.Equation(C, name="time_windows_lower_sftlimit", domain=i0)
    time_windows_upper_SOFTlimit = gp.Equation(C, name="time_windows_upper_sftlimit", domain=i0)
    
    time_windows_lower_depot_SOFTlimit = gp.Equation(C, name="time_windows_lower_sftdepotlimit", domain=i0)
    time_windows_upper_depot_SOFTlimit = gp.Equation(C, name="time_windows_upper_sftdepotlimit", domain=i0)    
    # OBJECTIVE FUNCTION
    objective_function = gp.Sum([i0,j0], tau[i0,j0]*x[i0,j0])+lambda_3*gp.Sum(i0,c_es*h_es_minus)+lambda_4*gp.Sum(i0,c_ls*h_ls_minus)#+ lambda_0*gp.Sum(i0,c_ot*h_ot_minus) + lambda_1*()

    # CONTRAINTS
    dont_enter_starts[s] = gp.Sum(i0,x[i0,s]) == 0
    start_leaves[s] = gp.Sum(j0,x[s,j0]) == 1

    if data_model["visited_nodes"]:
        customer_visits[i0].where[~d[i0] & ~vs[i0]] = gp.Sum(j0,x[i0,j0]) == 1
        time_windows_lower_SOFTlimit[i0].where[~d[i0] & ~vs[i0]] = e[i0] - a[i0] + h_es_plus - h_es_minus +t_D== 0
        time_windows_upper_SOFTlimit[i0].where[~d[i0] & ~vs[i0]] = a[i0] - l[i0] + h_ls_plus - h_ls_minus +t_D== 0
        departure_times[i0,j0].where[~vs[i0]] = a[j0] >= a[i0] + tau[i0,j0]*x[i0,j0] - (l[i0]-e[j0])*(1-x[i0,j0])

    else:
        customer_visits[i0].where[~d[i0]] = gp.Sum(j0,x[i0,j0]) == 1
        time_windows_lower_SOFTlimit[i0].where[~d[i0]] = e[i0] - a[i0] + h_es_plus - h_es_minus +t_D== 0
        time_windows_upper_SOFTlimit[i0].where[~d[i0]] = a[i0] - l[i0] + h_ls_plus - h_ls_minus +t_D== 0
        departure_times[i0,j0] = a[j0] >= a[i0] + tau[i0,j0]*x[i0,j0] - (l[i0]-e[j0])*(1-x[i0,j0])

    time_windows_lower_depot_SOFTlimit[i0].where[d[i0]] = e[i0] - a[i0] + h_es_plus - h_es_minus +t_D== 0
    time_windows_upper_depot_SOFTlimit[i0].where[d[i0]] = a[i0] - l[i0] + h_ls_plus - h_ls_minus +t_D== 0

    flow_conservation[i] = gp.Sum(j0, x[i,j0]) - gp.Sum(j0, x[j0,i]) == 0

    return_to_depot[d] = gp.Sum(i0, x[i0,d]) == len(data_model["considered_vehicles"])
    dont_leave_depot[d] = gp.Sum(j0, x[d,j0]) == 0
    # flow_conservation[j0] = gp.Sum(i0, x[i0,j0]) - gp.Sum(i0, x[j0,i0]) == 0
    # departure_times[i,j] = a[j] >= a[i] + tau[i,j]*x[i,j] - (l[i]-e[j])*(1-x[i,j]) # NOTE Worked quite well except start-j
    
     
    DVRPTW = gp.Model(
        container=C,
        name="DisruptedVRPTW",
        equations=C.getEquations(),
        problem="MIP",
        sense=gp.Sense.MIN,
        objective=objective_function,
    )

    if solver_time_limit:
        DVRPTW.solve(
            # output=sys.stdout,
            options=gp.Options(time_limit=float(solver_time_limit))
        )
    else:
        DVRPTW.solve(
            # output=sys.stdout
        )

    model_status = gp.ModelStatus(DVRPTW.status)
    print("")
    print("[Solve_DVRPTW_Exact]",gp.SolveStatus(DVRPTW.solve_status))
    print("[Solve_DVRPTW_Exact]",model_status)
    print("[Solve_DVRPTW_Exact] Objective Value:",DVRPTW.objective_value,"\n")

    # print(h_es_plus.name,"\n",h_es_plus.records)
    # print(h_es_minus.name,"\n",h_es_minus.records)
    # print(h_ls_plus.name,"\n",h_ls_plus.records)
    # print(h_ls_minus.name,"\n",h_ls_minus.records)

    # print("[Solve_VRPTW_Exact] Infeasibilities: ")
    # pprint(VRPTW.computeInfeasibilities())

    # print("\n[Solve_VRPTW_Exact] Sets : ")
    # print("\n",j.name,"\n",j.records)
    # print("\n",d.name,"\n",d.records)
    # print("\n",s.name,"\n",s.records)
    # print("\n",vs.name,"\n",vs.records)
    # print("\n[Solve_VRPTW_Exact] Time Windows: ")

    # Ensure both dataframes have matching types for comparison
    # time_matrix_df = tau.records.copy()
    # solution_df = x.records[x.records["level"]==1.0].copy()
    # print("\n[Solve_VRPTW_Exact] GAMSPy Solution: ")
    # print(solution_df)
    # time_matrix_df["from"] = time_matrix_df["from"].astype(int)
    # time_matrix_df["to"] = time_matrix_df["to"].astype(int)
    # solution_df["i0"] = solution_df["i0"].astype(int)
    # solution_df["j0"] = solution_df["j0"].astype(int)

    # # Create a set of (i0, j0) pairs from the solution dataframe
    # solution_pairs = set(zip(solution_df["i0"], solution_df["j0"]))

    # # Filter the time matrix dataframe to include only rows where (from, to) is in the solution pairs
    # filtered_df = time_matrix_df[
    #     time_matrix_df.apply(lambda row: (row["from"], row["to"]) in solution_pairs, axis=1)
    # ]
    # print("\n[Solve_VRPTW_Exact] Corrresponding time matrix values:")
    # print(filtered_df)
    # print("\n[Solve_VRPTW_Exact] Corrresponding time windows:")
    # tw_opens = a.records.copy()
    # tw_closes =  b.records.copy()
    # tw_merged = pd.merge(tw_opens,tw_closes,on=["node"], how="inner")
    # print(tw_merged)
    # print("\n[Solve_VRPTW_Exact] a values:")
    # print(a.records)
    # print("[Solve_VRPTW_Exact] Variables : ")
    # pprint(x.records.values.tolist())
    # else:
    #     print("[Solve_VRPTW_Exact] Routes: ")
    #     extract_routes_GAMSPy(
    #         x.records,
    #         starts=[sn[0] for sn in s.records.values.tolist()]
    #     )
    print("")
    return DVRPTW.objective_value, x.records