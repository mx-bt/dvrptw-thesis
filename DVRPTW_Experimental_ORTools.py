"""Disrupted Vehicles Routing Problem with Time Windows (D-VRPTW)."""
"""
Simple numerical experiment to validate the planned execution
Grade of dynamism measurement
Output in JSON
"""
plot_routes = False
po = False

from custom_tools_max.data_input import get_google_vrptw_sample

from custom_tools_max.miscellaneous import generate_unique_sorted_list
from custom_tools_max.miscellaneous import log_exp_res
from custom_tools_max.miscellaneous import timestamp
from custom_tools_max.miscellaneous import check_vrptw_infeasibilities

from custom_tools_max.simulation_utils import update_time_matrix_dynamic
from custom_tools_max.simulation_utils import Psi

from custom_tools_max.core_solvers import Solve_VRPTW_Heuristics
from custom_tools_max.core_solvers import Solve_DVRPTW_Heuristics
from custom_tools_max.core_solvers import Solve_VRPTW_Exact
from custom_tools_max.core_solvers import Solve_DVRPTW_Exact

from custom_tools_max.output_handling import plot_multiple_cycles
from custom_tools_max.output_handling import print_solution_VRPTW
from custom_tools_max.output_handling import get_routes_and_times
from custom_tools_max.output_handling import plot_nodes_and_paths
from custom_tools_max.output_handling import ProgressCallback
from custom_tools_max.output_handling import plot_improvements_and_history
from custom_tools_max.output_handling import get_all_arcs_np
from custom_tools_max.output_handling import get_routes

from custom_tools_max.data_transformations_disruption import get_vehicle_status
from custom_tools_max.data_transformations_disruption import adapt_parameters_at_disruption

from pprint import pprint
import numpy as np
import pandas as pd
import copy
import json
import os

def main():
    print("\n[Experiment] Started\n")

    def create_data_model_VRPTW():
        """Stores the data for the problem."""
        data = {}
        data['visited_nodes'] = []
        data['considered_vehicles'] = []
        data['lazy_finisher'] = []
        data['initial_solution_heur'] = []
        data["shift_duration"] = 25
        data["time_matrix"] = np.array([
            [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
            [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
            [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
            [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
            [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
            [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
            [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
            [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
            [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
            [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5], # [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
            [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
            [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
            [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
            [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
            [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
            [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
            [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0],
        ])
        data["time_windows"] = np.array([
            (0, data["shift_duration"]),  # depot (0, 5) 
            (7, 12),  # 1
            (10, 15),  # 2
            (16, 18),  # 3
            (10, 13),  # 4
            (0, 5),  # 5
            (5, 10),  # 6
            (0, 4),  # 7
            (5, 10),  # 8
            (0, 3),  # 9
            (10, 16),  # 10
            (10, 15),  # 11
            (0, 5),  # 12
            (5, 10),  # 13
            (7, 8),  # 14
            (10, 15),  # 15
            (11, 15),  # 16
        ])
        data["num_vehicles"] = 10
        data["depot_node_ids"] = [0 for _ in range(data["num_vehicles"])]
        data["starts"] = copy.deepcopy(data["depot_node_ids"])
        data["immediate_destinations"] = []
        # data["ends"] = [0 for _ in range(data["num_vehicles"])]
        data["vehicle_is_moving"] = [0 for _ in range(data["num_vehicles"])]
        data["latest_t_D"] = None,
        data["standard_operating_costs"] = 1, # 400$ for an 8h shift = 0.8333333333333334 $ / min

        return data

    early_services = [2,3,4]
    late_services = [3,4,5]
    overtimes = [2,3,4]
    customer_drops = [50,75,100]
    add_vehicles = [400]
    solver_times = [1,5]
    probabilities = [0.2,0.3,0.5]

    # SENSITIVITY
    for run_t, t_s in enumerate(solver_times):
        # TODO solve initial model here
        data_model_VRPTW_og = create_data_model_VRPTW()
        # =========================================================================
        # SOLVING INITIAL TSP 
        # =========================================================================
        SolutionVRPTW_og = Solve_VRPTW_Heuristics(data_model_VRPTW_og, solver_time_limit=t_s)
        manager_ort_og  = SolutionVRPTW_og[0]
        routing_ort_og = SolutionVRPTW_og[1]
        solution_ort_og = SolutionVRPTW_og[2]
        print(get_all_arcs_np(data_model_VRPTW_og,manager_ort_og,routing_ort_og,solution_ort_og)) if po else None

        data_per_cycle = []
        all_routes_per_cycle = []
        
        if solution_ort_og:
            # NOTE the objective below is customized!!!
            obj_val_init_vrptw_heur, arrival_sum_init, ov_details = print_solution_VRPTW(
                data_model_VRPTW_og, manager_ort_og, routing_ort_og, solution_ort_og
            )
            # print(f"[Solve_VRPTW_Heuristics] Objective Value: {obj_val_init_vrptw_heur}")
            solution_VRPTW_og = get_routes_and_times(data_model_VRPTW_og, manager_ort_og, routing_ort_og, solution_ort_og)
            all_routes_per_cycle.append(copy.deepcopy(solution_VRPTW_og))
            data_per_cycle.append(copy.deepcopy(data_model_VRPTW_og))
            time_of_disruption_per_cycle = [None]
            initial_solution_VRPTW = get_routes(solution_ort_og,routing_ort_og,manager_ort_og)[1:-1]
            pprint(initial_solution_VRPTW)
        else:
            print("[Simulation] Initial VRPTW not solvable, end simulation")
            break
        if plot_routes:
            nodes, _, _ = get_google_vrptw_sample()
            routes = get_routes(solution_ort_og,routing_ort_og,manager_ort_og)
            plot_nodes_and_paths(nodes, routes, depot=data_model_VRPTW_og["starts"])
        
        original_time_matrix = copy.deepcopy(data_model_VRPTW_og['time_matrix'])
        
        total_runs = 0
        for run_p, p in enumerate(probabilities): 
            for EARLY_SERVICE_PENALTY in early_services:
                for LATE_SERVICE_PENALTY in late_services:
                    for OVERTIME_PENALTY in overtimes:
                        for CUSTOMER_OUTSOURCE_PENTALY in customer_drops:
                            for ADD_VEHICLE_COST in add_vehicles:
                                for q_run in range(5):
                            
                                    run_id: str = f"ort_sample.{total_runs}.{q_run}"
                                    experiment_results_run = []
                                    t_D = 0

                                    print("\n",run_id,"-"*80)
                                    print("p=",p)
                                    print("t_s=",t_s,"\n")

                                    # Initial data model and solution
                                    data_model_VRPTW = copy.deepcopy(data_model_VRPTW_og)
                                    data_model_VRPTW["penalties"] = {
                                        "early_service": EARLY_SERVICE_PENALTY, # 2$/minute,
                                        "late_service": LATE_SERVICE_PENALTY, # 3$/minute NOTE @thesis: must ,
                                        "overtime": OVERTIME_PENALTY, 
                                        "customer_drop": CUSTOMER_OUTSOURCE_PENTALY,# 50$ ,
                                        "add_vehicle": ADD_VEHICLE_COST, # 400$ ~cost of one workshift (crew)
                                        }
                                    # SolutionVRPTW = copy.deepcopy(SolutionVRPTW_og) # cannot pickle 'SwigPyObject' object
                                    # manager_ort = copy.deepcopy(manager_ort_og)
                                    # routing_ort = copy.deepcopy(routing_ort_og)
                                    # solution_ort = copy.deepcopy(solution_ort_og)
                                    solution_VRPTW = copy.deepcopy(solution_VRPTW_og)
                                    
                                    print("\nInitial VRPTW") if po else None
                                    print("-"*80) if po else None

                                    for data_label in ["depot_node_ids","starts",'time_matrix','time_windows']:
                                        print(data_label) if po else None
                                        pprint(pd.DataFrame(data_model_VRPTW[data_label])) if po else None
    
                                    print("-"*80) if po else None

                                    global_adapted_tmins = {}
                                    potential_disruptions = {} # {(vehicle,node):t_D}
                                    
                                    previous_time_matrix = copy.deepcopy(original_time_matrix)

                                    max_simulations = data_model_VRPTW["shift_duration"]-1
                                    modified_arcs, modified_indices_history = [],[]
                                    simulation_period = 1 # Init, @0 the initial one is solved, should be feasible
                                    while simulation_period <= max_simulations:
                                        print(f"\n[Simulation] Start Period {int(simulation_period)}")
                                        print("-"*80) if po else None
                                        
                                        # =========================================================================
                                        # GENERATE DEVIATIONS
                                        # =========================================================================
                                        try:
                                            status_report = {
                                                vehicle_id: get_vehicle_status(
                                                    int(simulation_period),
                                                    vehicle_id, solution_VRPTW
                                                    ) for vehicle_id in solution_VRPTW.keys()
                                                }

                                        except KeyError as e:
                                            print("\n",t_D," caused a problem\n")
                                            print(e)
                                        
                                        # Lazy finish (quick and outsmarts a creepy bug)
                                        lazy_tests = set([s["pro_lazy_finish"] for s in status_report.values()])
                                        if len(lazy_tests) == 1 and list(lazy_tests)[0]:
                                            print(f"\n[Simulation] {run_id} Successfully Completed (lazy term)")
                                            break # Current simulation run

                                        excluded_arcs = []
                                        for vehicle_id, status in status_report.items():
                                            excluded_arcs.append(
                                                tuple((status["last_node"],status["next_node"]))
                                            )
                                        excluded_arcs_np = np.array(excluded_arcs)
                                        # print("\nExluded arcs from time matrix updates:")
                                        # print(excluded_arcs_np)
                                        
                                        if simulation_period == 1:
                                            time_matrix_deviated, modified_arcs, modified_indices_history = update_time_matrix_dynamic(
                                                data_model_VRPTW['time_matrix'],
                                                montufar_block = excluded_arcs_np,
                                                probability=p,
                                                )
                                            # modified_indices_history = modified_arcs
                                        else:
                                            time_matrix_deviated, modified_arcs, modified_indices_history = update_time_matrix_dynamic(
                                                time_matrix_deviated,
                                                montufar_block = excluded_arcs_np,
                                                modification_history=modified_indices_history,
                                                simulation_period = simulation_period,
                                                probability=p,
                                                )
                                            # modified_indices_history = np.vstack([modified_indices_history, modified_arcs])
                                                
                                        # print("time_matrix_deviated") #if po else None
                                        # print(pd.DataFrame(time_matrix_deviated))# if po else None
                                        # print("deviations_from_current_matrix")
                                        # print(pd.DataFrame(time_matrix_deviated-original_time_matrix))

                                        all_route_arcs = get_all_arcs_np(
                                            data_model_VRPTW,manager_ort,routing_ort,solution_ort
                                        ) if t_D else get_all_arcs_np(
                                            data_model_VRPTW,manager_ort_og,routing_ort_og,solution_ort_og
                                        )
                                        all_route_arcs_tuples = [tuple(row) for row in all_route_arcs]
                                        modified_arcs_tuples = [tuple(row) for row in modified_arcs]
                                        affected_arcs = np.array([arc for arc in all_route_arcs_tuples if arc in modified_arcs_tuples])
                                        # Print the affected arcs
                                        print("\nAffected route arcs: ") if po else None
                                        print(affected_arcs) if po else None
                                        
                                        # if p <= 14.0:
                                        print("\nFeasibility in Period = ",simulation_period) if po else None

                                        # results_Psi = []

                                        # =========================================================================
                                        # FEASABILITY CHECK + T_MIN Adaption
                                        # =========================================================================
                                        for vehicle_id in solution_VRPTW.keys():

                                            vehicle_status = status_report[vehicle_id]
                                            try:
                                                global_adapted_tmins[vehicle_id]
                                            except KeyError:
                                                global_adapted_tmins[vehicle_id] = {}
                                            
                                            psi_result = Psi(
                                                vehicle_status["last_node_index"],
                                                origin_index=vehicle_status["last_node_index"],
                                                t_mins_vehicle_adapted=global_adapted_tmins[vehicle_id],
                                                solution_vrptw= SolutionVRPTW,
                                                T_orig = previous_time_matrix,
                                                T_delta = time_matrix_deviated,
                                            ) if t_D else Psi(
                                                vehicle_status["last_node_index"],
                                                origin_index=vehicle_status["last_node_index"],
                                                t_mins_vehicle_adapted=global_adapted_tmins[vehicle_id],
                                                solution_vrptw= SolutionVRPTW_og,
                                                T_orig = previous_time_matrix,
                                                T_delta = time_matrix_deviated,
                                            )
                                            
                                            is_disrupted = psi_result[0]
                                            disrupted_node = psi_result[1]
                                            potential_t_D = psi_result[2]
                                            global_adapted_tmins[vehicle_id] = psi_result[3]
                                        

                                            # results_Psi.append(psi_result[0])
                                            # print(psi_result,"\n")
                                
                                            if is_disrupted:
                                                print(f"[Simulation] Route of vehicle {vehicle_id} disrupted at node {disrupted_node} (t_D={potential_t_D})")
                                                potential_disruptions[vehicle_id] = potential_t_D
                                            else:
                                                if vehicle_id in potential_disruptions.keys():
                                                    print(f"[Simulation] {vehicle_id} removed from list")
                                                    potential_disruptions.pop(vehicle_id)
                                                else:
                                                    pass
                                                print(f"[Simulation] Route of vehicle {vehicle_id} feasible")
                                            


                                        if potential_disruptions and simulation_period + 1 == min(list(potential_disruptions.values())):
                                            
                                            # =========================================================================
                                            # DISRUPTION HANDLING
                                            # =========================================================================

                                            t_D = simulation_period + 1
                                            
                                            obj_val_DVRPTW_heur = 0
                                            obj_val_DVRPTW_ex = 0 #proforma setup
                                            obj_val_VRPTW_ex = 0
                                            arrival_sum_dvrptw = 0
                                            arrival_sum_vrptw = 0
                                            data_model_VRPTW["latest_t_D"] = t_D
                                            print(f"\n[Simulation] D-VRPTW to solve for t_D={t_D}")
                                            keys = (
                                                "time_matrix",
                                                "starts",
                                                "visited_nodes",
                                                "considered_vehicles", 
                                                "immediate_destinations",
                                                "time_windows",
                                                "vehicle_is_moving",
                                                "lazy_finisher"
                                                )
                                            
                                            
                                            # original instance solution -> could be adapted
                                            status_report = {}
                                            status_report = {
                                                vehicle_id: get_vehicle_status(t_D, vehicle_id, solution_VRPTW) for vehicle_id in solution_VRPTW.keys()
                                                }
                                            
                                            # NOTE the following line determines if actually deviated matrices are passed!
                                            data_model_VRPTW['time_matrix'] = time_matrix_deviated
                                            data_model_VRPTW.update(
                                                dict(zip(keys,adapt_parameters_at_disruption(status_report,data_model_VRPTW,t_D)))
                                            )
                                            initial_routes = []
                                            exclude = data_model_VRPTW["visited_nodes"] + data_model_VRPTW["starts"] + data_model_VRPTW["depot_node_ids"]
                                            ir_exist = False
                                            for v_init in range(data_model_VRPTW["num_vehicles"]):
                                                # cut_route = []
                                                if v_init in data_model_VRPTW["considered_vehicles"]:
                                                    cut_route = [s["node_id"] for s in solution_VRPTW[v_init] if s["node_id"] not in exclude]
                                                    ir_exist = True
                                                    initial_routes.append(cut_route)

                                            data_model_VRPTW['initial_solution_heur'] = initial_routes if ir_exist else []
                                            # pprint(initial_routes)

                                            # data_model_VRPTW["time_matrix"] = original_time_matrix
                                            SolutionVRPTW = Solve_VRPTW_Heuristics(data_model_VRPTW, solver_time_limit=t_s)
                                            manager_ort  = SolutionVRPTW[0]
                                            routing_ort = SolutionVRPTW[1]
                                            solution_ort = SolutionVRPTW[2]
                                            solver_status = SolutionVRPTW[3]

                                            
                                            if solution_ort:
                                                obj_val_DVRPTW_heur, arrival_sum_vrptw, ov_details = print_solution_VRPTW(
                                                    data_model_VRPTW, manager_ort, routing_ort, solution_ort, t_D=t_D
                                                )

                                                solution_VRPTW = {}
                                                solution_VRPTW = get_routes_and_times(data_model_VRPTW, manager_ort, routing_ort, solution_ort)
                                                previous_time_matrix = copy.deepcopy(data_model_VRPTW['time_matrix'])
                                                
                                                # Reoptimisation
                                                # print("starts: ",data_model_VRPTW['starts'])
                                                # print('depot_node_ids',data_model_VRPTW['depot_node_ids'])
                                                # print("intersection: ",set(data_model_VRPTW['starts'])&set(data_model_VRPTW['depot_node_ids']))

                                                # Exclude if starts include depot (set desaster ahead)
                                                eff_starts = set([data_model_VRPTW['starts'][v] for v in data_model_VRPTW["considered_vehicles"]])
                                                eff_depots = set([data_model_VRPTW['depot_node_ids'][v] for v in data_model_VRPTW["considered_vehicles"]])

                                                # Heuristics don't solve to optimality and no depot in start node
                                                if solver_status != 7 and len(eff_starts&eff_depots) == 0:
                                                    initial_solution_arcs =  get_all_arcs_np(data_model_VRPTW,manager_ort,routing_ort,solution_ort) 
                                                    # print("\nInitial arcs:")
                                                    # print(initial_solution_arcs)
                                                    obj_val_VRPTW_ex, _ = Solve_VRPTW_Exact(   
                                                        data_model_VRPTW,
                                                        solver_time_limit = t_s,
                                                        initial_solution = initial_solution_arcs
                                                    )
                                                
                                                # VRPTW solvable after t_D
                                                experiment_results_run.append(
                                                    log_exp_res(
                                                        "google_ort_sample",
                                                        run_id,p,t_s,
                                                        EARLY_SERVICE_PENALTY,
                                                        LATE_SERVICE_PENALTY,
                                                        OVERTIME_PENALTY,
                                                        CUSTOMER_OUTSOURCE_PENTALY,
                                                        ADD_VEHICLE_COST,
                                                        t_D,
                                                        "I",
                                                        obj_val_init_vrptw_heur,
                                                        obj_val_DVRPTW_heur,
                                                        obj_val_VRPTW_ex,
                                                        obj_val_DVRPTW_ex,
                                                        arrival_sum_init,
                                                        arrival_sum_vrptw,
                                                        arrival_sum_dvrptw,
                                                        ov_details["operating_costs"],
                                                        ov_details["overtimes"],
                                                        ov_details["early_service"],
                                                        ov_details["late_service"],
                                                        ov_details["outsource_customer"],
                                                        ov_details["additional_vehicles"],
                                                    )
                                                )

                                                if plot_routes:
                                                    routes = get_routes(solution_ort, routing_ort, manager_ort)
                                                    plot_nodes_and_paths(nodes, routes, depot=data_model_VRPTW["starts"])
                                                else:
                                                    pass
                                                    # plot_improvements_and_history(progress_callback)
                                            else:
                                                print(f"\n[Simulation] No Solution from VRPTW, trying with DVRPTW")
                                                SolutionVRPTW = Solve_DVRPTW_Heuristics(data_model_VRPTW, solver_time_limit=t_s)
                                                manager_ort  = SolutionVRPTW[0]
                                                routing_ort = SolutionVRPTW[1]
                                                solution_ort = SolutionVRPTW[2]
                                                solver_status = SolutionVRPTW[3]

                                                if solution_ort:
                                                    obj_val_DVRPTW_heur,arrival_sum_dvrptw, ov_details = print_solution_VRPTW(
                                                        data_model_VRPTW, manager_ort, routing_ort, solution_ort, t_D=t_D
                                                    )

                                                    # Reoptimisation 2
                                                    # Exclude if starts include depot (set desaster ahead)
                                                    eff_starts = set([data_model_VRPTW['starts'][v] for v in data_model_VRPTW["considered_vehicles"]])
                                                    eff_depots = set([data_model_VRPTW['depot_node_ids'][v] for v in data_model_VRPTW["considered_vehicles"]])

                                                    # Heuristics don't solve to optimality and no depot in start node
                                                    # if solver_status != 7 and len(eff_starts&eff_depots) == 0:
                                                    if len(eff_starts&eff_depots) == 0:
                                                        initial_solution_arcs =  get_all_arcs_np(data_model_VRPTW,manager_ort,routing_ort,solution_ort) 
                                                        # print("\nInitial arcs:")
                                                        # print(initial_solution_arcs)
                                                        obj_val_DVRPTW_ex, _ = Solve_DVRPTW_Exact(   
                                                            data_model_VRPTW,
                                                            solver_time_limit = t_s,
                                                            initial_solution = initial_solution_arcs
                                                        )

                                                    # only DVRPTW solvable after t_D
                                                    experiment_results_run.append(
                                                        log_exp_res(
                                                            "google_ort_sample",
                                                            run_id,p,t_s,
                                                            EARLY_SERVICE_PENALTY,
                                                            LATE_SERVICE_PENALTY,
                                                            OVERTIME_PENALTY,
                                                            CUSTOMER_OUTSOURCE_PENTALY,
                                                            ADD_VEHICLE_COST,
                                                            t_D,
                                                            "II",
                                                            obj_val_init_vrptw_heur,
                                                            obj_val_DVRPTW_heur,
                                                            obj_val_VRPTW_ex,
                                                            obj_val_DVRPTW_ex,
                                                            arrival_sum_init,
                                                            arrival_sum_vrptw,
                                                            arrival_sum_dvrptw,
                                                            ov_details["operating_costs"],
                                                            ov_details["overtimes"],
                                                            ov_details["early_service"],
                                                            ov_details["late_service"],
                                                            ov_details["outsource_customer"],
                                                            ov_details["additional_vehicles"],
                                                        )
                                                    )

                                                else:
                                                    obj_val_DVRPTW_heur = None
                                                    pprint(data_model_VRPTW, width=175)
                                                    check_vrptw_infeasibilities(data_model_VRPTW)
                                                    print(f"[Simulation] Period {t_D} Not solvable anymore with regular VRPTW nor DVRPTW")# if po else None
                                                    print(f"[Simulation] (this should NEVER be the case LOL)")
                                                    # print("\nEffective Deviations from last solvable instance") #if po else None
                                                    # print(pd.DataFrame(data_model_VRPTW['time_matrix'] - previous_time_matrix))
                                                    print("-"*80)
        
                                                    break # current simulation run

                                            # pprint(solution_VRPTW)
                                            if solution_VRPTW:
                                                all_routes_per_cycle.append(copy.deepcopy(solution_VRPTW))
                                                data_per_cycle.append(copy.deepcopy(data_model_VRPTW))
                                                time_of_disruption_per_cycle.append(t_D)
                                                potential_disruptions = {} # NOTE should be
                                            simulation_period += 1 # since next period is feasible given t_D recalc!
                                        else:
                                            print("Continue as usual") if po else None

                                        if simulation_period == max_simulations:
                                            print(f"\n[Simulation] {run_id} Successfully Completed (max it)") #if po else None
                                            print("\nEffective Final Deviation Deltas") if po else None
                                            print(previous_time_matrix-original_time_matrix) if po else None
                                        
                                        else:
                                            print(f"\n[Simulation] End Current Period") if po else None
                                        print("-"*80) if po else None
                                        simulation_period += 1

                                    file_path = "exp_results/exp_data_ort_sample.json"
                                    if os.path.exists(file_path):
                                        with open(file_path, "r") as json_file:
                                            existing_data = json.load(json_file)
                                    else:
                                        existing_data = []

                                    existing_data.extend(experiment_results_run)

                                    with open(file_path, "w") as json_file:
                                        json.dump(existing_data, json_file, indent=4)
                                    total_runs += 1

            # if len(time_of_disruption_per_cycle) > 1:
            # plot_multiple_cycles(all_routes_per_cycle, data_per_cycle, time_of_disruption_per_cycle)
    
    
    


    print("\n[Experiment] Concluded :-)")

if __name__ == "__main__":
    main()
    

    
        

    