"""Disrupted Vehicles Routing Problem with Time Windows (D-VRPTW)."""
"""
Simple numerical experiment to validate the planned execution
Grade of dynamism measurement
Output in JSON
"""

plot_routes = False
po = False

# from custom_tools_max.data_input import get_google_vrptw_sample
from custom_tools_max.data_input import get_solomon_benchmark
from custom_tools_max.master_config import LAMBDA_early_service
from custom_tools_max.master_config import LAMBDA_late_service
from custom_tools_max.master_config import LAMBDA_outsourcing_deliveries
from custom_tools_max.master_config import LAMBDA_overt_time_for_drivers
from custom_tools_max.master_config import LAMBDA_utilisation_of_backup_trucks      

# from custom_tools_max.miscellaneous import generate_unique_sorted_list
from custom_tools_max.miscellaneous import log_exp_res
from custom_tools_max.miscellaneous import timestamp
# from custom_tools_max.miscellaneous import check_vrptw_infeasibilities

from custom_tools_max.simulation_utils import update_time_matrix_dynamic
from custom_tools_max.simulation_utils import Psi

from custom_tools_max.core_solvers import Solve_VRPTW_Heuristics
from custom_tools_max.core_solvers import Solve_DVRPTW_Heuristics
# from custom_tools_max.core_solvers import Solve_VRPTW_Exact
# from custom_tools_max.core_solvers import Solve_DVRPTW_Exact

from custom_tools_max.output_handling import plot_multiple_cycles
from custom_tools_max.output_handling import print_solution_VRPTW
from custom_tools_max.output_handling import get_routes_and_times
# from custom_tools_max.output_handling import plot_nodes_and_paths
# from custom_tools_max.output_handling import ProgressCallback
# from custom_tools_max.output_handling import plot_improvements_and_history
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
import time

file_id = input("Log File ID: int = ")
traffic_scenario = float(input("Traffic Scenario (0.1=sc.1 0.6=sc.2): float = "))
lambda_mode = int(input("Lambda mode (0=none 1=standard 2=squared 3=cubed): int ="))
solvertime = int(input("Solver time: int = "))
s = int(input("Start from run: int = "))

def log_experiment_result(result):
    try:
        file_path_kpi = f"exp_results/solomonRC101_{file_id}.json"
        if os.path.exists(file_path_kpi):
            with open(file_path_kpi, "r") as json_file:
                existing_data = json.load(json_file)
        else:
            existing_data = []

        # existing_data = result
        existing_data.append(result)

        with open(file_path_kpi, "w") as json_file:
            json.dump(existing_data, json_file,indent=4)

        return 0
    except Exception as e:
        return 1



def log_routes(routes_current_run,dataset,run,disruption_time):
    try:
        file_path_t_D_route = f"route_cache\tD_routes_solomonRC101_{file_id}.json"
        if os.path.exists(file_path_t_D_route):
            with open(file_path_t_D_route, "r") as json_file:
                existing_routes = json.load(json_file)
        else:
            existing_routes = {}
        
        existing_routes[f"{dataset}_{run}_{disruption_time}"] = routes_current_run

        with open(file_path_t_D_route, "w") as json_file:
            json.dump(existing_routes, json_file)

        return 0
    except Exception as e:
        return 1

def main():
    start_time_stamp = timestamp()
    start_time = time.time()
    print(f"\n[Experiment] {start_time_stamp} Started\n")
    dataset_name = "traffic2"
    all_exp_FULL = []
    def create_data_model_VRPTW():
        """Stores the data for the problem."""
        # 55 kmh because this is the np.nanmean of the 2828 dataset
        nodes, time_matrix, time_windows, _  = get_solomon_benchmark(velocity=55,add_service_time = True)
        data = {}
        data["nodes_xy"] = nodes
        data['visited_nodes'] = []
        data['considered_vehicles'] = []
        data['lazy_finisher'] = []
        data['initial_solution_heur'] = []
        data["shift_duration"] = 240
        data["time_matrix"] = time_matrix
        data["time_windows"] = time_windows
        data["num_vehicles"] = 60
        data["depot_node_ids"] = [0 for _ in range(data["num_vehicles"])]
        data["starts"] = copy.deepcopy(data["depot_node_ids"])
        data["immediate_destinations"] = []
        data["ends"] = [0 for _ in range(data["num_vehicles"])]
        data["vehicle_is_moving"] = [0 for _ in range(data["num_vehicles"])]
        data["latest_t_D"] = None,
        data["standard_operating_costs"] = 1, # 400$ for an 8h shift = 0.8333333333333334 $ / min

        assert len(data["time_windows"])==len(data["time_matrix"][0])
        assert data["num_vehicles"] == len(data["starts"]) == len(data["ends"]) == len(data["depot_node_ids"])

        return data
    
    lambdas_standard = {
        "mode": "Direct TD-DVRPTW",
        "values": {
            "LAMBDA_overt_time_for_drivers": LAMBDA_overt_time_for_drivers,
            "LAMBDA_utilisation_of_backup_trucks": LAMBDA_utilisation_of_backup_trucks,
            "LAMBDA_outsourcing_deliveries": LAMBDA_outsourcing_deliveries,
            "LAMBDA_early_service": LAMBDA_early_service,
            "LAMBDA_late_service": LAMBDA_late_service
        }
    }
    lambdas_squared = {"mode": "Leveraged TD-DVRPTW","values":{key:value**2 for key,value in lambdas_standard["values"].items()}}
    lambdas_cubed = {"mode": "Leveraged TD-DVRPTW","values":{key:value**3 for key,value in lambdas_standard["values"].items()}}
    lambdas_zero = {"mode": "Standard DVRPTW","values":{key:1.0 for key,_ in lambdas_standard["values"].items()}}
    lambdas_options = [lambdas_zero,lambdas_standard,lambdas_squared,lambdas_cubed]
    pprint(lambdas_options[lambda_mode])
    # pprint(lambdas_squared)

    div = 100
    early_services = [2]#,3,4]
    late_services = [3]#,4,5]
    overtimes =[2] #,3,4]
    customer_drops = [50]#[50,75,100]
    add_vehicles = [400]#[400]
    solver_times = [solvertime]#[5]#30
    logshapes = [0.6] # peaks
    logscales = [traffic_scenario] # spreads 
    params_to_test = [early_services,late_services,overtimes,customer_drops,add_vehicles,solver_times,logscales,logshapes]
    no_scenarios = 1
    for l in params_to_test:
        no_scenarios *= len(l) 

    exp_log_err = 0
    route_log_err = 0

    for t_s in solver_times:
       
        data_model_VRPTW_og = create_data_model_VRPTW()
        # =========================================================================
        # SOLVING INITIAL VRPTW
        # =========================================================================
        file_path_vrptw_sol = r"route_cache\initial_routes_solomonRC101_v55.json"
        if os.path.exists(file_path_vrptw_sol):
            with open(file_path_vrptw_sol, "r") as json_file:
                existing_routes = json.load(json_file)
        else:
            existing_routes = {}

        if existing_routes:
            min_ov = min([int(ov) for ov in existing_routes.keys()])
            data_model_VRPTW_og['initial_solution_heur'] = existing_routes[str(min_ov)]

        SolutionVRPTW_og = Solve_VRPTW_Heuristics(data_model_VRPTW_og, solver_time_limit=1)
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
            # pprint(initial_solution_VRPTW)
        else:
            print("[Simulation] Initial VRPTW not solvable, end simulation")
            break
        # if plot_routes:
            # nodes, _, _ = get_google_vrptw_sample()
        routes = get_routes(solution_ort_og,routing_ort_og,manager_ort_og)
        #plot_nodes_and_paths(data_model_VRPTW_og["nodes_xy"] , routes, depot=data_model_VRPTW_og["depot_node_ids"])
        
        original_time_matrix = copy.deepcopy(data_model_VRPTW_og['time_matrix'])
        
        total_runs = 0
        for lambda_value in [lambdas_options[lambda_mode]]:
            for lshape in logshapes:
                for lscale in logscales:
                    for EARLY_SERVICE_PENALTY in early_services:
                        for LATE_SERVICE_PENALTY in late_services:
                            for OVERTIME_PENALTY in overtimes:
                                for CUSTOMER_OUTSOURCE_PENTALY in customer_drops:
                                    for ADD_VEHICLE_COST in add_vehicles:
                                        
                                        for q_run in range(s,div):
                                            print("-"*80)
                                            print(f"{total_runs+1}/{div*no_scenarios} in progress...")
                                            print("-"*80)
                                            run_id: str = f"solomon.{total_runs}.{q_run}"
                                            experiment_results_run = []
                                            t_D = 0

                                            print("\n",run_id,"-"*80)
                                            print("lshape=",lshape,"lscale=",lscale)
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

                                            max_simulations = (data_model_VRPTW["shift_duration"]*2)
                                            modified_arcs, modified_indices_history = [],[]
                                            simulation_period = 1 # Init, @0 the initial one is solved, should be feasible
                                            while simulation_period <= max_simulations:
                                                print(f"\n[Simulation] [{timestamp()}] Start Period {int(simulation_period)}")
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
                                                    print(f"\n[Simulation] [{timestamp()}] {run_id} Successfully Completed (lazy term)")
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
                                                        logshape = lshape,
                                                        logscale = lscale,
                                                        )
                                                    # modified_indices_history = modified_arcs
                                                else:
                                                    time_matrix_deviated, modified_arcs, modified_indices_history = update_time_matrix_dynamic(
                                                        time_matrix_deviated,
                                                        montufar_block = excluded_arcs_np,
                                                        modification_history=modified_indices_history,
                                                        simulation_period = simulation_period,
                                                        logshape = lshape,
                                                        logscale = lscale,
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
                                                # print("\nstatus_report")
                                                # pprint(status_report)
                                                # print("\nsolution_VRPTW")
                                                # pprint(solution_VRPTW)
                                                assert len(status_report) == len(solution_VRPTW)
                                                for vehicle_id in solution_VRPTW.keys():

                                                    # vehicle_status = status_report[vehicle_id]
                                                    try:
                                                        global_adapted_tmins[vehicle_id]
                                                    except KeyError:
                                                        global_adapted_tmins[vehicle_id] = {}
                                                    
                                                    # print("\nVehicle ID", vehicle_id)
                                                    psi_result = Psi(
                                                        routing_ort.Start(vehicle_id),
                                                        origin_index=routing_ort.Start(vehicle_id),
                                                        t_mins_vehicle_adapted=global_adapted_tmins[vehicle_id],
                                                        solution_vrptw= SolutionVRPTW,
                                                        T_orig = previous_time_matrix,
                                                        T_delta = time_matrix_deviated,
                                                    ) if t_D else Psi(
                                                        routing_ort_og.Start(vehicle_id),
                                                        origin_index=routing_ort_og.Start(vehicle_id),
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
                                                        # print(f"[Simulation] Route of vehicle {vehicle_id} disrupted at node {disrupted_node} (t_D={potential_t_D})")
                                                        potential_disruptions[vehicle_id] = potential_t_D
                                                    else:
                                                        if vehicle_id in potential_disruptions.keys():
                                                            # print(f"[Simulation] {vehicle_id} removed from list")
                                                            potential_disruptions.pop(vehicle_id)
                                                        else:
                                                            pass
                                                        # print(f"[Simulation] Route of vehicle {vehicle_id} feasible")
                                                    


                                                if potential_disruptions and simulation_period + 1 == min(list(potential_disruptions.values())):
                                                    
                                                    # =========================================================================
                                                    # DISRUPTION HANDLING
                                                    # =========================================================================

                                                    t_D = simulation_period+1
                                                    
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
                                                    
                                                    for v_init in range(data_model_VRPTW["num_vehicles"]):
                                                        
                                                        if v_init in data_model_VRPTW["considered_vehicles"]:
                                                            add_route = [s["node_id"] for s in solution_VRPTW[v_init]]
                                                            initial_routes.append(add_route)
                                                        else:
                                                            v_depot = data_model_VRPTW["depot_node_ids"][v_init]
                                                            initial_routes.append([v_depot,v_depot])

                                                    data_model_VRPTW['initial_solution_heur'] = initial_routes
                                                    # pprint(initial_routes)

                                                    if False:
                                                        # data_model_VRPTW["time_matrix"] = original_time_matrix
                                                        SolutionVRPTW = Solve_VRPTW_Heuristics(data_model_VRPTW, solver_time_limit=t_s)
                                                        manager_ort  = SolutionVRPTW[0]
                                                        routing_ort = SolutionVRPTW[1]
                                                        solution_ort = SolutionVRPTW[2]
                                                        solver_status = SolutionVRPTW[3]

                                                    if False:
                                                    # if solution_ort:
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

                                                        # --------------------------------------------------------------
                                                        # REOPT VRPTW 
                                                        # --------------------------------------------------------------
                                                        # Heuristics don't solve to optimality and no depot in start node
                                                        # if solver_status != 7 and len(eff_starts&eff_depots) == 0:
                                                        #     initial_solution_arcs =  get_all_arcs_np(data_model_VRPTW,manager_ort,routing_ort,solution_ort) 
                                                        #     # print("\nInitial arcs:")
                                                        #     # print(initial_solution_arcs)
                                                        #     obj_val_VRPTW_ex, _ = Solve_VRPTW_Exact(   
                                                        #         data_model_VRPTW,
                                                        #         solver_time_limit = t_s,
                                                        #         initial_solution = initial_solution_arcs
                                                        #     )
                                                        
                                                        # VRPTW solvable after t_D
                                                        result_tD = log_exp_res(
                                                            dataset_name,
                                                            run_id,
                                                            t_s,
                                                            lshape,
                                                            lscale,
                                                            EARLY_SERVICE_PENALTY,
                                                            LATE_SERVICE_PENALTY,
                                                            OVERTIME_PENALTY,
                                                            CUSTOMER_OUTSOURCE_PENTALY,
                                                            ADD_VEHICLE_COST,
                                                            t_D,
                                                            "I",
                                                            obj_val_init_vrptw_heur,
                                                            obj_val_DVRPTW_heur,
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
                                                        print("\n Results...")

                                                        exp_log_err += log_experiment_result(result_tD)
                                                        route_log_err += log_routes(get_routes(solution_ort, routing_ort, manager_ort),dataset_name,run_id,t_D)
                                                        all_exp_FULL.append(result_tD)

                                                    
                                                    else:
                                                        # print(f"\n[Simulation] No Solution from VRPTW, trying with DVRPTW")
                                                        print(f"\n[Simulation] DVRPTW ...")
                                                        SolutionVRPTW = Solve_DVRPTW_Heuristics(
                                                            data_model_VRPTW,
                                                            solver_time_limit=t_s,
                                                            stakeholder_input=lambda_value["values"]
                                                            )
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
                                                            # --------------------------------------------------------------
                                                            # REOPT TD-D-VRPTW TODO -> further research
                                                            # --------------------------------------------------------------
                                                            # Heuristics don't solve to optimality and no depot in start node
                                                            # if solver_status != 7 and len(eff_starts&eff_depots) == 0:
                                                            # if len(eff_starts&eff_depots) == 0:
                                                            #     initial_solution_arcs =  get_all_arcs_np(data_model_VRPTW,manager_ort,routing_ort,solution_ort) 
                                                            #     # print("\nInitial arcs:")
                                                            #     # print(initial_solution_arcs)
                                                            #     obj_val_DVRPTW_ex, _ = Solve_DVRPTW_Exact(   
                                                            #         data_model_VRPTW,
                                                            #         solver_time_limit = t_s,
                                                            #         initial_solution = initial_solution_arcs
                                                            #     )

                                                            # only DVRPTW solvable after t_D
                                                            result_tD = log_exp_res(
                                                                dataset_name,
                                                                run_id,
                                                                t_s,
                                                                lshape,
                                                                lscale,
                                                                EARLY_SERVICE_PENALTY,
                                                                LATE_SERVICE_PENALTY,
                                                                OVERTIME_PENALTY,
                                                                CUSTOMER_OUTSOURCE_PENTALY,
                                                                ADD_VEHICLE_COST,
                                                                t_D,
                                                                lambda_value["mode"],
                                                                obj_val_init_vrptw_heur,
                                                                obj_val_DVRPTW_heur,
                                                                arrival_sum_init,
                                                                arrival_sum_vrptw,
                                                                arrival_sum_dvrptw,
                                                                ov_details["ov_operating_costs"],
                                                                ov_details["ov_overtimes"],
                                                                ov_details["ov_early_service"],
                                                                ov_details["ov_late_service"],
                                                                ov_details["ov_outsource_customer"],
                                                                ov_details["ov_additional_vehicles"],                                                               
                                                                ov_details["count_overtimes"],                                                                    
                                                                ov_details["count_early_service"],                                                                    
                                                                ov_details["count_late_service"],                                                                    
                                                                ov_details["count_outsource_customer"],                                                                    
                                                                ov_details["count_additional_vehicles"],                                                                                                                                    
                                                            )
                                                            print("\n Results...")
                                                            # pprint(result_tD)
                                                            
                                                            all_exp_FULL.append(result_tD)
                                                            exp_log_err += log_experiment_result(result_tD)
                                                            route_log_err += log_routes(get_routes(solution_ort,routing_ort, manager_ort),dataset_name,run_id,t_D)

                                                            solution_VRPTW = {}
                                                            # print("get_routes_and_times")
                                                            solution_VRPTW = get_routes_and_times(data_model_VRPTW, manager_ort, routing_ort, solution_ort)
                                                            # print("copy.deepcopy(data_model_VRPTW['time_matrix'])")
                                                            previous_time_matrix = copy.deepcopy(data_model_VRPTW['time_matrix'])

                                                        else:
                                                            obj_val_DVRPTW_heur = None
                                                            pprint(data_model_VRPTW, width=175)
                                                            # check_vrptw_infeasibilities(data_model_VRPTW)
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
                                                    simulation_period += 1
                                                else:
                                                    print("Continue as usual") if po else None

                                                if simulation_period == max_simulations:
                                                    print(f"\n[Simulation] [{timestamp()}] {run_id} Successfully Completed (max it)") #if po else None
                                                    print("\nEffective Final Deviation Deltas") if po else None
                                                    print(previous_time_matrix-original_time_matrix) if po else None
                                                
                                                else:
                                                    print(f"\n[Simulation] [{timestamp()}] End Current Period") if po else None
                                                print("-"*80) if po else None
                                                simulation_period += 1
                    
                                            total_runs += 1

            # if len(time_of_disruption_per_cycle) > 1:
            # plot_multiple_cycles(all_routes_per_cycle, data_per_cycle, time_of_disruption_per_cycle)
    
    
    # pprint(all_exp_FULL)
    # log_experiment_result(all_exp_FULL)time_delta = end_time - start_time
    end_time = time.time()
    time_delta = end_time - start_time
    hours, rem = divmod(time_delta, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n[Experiment] Error during route logs = {route_log_err}, result logs = {exp_log_err}")
    print(f"\n[Experiment] {start_time_stamp} - {timestamp()} Concluded :-)")
    print(f"\n[Experiment] Duration: {int(hours)} hour(s), {int(minutes)} minute(s), {seconds:.2f} second(s)")

if __name__ == "__main__":
    main()
    

    
        

    