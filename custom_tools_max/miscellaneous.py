import random
def generate_unique_sorted_list(min_value=1, max_value=20, min_length=2, max_length=2):
    """To create a small list of random numbers for t_D testing"""
    list_length = random.randint(min_length, max_length)
    random_numbers = random.sample(range(min_value, max_value + 1), list_length)
    return sorted(random_numbers)

def repeating_sequence(d, v):
    sequence = list(range(d + 1))  # Create the sequence 0, 1, 2, ..., d
    result = sequence * int(v)  # Repeat the sequence v times
    return result

def log_exp_res(
        dataset: str = "unknwon_dataset",
        experiment_id: str = "unknown_id",
        # probability: float = 0,
        solver_time: int = 0,
        shape: float = 0.0,
        scale: float = 0.0,
        c_early: int = 0,
        c_late: int = 0,
        c_overtime: int = 0,
        c_outsource: int =0,
        c_add_veh: int=0,
        disruption_time: int = 0,
        t_D_policy: bool = True,
        VRPTW_objective_value_heuristics: int = 0,
        DVRPTW_objective_value_heuristics: int = 0,
        arrival_sum_init: int = 0,
        arrival_sum_vrptw: int = 0,   
        arrival_sum_dvrptw: int = 0,  
        OP_COST: int = 0,
        EARLY_SERVICE: int = 0,
        LATE_SERVICE: int = 0,
        OVERTIME: int = 0,
        CUSTOMER_OUTSOURCE: int = 0,
        ADD_VEHICLE: int = 0,
        count_overtimes: int = 0,
        count_early_service: int = 0,
        count_late_service: int = 0,
        count_outsource_customer: int = 0,
        count_additional_vehicles: int = 0,
    ) -> dict:

    res_dict = dict({
        "dataset": dataset,
        "exp_id": experiment_id,
        "std_op_cost": 1, 
        "c_early": int(c_early),
        "c_late": int(c_late),
        "c_overtime": int(c_overtime),
        "c_outsource": int(c_outsource),
        "c_add_veh": int(c_add_veh),
        # "p": int(probability),
        "log_shape": shape,
        "log_scale": scale,
        "t_s": int(solver_time),
        "t_D": int(disruption_time),
        "policy": t_D_policy,
        "arrival_sum_init": arrival_sum_init,
        "arrival_sum_vrptw": arrival_sum_vrptw,
        "arrival_sum_dvrptw": arrival_sum_dvrptw,
        "obj_VRPTW_heur": int(VRPTW_objective_value_heuristics),
        "obj_DRPTW_heur": int(DVRPTW_objective_value_heuristics),
        "obj_operating_cost": int(OP_COST),
        "obj_early_service": int(EARLY_SERVICE),
        "obj_late_service": int(LATE_SERVICE),
        "obj_overtime": int(OVERTIME),
        "obj_customer_drop": int(CUSTOMER_OUTSOURCE),
        "obj_add_vehicle": int(ADD_VEHICLE),
        "count_overtimes": int(count_overtimes),
        "count_early_service": int(count_early_service),
        "count_late_service": int(count_late_service),
        "count_outsource_customer": int(count_outsource_customer),
        "count_additional_vehicles": int(count_additional_vehicles),
    })
     
    return res_dict

from datetime import datetime, timezone, timedelta

def timestamp() -> str:
    """
    Creates current timestamp using GMT+2.
    Returns: [YYYY-MM-DD HH:MM:SS]
    """
    local_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=2)))
    return f"{local_time.strftime('%Y-%m-%d_%H-%M-%S')}"

#     Setup simple first experiment
# accumulate results in a json file (to convert to df later)
# Initial solution
# grade of dynamism
# Recovered solution
# handling not-solvable as None (clean, without crash)

# 1. remove disruptions when delay is gone
# 2. what's the sudden problem with the T updates?
