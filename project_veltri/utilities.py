import math
import numpy as np

def util(est_t, est_E, t_min, t_max, E_min, E_max, lam=0.25):    
    util_t = 1 - ((est_t - t_min) / (t_max - t_min))
    if est_t <= t_min:
        util_t = 1
    elif est_t >= t_max:
        util_t = 0
        
    util_E = 1 - ((est_E - E_min) / (E_max - E_min))
    if est_E <= E_min:
        util_E = 1
    elif est_E >= E_max:
        util_E = 0
    
    return math.pow(util_t, lam) * math.pow(util_E, lam)

def evaluate_abc(abc_tuple, throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes):
    if len(abc_tuple) != 3:
        raise ValueError('tuple must have length 3')
    stage_assignements = []
    for _ in range(abc_tuple[0]):
        stage_assignements.append(0)
    for _ in range(abc_tuple[1]):
        stage_assignements.append(1)
    for _ in range(abc_tuple[2]):
        stage_assignements.append(0)
       
    return evaluate_n(stage_assignements, throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes)

def evaluate_n_smoothed(stage_assignements, throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes):
    alpha = 0
    beta = 0
    gamma = 0
    alpha_broke = False
    
    for i in range(len(stage_assignements)):
        if stage_assignements[i] == 1:
            alpha_broke = True
            break
        else:
            alpha = alpha + 1
    
    if alpha_broke:
        for i in range(len(stage_assignements)-1, -1, -1):
            if stage_assignements[i] == 1:
                break
            else:
                gamma = gamma + 1
        beta = len(stage_assignments) - alpha - gamma        
    
    return evaluate_abc((alpha, beta, gamma), throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes)

def evaluate_n(stage_assignments, throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes):   
    idle_power = 0.50
    loaded_power = 1.5
    transmit_power = 1.2589
    
    t_min = 0.25 * sum(stage_local_comp_times)
    t_max = 1.1 * sum(stage_local_comp_times)
    E_min = 0
    E_max = t_max/1000.0 * loaded_power
    
    local_comp_t = 0
    remote_comp_t = 0
    msg_transmit_bytes = 0
    msg_receive_bytes = 0
    rtts = 0  
    
    for i in range(len(stage_assignments)):
        if stage_assignments[i] == 0:
            local_comp_t = local_comp_t + stage_local_comp_times[i]
            if i > 0 and stage_assignments[i-1] == 1:
                msg_receive_bytes = msg_receive_bytes + stage_msg_sizes[i]
        else:
            remote_comp_t = remote_comp_t + stage_remote_comp_times[i]
            if i == 0 or stage_assignments[i-1] == 0:
                if rtts == 0:
                    rtts = 2
                else:
                    rtts = rtts + 1
                msg_transmit_bytes = msg_transmit_bytes + stage_msg_sizes[i]                       
            if i == (len(stage_assignments)-1):
                msg_receive_bytes = msg_receive_bytes + stage_msg_sizes[-1]

    msg_send_time = (8000.0 * msg_transmit_bytes) / throughput
    msg_receive_time = (8000.0 * msg_receive_bytes) / throughput
    rtt_time = rtts * rtt
    
    est_t = local_comp_t + remote_comp_t + msg_send_time + msg_receive_time + rtt_time
    est_E = loaded_power*local_comp_t/1000.0 + idle_power*(remote_comp_t + msg_receive_time + rtt_time)/1000.0 + transmit_power*msg_send_time/1000.0
        
    U = util(est_t, est_E, t_min, t_max, E_min, E_max)    
    return U 
   

def find_optimial(throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes):
    idle_power = 0.50
    loaded_power = 1.5
    transmit_power = 1.2589
    
    t_min = 0.25 * sum(stage_local_comp_times)
    t_max = 1.1 * sum(stage_local_comp_times)
    E_min = 0
    E_max = t_max/1000.0 * loaded_power
    
    N = len(stage_local_comp_times)    
    
    U_max = -1
    abc_U_max = None    
    minBeta = 0
    for alpha in range(0, N):
        for beta in range(minBeta, N-alpha+1):
            gamma = N - beta - alpha
            local_comp_t = 0
            remote_comp_t = 0
            
            for i in range(0, alpha):
                local_comp_t = local_comp_t + stage_local_comp_times[i]
                
            for i in range(alpha, alpha+beta):
                remote_comp_t = remote_comp_t + stage_remote_comp_times[i]

            for i in range(alpha+beta, alpha+beta+gamma):
                local_comp_t = local_comp_t + stage_local_comp_times[i]
                
            send_time = (8000.0 * stage_msg_sizes[alpha]) / throughput
            receive_msg_size = stage_msg_sizes[-1] if gamma == 0 else stage_msg_sizes[alpha+beta]
            receive_time = (8000.0 * receive_msg_size) / throughput
            
            comm_travel_time = 2 * rtt 
            
            est_t = local_comp_t
            if beta > 0:
                est_t = est_t + remote_comp_t + comm_travel_time + send_time + receive_time
            
            est_E = loaded_power * local_comp_t/1000.0
            if beta > 0:
                est_E = est_E + (idle_power * (remote_comp_t + receive_time + comm_travel_time)/1000.0) + (transmit_power * send_time/1000.0)
            
            if beta == 0:
                minBeta = 1 
            
            U = util(est_t, est_E, t_min, t_max, E_min, E_max)
            
            if U > U_max:
                U_max = U
                abc_U_max = [alpha, beta, gamma]
    
    optimal_assignements = []
    for _ in range(abc_U_max[0]):
        optimal_assignements.append(0)
    for _ in range(abc_U_max[1]):
        optimal_assignements.append(1)
    for _ in range(abc_U_max[2]):
        optimal_assignements.append(0)   
    return abc_U_max, optimal_assignements, U_max

def abc_to_n(abc):
    n = []
    for _ in range(abc[0]):
        n.append(0)
    for _ in range(abc[1]):
        n.append(1)
    for _ in range(abc[2]):
        n.append(0)
    return n

def err_n(abc_1, abc_2):
    n_1 = abc_to_n(abc_1)
    n_2 = abc_to_n(abc_2)
    num_error_stages = sum(np.abs(np.array(n_1)-np.array(n_2)))
    return num_error_stages
    