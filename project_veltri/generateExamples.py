import math
import numpy as np
np.set_printoptions(precision=3, threshold=np.nan)
import _random
import utilities 

def noisy_throughput_seq(n=1000, max_throughput=5e7, stddev_multiplier=0.05):
    throughputs = [None] * n
    for i in range(n):
        x = float(i)/n    
        true_throughput = (max_throughput/2.0) * math.sin(40*x/math.pi) + (max_throughput/2.0)
        noisy_throughput = np.random.normal(true_throughput, stddev_multiplier*true_throughput)
        throughputs[i] = noisy_throughput
    return np.array(throughputs)

def random_scenario(num_stages=20):
    rtt = np.random.uniform(low=20.0, high=700.0)
    stage_local_comp_times = []
    stage_remote_comp_times = []
    stage_msg_sizes = []
    
    for _ in range(num_stages):
        stage_local_comp_times.append(np.random.gamma(1.5,100.0))
    for local_comp_time in stage_local_comp_times:
        remote_speedup_factor = np.random.normal(15.0, 5.0)
        stage_remote_comp_times.append(local_comp_time/remote_speedup_factor)
    for i in range(num_stages+1):
        stage_msg_sizes.append(1000*np.random.gamma(3.0,1000.0)/math.sqrt(i+1))
    
    return rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes

num_stages_list = [10, 15, 30]#, 50]
for num_stages in num_stages_list:     
    num_sequences = 10 #2000
    sequence_length = 25000 #250
    
    # =========================== examples =========================== #
    # Dimension 1: throughput sequence
    # Dimension 2: sequence element number
    # Dimension 3: 
    #    Element 0: throughput in bits per second
    #    Element 1: rtt in ms
    #    Elements 2-num_stages+1: stage local computation times in ms
    #    Elements num_stages+2-2*num_stages+2: stage remote computation times in ms
    #    Elements 2*num_stages+2-end: stage message sizes in bytes
    examples = np.zeros((num_sequences, sequence_length, (3*num_stages)+3))
    
    # =========================== targets_n =========================== #
    # Dimension 1: throughput sequence
    # Dimension 2: sequence element number
    # Dimension 3: contains num_stages elements corresponding to the optimal offloading solution; 0 => local and 1 => remote 
    targets_n = np.zeros((num_sequences, sequence_length, num_stages))
    
    # =========================== targets_abc =========================== #
    # Dimension 1: throughput sequence
    # Dimension 2: sequence element number
    # Dimension 3: contains 3 elements corresponding to the optimal alpha, beta, and gamma values
    targets_abc = np.zeros((num_sequences, sequence_length, 3))
    
    # =========================== best_utils =========================== #
    # Dimension 1: throughput sequence
    # Dimension 2: sequence element number
    # Element Value: utility corresponding to best abc/n
    best_utils = np.zeros((num_sequences, sequence_length))
    
    
    for i in range(num_sequences):
        #if i%250 == 0:
        print 'generating sequence ', i
            
        tputs = noisy_throughput_seq(n=sequence_length)            
        examples[i,:,0] = tputs
        
        rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes = random_scenario(num_stages=num_stages)
        for j in range(sequence_length):
            tput = examples[i,j,0]
            examples[i,j,1] = rtt
            examples[i,j,2:num_stages+2] = stage_local_comp_times
            examples[i,j,num_stages+2:2*num_stages+2] = stage_remote_comp_times
            examples[i,j,2*num_stages+2:] = stage_msg_sizes
            
            optimal_abc, optimal_n, optimal_util = utilities.find_optimial(tput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes)
            
            targets_abc[i,j,:] = optimal_abc
            targets_n[i,j,:] = optimal_n
            best_utils[i,j] = optimal_util
    
    
    # =========================== Output Examples to CSV =========================== #
    filename = 'examples_numSequences=' + str(num_sequences) + '_sequenceLength=' + str(sequence_length) + '_numStages=' + str(num_stages) + '.csv' 
    with open(filename, 'w') as f:
        f.write('sequence_id,throughput,rtt')
        for i in range(num_stages):
           f.write(',local_comp_t_' + str(i))
        for i in range(num_stages):
           f.write(',remote_comp_t_' + str(i))
        for i in range(num_stages+1):
           f.write(',msg_size_' + str(i))
        f.write(',best_alpha,best_beta,best_gamma')
        for i in range(num_stages):
            f.write(',s_' + str(i))
        f.write(',best_util')
        f.write('\n')
        
        for sequence_id in range(examples.shape[0]):       
            sequence = examples[sequence_id]
            for timestep_id in range(examples.shape[1]):
                timestep_inputs = examples[sequence_id,timestep_id]
                timestep_targets_n = targets_n[sequence_id,timestep_id]
                timestep_targets_abc = targets_abc[sequence_id,timestep_id]
                best_util = best_utils[sequence_id,timestep_id]
                f.write(str(sequence_id))
                for input in timestep_inputs:
                    f.write(',' + str(input))
                for target_abc in timestep_targets_abc:
                    f.write(',' + str(target_abc))
                for target_n in timestep_targets_n:
                    f.write(',' + str(target_n))
                f.write(',' + str(best_util))
                f.write('\n')
        