from ucb1 import *
from exampleReader import *
import utilities
import time
import numpy as np

#filenames = ['examples_numSequences=10_sequenceLength=25000_numStages=10.csv']
filenames = ['examples_numSequences=500_sequenceLength=2000_numStages=10.csv','examples_numSequences=500_sequenceLength=2000_numStages=15.csv','examples_numSequences=500_sequenceLength=2000_numStages=30.csv']

for filename in filenames:
    print '==========================='
    print filename    
    exampleReader = ExampleReader(filename, num_folds=1, abc_outputs=True)
    sequences = exampleReader.getSequences()
    
    sequences_processed = 0
    
    fracs_correct = []
    fracs_fully_correct = []
    util_losses = []
    predict_times = []
    exhaustive_optimization_times = []
    
    for sequence in sequences:
        num_fully_correct = 0
        num_fully_correct_denominator = 0
        num_stages_correct = 0
        num_stages_correct_denominator = 0
                
        throughput_sequence = sequence[0]
        const_inputs = sequence[1]
        optimal_action_sequence = sequence[2]
        sequence_id = sequence[3]
        
        n = (len(const_inputs)-2)/3
        rtt = const_inputs[0]
        stage_local_comp_times = const_inputs[1:1+n]
        stage_remote_comp_times = const_inputs[1+n:1+2*n]
        stage_msg_sizes = const_inputs[1+2*n:]
        
        actions = []
        
        minBeta = 0
        for alpha in range(0, n):
            for beta in range(minBeta, n-alpha+1):
                if beta == 0:
                    minBeta = 1
                gamma = n - beta - alpha
                actions.append((alpha, beta, gamma))

        ucb_learner = UCB1(len(actions))
        
        for i in range(len(throughput_sequence)):
            throughput = throughput_sequence[i]
            
            start_time = time.clock()
            action_index, bound = ucb_learner.select()
            end_time = time.clock()
            ucb_time = 1000*(end_time-start_time)
            
            selected_action = actions[action_index]
            selected_action_util = utilities.evaluate_abc(selected_action, throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes)
            selected_action_reward = 2 * (-0.5 + selected_action_util)
            
            start_time = time.clock()
            ucb_learner.update(action_index, selected_action_reward)
            end_time = time.clock()
            ucb_time = ucb_time + 1000*(end_time-start_time)
            predict_times.append(ucb_time)
            
            start_time = time.clock()
            optimal_action_abc, optimal_action_n, optimal_util = utilities.find_optimial(throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes)
            end_time = time.clock()
            exhaustive_optimization_time = 1000*(end_time-start_time)
            exhaustive_optimization_times.append(exhaustive_optimization_time)
            
            optimal_reward = 2 * (-0.5 + optimal_util)            
            
            util_loss = optimal_util - selected_action_util
            util_losses.append(util_loss)
            
            #print optimal_action_abc, list(selected_action)
            
            fully_correct = list(selected_action) == optimal_action_abc
            num_fully_correct_denominator = num_fully_correct_denominator + 1
            if fully_correct:
                num_fully_correct = num_fully_correct + 1
            
            num_stages_correct = num_stages_correct + n - utilities.err_n(optimal_action_abc, selected_action)
            num_stages_correct_denominator = num_stages_correct_denominator + n
        
        fracs_correct.append(float(num_stages_correct) / float(num_stages_correct_denominator))
        fracs_fully_correct.append(float(num_fully_correct) / float(num_fully_correct_denominator))
        
        sequences_processed = sequences_processed + 1
            
    print 'avg_frac_correct', np.mean(fracs_correct), 'stddev', np.std(fracs_correct)
    print 'avg_frac_fully_correct', np.mean(fracs_fully_correct), 'stddev', np.std(fracs_fully_correct)
    print 'avg_util_loss', np.mean(util_losses), 'stddev', np.std(util_losses)
    print 'avg_predict_time', np.mean(predict_times), 'stddev', np.std(predict_times)
    print 'avg_exhaustive_optimization_time', np.mean(exhaustive_optimization_times), 'stddev', np.std(exhaustive_optimization_times)
        
