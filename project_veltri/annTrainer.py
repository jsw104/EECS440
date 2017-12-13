import random
import numpy as np
np.set_printoptions(threshold=np.nan)
import time

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from exampleReader import * 
import utilities

#filenames = ['examples_numSequences=250_sequenceLength=250_numStages=10.csv','examples_numSequences=250_sequenceLength=250_numStages=15.csv','examples_numSequences=250_sequenceLength=250_numStages=30.csv']
filenames = ['examples_numSequences=1000_sequenceLength=250_numStages=10.csv','examples_numSequences=1000_sequenceLength=250_numStages=15.csv','examples_numSequences=1000_sequenceLength=250_numStages=30.csv']
#filenames = ['examples_numSequences=2000_sequenceLength=250_numStages=10.csv','examples_numSequences=2000_sequenceLength=250_numStages=15.csv','examples_numSequences=2000_sequenceLength=250_numStages=30.csv']
ann_layer_configs = [(10,10),(50,50),(250,250),(500,500)]

for filename in filenames:
    
    num_folds = 5
    exampleReader = ExampleReader(filename, num_folds)
    folds = exampleReader.getFolds()
        
    print 'imported examples'
        
    for ann_layer_config in ann_layer_configs:
        print '=================================================================================='
        print 'file:', filename
        print 'ann_layer_config:', ann_layer_config
            
        fold_fracs_correct = []
        fold_fracs_fully_correct = []
        fold_avg_predict_times = []
        fold_avg_util_losses = []
        fold_exhaustive_optimization_times = []
        for test_fold_index in range(num_folds):
            print '***processing fold', test_fold_index+1
            
            training_inputs = []
            test_inputs = []
            training_outputs = []
            test_outputs = []
            
            for fold_index in range(num_folds):
                if fold_index == test_fold_index:
                    test_inputs = folds[fold_index][0]
                    test_outputs = folds[fold_index][1]
                else:
                    training_inputs = training_inputs + folds[fold_index][0]
                    training_outputs = training_outputs + folds[fold_index][1]
                    
            scaler = StandardScaler()
            normalized_training_inputs = scaler.fit_transform(training_inputs)
            normalized_test_inputs = scaler.transform(test_inputs)
            
            training_examples = (normalized_training_inputs, np.array(training_outputs).astype(int))
            test_examples = (normalized_test_inputs, np.array(test_outputs).astype(int))
        
            clf = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=ann_layer_config, early_stopping=True, max_iter=5000)
            clf.fit(training_examples[0], training_examples[1])
            
            start_time = time.clock()
            predictions = clf.predict(test_examples[0])
            end_time = time.clock()
            avg_predict_time = 1000*(end_time-start_time)/float(len(test_examples[0])) #in milliseconds
            fold_avg_predict_times.append(avg_predict_time)
            
            num_fully_correct = 0
            for prediction_index in range(predictions.shape[0]):
                prediction = predictions[prediction_index]
                if np.array_equiv(prediction, test_examples[1][prediction_index]):
                    num_fully_correct = num_fully_correct + 1
            fraction_fully_correct = num_fully_correct / float(predictions.shape[0]) 
            #print 'fraction_fully_correct', fraction_fully_correct
            fold_fracs_fully_correct.append(fraction_fully_correct)
    
            fraction_correct = 1 - (sum(sum(np.abs(predictions - test_examples[1]))) / float(predictions.shape[0] * predictions.shape[1]))
            #print 'fraction_correct', fraction_correct
            fold_fracs_correct.append(fraction_correct)
            
            total_util_loss = 0
            total_exhaustive_optimization_time = 0
            num_stages = (len(test_examples[0][0])-3)/3
            for i in range(len(test_examples[0])):
                example = test_inputs[i]
                throughput = example[0]
                rtt = example[1]
                stage_local_comp_times = example[2:2+num_stages]
                stage_remote_comp_times = example[2+num_stages:2+2*num_stages]
                stage_msg_sizes = example[2+2*num_stages:]
                
                prediction = predictions[i]
                prediction_util = utilities.evaluate_n(prediction, throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes)
                
                start_time = time.clock()
                _, _, best_util = utilities.find_optimial(throughput, rtt, stage_local_comp_times, stage_remote_comp_times, stage_msg_sizes)
                end_time = time.clock()
                total_exhaustive_optimization_time = total_exhaustive_optimization_time + 1000*(end_time-start_time)
                
                util_loss = best_util - prediction_util
                total_util_loss = total_util_loss + util_loss
                
            fold_avg_util_losses.append(total_util_loss/float(len(test_examples[0])))
            fold_exhaustive_optimization_times.append(total_exhaustive_optimization_time/float(len(test_examples[0])))
        
        print 'avg_frac_correct', np.mean(fold_fracs_correct), 'stddev', np.std(fold_fracs_correct)
        print 'avg_frac_fully_correct', np.mean(fold_fracs_fully_correct), 'stddev', np.std(fold_fracs_fully_correct)
        print 'avg_util_loss', np.mean(fold_avg_util_losses), 'stddev', np.std(fold_avg_util_losses)
        print 'avg_predict_time', np.mean(fold_avg_predict_times), 'stddev', np.std(fold_avg_predict_times)
        print 'avg_exhaustive_optimization_time', np.mean(fold_exhaustive_optimization_times), 'stddev', np.std(fold_exhaustive_optimization_times)
        
    