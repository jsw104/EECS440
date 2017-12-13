import csv
import random
from setuptools.dist import sequence

class ExampleReader:
    
    def __init__(self, filepath, num_folds=5, abc_outputs=False):
        if num_folds <= 0:
            raise ValueError('num_folds must be greater than zero')
        
        folds = []
        for _ in range(num_folds):
            folds.append([[],[],[]])
            
        with open(filepath, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            read_header = False
            
            curr_sequence_id = -1
            curr_fold_index = -1
            for row in reader:
                if not read_header:
                    read_header = True
                else:
                    sequence_id, inputs, outputs_abc, outputs_n, best_util = self._process_line(row, (len(row)-6)/4)
                    if sequence_id != curr_sequence_id:
                         curr_fold_index = random.randrange(0, num_folds)
                         curr_sequence_id = sequence_id
                        
                    folds[curr_fold_index][0].append(inputs)
                    folds[curr_fold_index][1].append(outputs_abc if abc_outputs else outputs_n)
                    folds[curr_fold_index][2].append(sequence_id)
        
        self.folds = folds
        
    def getFolds(self):
        return self.folds
    
    def getSequences(self):
        sequences = []
        curr_sequence = None
        
        for fold in self.folds:
            for i in range(len(fold[0])):
                inputs = fold[0][i]
                outputs = fold[1][i]
                sequence_id = fold[2][i]
                
                if curr_sequence is None or curr_sequence[3] != sequence_id:
                    if curr_sequence is not None:
                        sequences.append(curr_sequence)
                    curr_sequence = [[] ,inputs[1:], [], sequence_id]
                
                curr_sequence[0].append(inputs[0])
                curr_sequence[2].append(outputs)
                
        if curr_sequence is not None:
            sequences.append(curr_sequence)
                
        return sequences            
    
    def _process_line(self, line, n):
        sequence_id = int(line[0])
        line = line[1:]
            
        inputs = [None] * (3 + 3*n)
        outputs_abc = [None] * 3
        outputs_n = [None] * n 
        
        for i in range(3+3*n):
            inputs[i] = float(line[i])
        for i in range(3+3*n, 6+3*n):
            outputs_abc[i-(3+3*n)] = int(float(line[i]))
        for i in range(6+3*n, len(line)-1):
            outputs_n[i-(6+3*n)] = float(line[i])    
        
        best_util = line[-1]
    
        return sequence_id, inputs, outputs_abc, outputs_n, best_util