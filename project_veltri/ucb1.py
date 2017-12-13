import math

class UCB1:
    
    def __init__(self, numActions):
        self.t = 0
        self.numActions = numActions
        self.rewards = [0.0] * numActions
        self.actionCounts = [0] * numActions
        
    def select(self):
        #print '========================'
        max_ucb = (-1,-1)
        for i in range(self.numActions):
            ucb = 1.0
            adjustment = 0.0
            if self.actionCounts[i] > 0 and self.t > 0:
                adjustment = math.sqrt(2 * math.log(self.t)/self.actionCounts[i])
                ucb = self.rewards[i] + math.sqrt(2 * math.log(self.t)/self.actionCounts[i])
            
            #print i, self.rewards[i], self.actionCounts[i], ucb, adjustment
            if ucb > max_ucb[1]:
                max_ucb = (i, ucb)
                
        return max_ucb
    
    def update(self, lastAction, lastReward):
        self.t = self.t + 1
        self.actionCounts[lastAction] = self.actionCounts[lastAction] + 1
        self.rewards[lastAction] = self.rewards[lastAction] + (1.0/self.actionCounts[lastAction] * (lastReward - self.rewards[lastAction]))
         