import numpy as np

class BanditAgent:

    def __init__(self, k, alpha=None, c=0, init=0):
        # k: number of arms
        # alpha: step-size (None -> sample average)
        # c: USB exploration constant ( 0 -> ε-greedy exploration)
        # init: optimist Q_0
        
        self.k = k
        self.alpha = alpha
        self.c = c
        self.Q = np.full(k, init) # Action values
        self.N = np.zeros(k, dtype=int) # N[a] = number of times arm `a` has been chosen
        self.t = 0 # Total time-steps (overall pulls made so far)
    
   