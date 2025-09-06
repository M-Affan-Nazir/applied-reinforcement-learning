import numpy as np

class BanditAgent:

    def __init__(self, k, alpha=None, c=0, init=0, ε = 0.1):
        # k: number of arms
        # alpha: step-size (None -> sample average)
        # c: USB exploration constant ( 0 -> ε-greedy exploration)
        # init: optimist Q_0
        
        self.k = k
        self.alpha = alpha
        self.c = c
        self.ε = ε
        self.Q = np.full(k, init) # Action values
        self.N = np.zeros(k, dtype=int) # N[a] = number of times arm `a` has been chosen
        self.t = 0 # Total time-steps (overall pulls made so far)
    
    def select_action(self):
        self.t += 1
        if self.c == 0:
            if np.random.rand() <= self.ε:
                action = np.random.randint(0, self.k)
                return action
            else:
                return np.argmax(self.Q)
        else:
            best_preference = np.NINF
            best_action_index = 0
            for i in range(self.k):
                if self.N[i] == 0:
                    preference = np.inf 
                    # Sets preference of an untried arm to infinity. 
                    # This makes sure that the arm is explored atleat once. 
                else:
                    preference = self.Q[i] + self.c * np.sqrt(np.log(self.t) / self.N[i])
                if preference > best_preference:
                    best_preference = preference
                    best_action_index = i
            return best_action_index
    
    def update(self, a, r):
        self.N[a] += 1
        step = (1/self.N[a]) if self.alpha is None else self.alpha
        self.Q[a] += step * (r - self.Q[a])