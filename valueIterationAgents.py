import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            temp = util.Counter()
            for state in self.mdp.getStates():
                if(state == "TERMINAL_STATE"):#Check for terminal state.
                    temp[state] = 0
                    continue
                else: #Get maximum Q value and assign that as value of the state.
                    max_value = -100000000
                    for action in self.mdp.getPossibleActions(state):
                        if(self.computeQValueFromValues(state, action) > max_value):
                            max_value = self.computeQValueFromValues(state, action)
                    temp[state] = max_value
            self.values = temp

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        if(self.mdp.isTerminal(state)):#Using formula to compute Q value.
            return(0)
        for prob in self.mdp.getTransitionStatesAndProbs(state, action):
            q_val = q_val + prob[1]*(self.mdp.getReward(state, action, prob[0]) + self.discount*self.values[prob[0]])
        return(q_val)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if(state == "TERMINAL_STATE"):
            return(None)
        else:
            if(len(self.mdp.getPossibleActions(state)) > 0):
                temp_action = []
                temp_q = -100000000
                for action in self.mdp.getPossibleActions(state):#Computing direction of the largest Q value.
                    if(self.computeQValueFromValues(state, action) > temp_q):
                        temp_action = [action]
                        temp_q = self.computeQValueFromValues(state, action)
                    elif(self.computeQValueFromValues(state, action) == temp_q):
                        temp_action.append(action)
                        temp_q = self.computeQValueFromValues(state, action)
                return(temp_action[0])
            else:
                return(None)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
