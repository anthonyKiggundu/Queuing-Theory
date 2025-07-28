"""
Minimal MarkovStateMachine module for testing purposes.
"""

class StateMachine:
    def __init__(self, orig=None):
        self.feature = None
        self.current_state = 0
        self.num_states = 2
        
    def updateState(self):
        pass
        
    def integratedEffectiveFeature(self, samples, start_belief, features):
        return 1.0