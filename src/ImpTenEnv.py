from collections import OrderedDict
from enum import Enum
import numpy as np
import random
from RenegeJockey import RequestQueue, Queues, Observations

class Actions(Enum):
    RENEGE = 0
    JOCKEY = 1
    SERVED = -1
    

class ImpatientTenantEnv:
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.queue_state = {}
        self.action = ""  
        self.utility_basic = 1.0
        self.discount_coef = 0.1
        self.history = {}
        # self.queueObj = Queues()
        self.Observations = Observations()        
        self.endutil = 0.0 
        self.intensity = 0.0
        self.jockey = 0.0
        self.queuesize = 0.0
        self.renege = 0.0
        self.reward = 0.0 
        self.servrate = 0.0
        self.waitingtime = 0.0
        self.ren_state_after_action = {}
        self.jock_state_after_action = {}
        self.requestObj = RequestQueue(self.utility_basic, self.discount_coef)
        #duration = 3
        #self.requestObj.run(duration)        
        self.queue_id = self.requestObj.get_curr_queue_id()
  
        self.action_space = 2  # Number of discrete actions
        self.history = self.requestObj.get_history()        

        self.observation_space = {
            "ServerID": ("1", "2"),
            "rate_jockeyed": (0.0, 1.0),            
            "this_busy": (0.0, 20),
            "expected_service_time": (0.0, 50.0),
            "time_service_took": (0.0, 50.0),
            "rate_reneged": (0.0, 1.0),
            "reward": (0.0, 1.0),
            "at_pose": (1.0, 50),
            "long_avg_serv_time": (0.0,50),
            "action": ("served","reneged","jockeyed"),
        }
        
       

        self._action_to_state = {
            Actions.RENEGE.value: self.get_renege_action_outcome(self.queue_id), 
            Actions.JOCKEY.value: self.get_jockey_action_outcome(self.queue_id)
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        

    def _get_obs(self):
        obs = {key: np.zeros(1) for key in self.observation_space.keys()}
        
        return obs
        

    def _get_info(self):
		
        return self._action_to_state 
        

    def reset(self, seed=None, options=None):
        random.seed(seed)
        np.random.seed(seed)
        observation = [0.0] * state_dim #self.Observations.get_obs() # self._get_obs()         
        info = self._get_info()                                 
        return observation, info
   
        

    def step(self, action):
        new_state = self._action_to_state[action]
        terminated = new_state["QueueSize"] <= 0.0
        reward = new_state['Reward']
        observation = self._get_obs()
        info = self._get_info()
        flattened_observation = np.concatenate([observation[key].flatten() for key in observation.keys()])

        if self.render_mode == "human":
            self._render_frame()
        
        return flattened_observation, reward, terminated, info        

 
    def get_state_action_info(self):
        state_action_info = {
            "state": self._get_obs(),
            "action_to_state": self._get_info() # self._action_to_state
        }
        
        return state_action_info
        

    def get_raw_server_status(self):
        raw_server_status = {
            "queue_id": self.queue_id,
            "queuesize": self.queuesize,
            "servrate": self.servrate,
            "waitingtime": self.waitingtime
        }
        return raw_server_status