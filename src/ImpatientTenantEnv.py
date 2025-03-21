__credits__ = ["Anthony Kiggundu"]

from collections import OrderedDict
from enum import Enum
#from gym import spaces
from gymnasium.spaces import Text
from RenegeJockey import RequestQueue, Queues, Observations
from gymnasium.core import ActType, ObsType
import numpy as np
import random

#import gymnasium
import gymnasium as gym
from gymnasium import spaces
import torch.optim as optim
import json


LR = .001  # Learning rate
SEED = None  # Random seed for reproducibility
MAX_EPISODES = 10000  # Max number of episodes

#
# Train
#

class Actions(Enum):
    RENEGE = 0
    JOCKEY = 1


class ImpatientTenantEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        #self.size = size  # The size of the square grid
        #self.window_size = 512  # The size of the PyGame window
        # Environment is made of the queues with the apparent xteristics ()
        self.queue_state = {}
        self.action = ""  
        self.utility_basic = 1.0
        self.discount_coef = 0.1
        time_entrance = 0.0
        pos_in_queue = 0
        self.history = {}
        # self.exp_time_service_end = 
        self.queueObj = Queues()
        self.Observations = Observations()        
        
        self.endutil = 0.0 
        self.intensity = 0.0
        self.jockey = 0.0 #False
        self.queuesize = 0.0
        self.renege = 0.0 #False
        self.reward = 0.0 
        self.servrate = 0.0
        self.waitingtime = 0.0
        self.ren_state_after_action = {}
        self.jock_state_after_action = {}

        # Observations are dictionaries with information about the state of the servers.
        # {EndUtility, Intensity, Jockey, QueueSize, Renege, Reward, ServRate, Waited}

        # We have 2 actions, corresponding to "Renege", "Jockey"
        # Since gym does not allow for space defns with string,
        # Renege = 0, Jockey = 1
        
        self.requestObj = RequestQueue(self.utility_basic, self.discount_coef)
        duration = 3
        self.requestObj.run(duration)
        
        self.QueueObj = Queues()

        self.queue_id = self.requestObj.get_curr_queue_id()

        srv1, srv2 = self.requestObj.get_queue_sizes() #self.srvs.get("Server2")

        if srv1 < srv2:
            low = srv1
            high = srv2
        else:
            low = srv2
            high = srv1            

        self.action_space = spaces.Discrete(2) # spaces.Box(low=0.0,high=1.0, shape=(1,), dtype=np.float32) # spaces.Discrete(2)
        
        self.history = self.requestObj.get_history()

        serv_rate_one, serv_rate_two = self.requestObj.get_server_rates()
         
        #queue_one_state = np.array([srv1, serv_rate_one, self.reward, self.action])
        #queue_two_state = np.array([srv2, serv_rate_two, self.reward, self.action])
          
        self.observation_space = spaces.Dict ({                                                               
		    "ServerID": spaces.Box(low=1,high=2, shape=(1,), dtype=np.float32), #spaces.Text(8),
            "Renege": spaces.Box(low=0.0,high=1.0, shape=(1,), dtype=np.float32), # [true=1, false= 0] # spaces.Discrete(1), # 
            "ServRate": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5), self.QueueObj.get_arrivals_rates()
            "Intensity": spaces.Box(low=0.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
            "Jockey": spaces.Box(low=0.0,high=1.0, shape=(1,), dtype=np.float32), # [true=1, false= 0] spaces.Discrete(1), spaces.Discrete(1), #
            "Waited": spaces.Box(low=-1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(7),
            "EndUtility": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), # spaces.Text(8),
            "Reward": spaces.Box(low=0.0,high=1.0, shape=(1,), dtype=np.float32), #spaces.Text(1),
            "QueueSize": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
        })


        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # ToDo :: here dpending on the to_from mapping, set the right values
        # For now we set the values as below

        self._action_to_state = {
            Actions.RENEGE.value: self.get_renege_action_outcome(), 
            Actions.JOCKEY.value: self.get_jockey_action_outcome()
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

    
    '''
        need to compute observations both in reset and step
        method _get_obs that translates the environment’s state into an observation
    '''
    # The agent was in a given queue at a time t, then took some action and received some reward

    def _get_obs(self):
		
        from gymnasium.vector.utils import create_empty_array
		
        n_envs = 9
		
        obs = create_empty_array(self.observation_space, n=n_envs, fn=np.zeros )        

        return obs
	
	
    def _get_info(self):

        return self._action_to_state 
        
        
    def get_matched_dict(self):
		
        jockeyed_matched_dict = []           
        reneged_matched_dict = []   
                            
        if "Server1" in self.queue_id:
            _id_ = 1
            self.queuesize = len(self.requestObj.get_curr_queue())
        else:
            _id_ = 2
            self.queuesize = len(self.requestObj.get_curr_queue())
		
        #for item in self.history:
        for item in self.requestObj.get_curr_obs_renege():
            hist_id = item.get("ServerID")
            size = item.get("QueueSize")           
            #reneged = item.get("Renege")
                                 
            if "_reneged" in self.requestObj.customerid:
                if size == self.queuesize and hist_id == _id_: # and reneged:
                    reneged_matched_dict.append(item)
                   
        
        for item in self.requestObj.get_curr_obs_jockey():
            hist_id = item.get("ServerID")
            size = item.get("QueueSize")
            #jockeyed = item.get("Jockey")             
            
            if "_jockeyed" in self.requestObj.customerid:      
                if size == self.queuesize and hist_id == _id_: # and jockeyed:            
                    jockeyed_matched_dict.update(item)                
                    
        return reneged_matched_dict, jockeyed_matched_dict
            
    
    def reset(self, seed=None, options=None): # -> tuple[ObsType, dict[str, any]] :    action_to_state    
		
        # We need the following line to seed self.np_random
        super().reset(seed=seed) # seed  
           			                             
        observation = self.observation_space.sample()         
        info = self._get_info()                                 
                       
        return observation, info
        
    
    def get_renege_action_outcome(self):
	    	
        curr_state = self.requestObj.get_queue_curr_state()
        # print("\n CURRENT STATE: ", curr_state, " TYPED =? ", type(curr_state))        
        
        srv = curr_state.get('ServerID')
        # print("\n -----> ", self.requestObj.get_curr_obs_renege(srv))
        # observed_srv = 
        if srv == 1:
            #outcome_jockey_action
            if len(self.requestObj.get_curr_obs_renege(srv)) > 0:
                curr_state['QueueSize'] = self.queuesize-1
			    #queue_intensity = self.QueueObj.get_arrivals_rates()/srv1
                curr_state["Reward"] = self.requestObj.get_curr_obs_renege(srv)['reward']
			    #curr_state["Waited"] = requestObj.estimateMarkovWaitingTime(float(pose), queue_intensity, self.time)
        else:
            if len(self.requestObj.get_curr_obs_renege(srv)) > 0:		            
                curr_state['QueueSize'] = self.queuesize-1
			    #queue_intensity = self.QueueObj.get_arrivals_rates()/srv2
                curr_state["Reward"] = self.requestObj.get_curr_obs_renege(srv)['reward']
			    #requestObj.estimateMarkovWaitingTime(float(pose), queue_intensity, self.time)
        
        return curr_state
        
		
    def get_jockey_action_outcome(self):                    
    
        curr_state = self.requestObj.get_queue_curr_state()        
        
        srv = curr_state.get('ServerID')
        if srv == 1:
            #outcome_jockey_action
            if len(self.requestObj.get_curr_obs_jockey(srv)) > 0:
                curr_state['QueueSize'] = self.queuesize+1
                # queue_intensity = self.QueueObj.get_arrivals_rates()/srv1
                curr_state["Reward"] = self.requestObj.get_curr_obs_jockey(srv)['reward']
			    #curr_state["Waited"] = requestObj.estimateMarkovWaitingTime(float(pose), queue_intensity, self.time)
        else:
            if len(self.requestObj.get_curr_obs_jockey(srv)) > 0:		            
                curr_state['QueueSize'] = self.queuesize+1
                #queue_intensity = self.QueueObj.get_arrivals_rates()/srv2
                curr_state["Reward"] = self.requestObj.get_curr_obs_jockey(srv)['reward'] 
			    #requestObj.estimateMarkovWaitingTime(float(pose), queue_intensity, self.time),
        
        return curr_state
        
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
                
        # An episode is done iff the agent has reached the target
        # In our case, an episode is done if the customer has left the queue
        
        # terminated = np.array_equal(self._agent_location, self._target_location)
        
        # According to the network 1 is when terminated and 0 for still running simulation
        
        new_state = self._action_to_state[action] 
        # Terminal state is if the next action returns a reward
        # and that action leads to an empty queue.
         
        if new_state("QueueSize") > 0.0:
            terminated = False #  True
        else:
            terminated = True # False

        #reward = 1 if terminated else 0  # Binary sparse rewards
        reward = new_state['reward']
        observation = self._get_obs()
        info = self._get_info()
    
        if self.render_mode == "human":
            self._render_frame()
    
        counter = counter + 1


        return observation, reward, terminated, info  # False
