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
        self.queueObj = Queues()
        self.Observations = Observations()        
        
        self.endutil = 0.0 
        self.intensity = 0.0
        self.jockey = False
        self.queuesize = 0.0
        self.renege = False
        self.reward = 0.0 
        self.servrate = 0.0
        self.waitingtime = 0.0

        # Observations are dictionaries with information about the state of the servers.
        # {EndUtility, Intensity, Jockey, QueueSize, Renege, Reward, ServRate, Waited}

        # We have 2 actions, corresponding to "Renege", "Jockey"
        # Since gym does not allow for space defns with string,
        # Renege = 0, Jockey = 1
        
        self.requestObj = RequestQueue(self.utility_basic, self.discount_coef)
        duration = 3
        self.requestObj.run(duration)

        self.queue_id = self.requestObj.get_curr_queue_id()

        srv1, srv2 = self.requestObj.get_queue_sizes() #self.srvs.get("Server2")

        if srv1 < srv2:
            low = srv1
            high = srv2
        else:
            low = srv2
            high = srv1            

        self.action_space = spaces.Discrete(2) #MultiBinary(2, seed=42)
        
        self.history = self.requestObj.get_history()

        serv_rate_one, serv_rate_two = self.requestObj.get_server_rates()
         
        queue_one_state = np.array([srv1, serv_rate_one, self.reward, self.action])
        queue_two_state = np.array([srv2, serv_rate_two, self.reward, self.action])
        			       
        #self.observation_space = spaces.Dict({
        #       "ServerID": spaces.Text(7), 
        #       "Status":
          
        self.observation_space = spaces.Dict ({                                                               
		    "ServerID": spaces.Box(low=1,high=2, shape=(1,), dtype=np.float32), #spaces.Text(8),
            "Renege": spaces.Discrete(1),
            "ServRate": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
            "Intensity": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
            "Jockey": spaces.Discrete(1),
            "Waited": spaces.Box(low=-1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(7),
            "EndUtility": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), # spaces.Text(8),
            "Reward": spaces.Box(low=0.0,high=1.0, shape=(1,), dtype=np.float32), #spaces.Text(1),
            "QueueSize": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
        })
       
                
       #self.observation_space = spaces.Dict ({
	   #	                "ServerID": spaces.Text(8), 
	   #	                "Status": spaces.Dict ({
       #                     "Renege": spaces.Discrete(1),        
       #                     "ServRate": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
       #                     "Intensity": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
       #                     "Jockey": spaces.Discrete(1),
       #                     "Waited": spaces.Box(low=-1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(7),
       #                     "EndUtility": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), # spaces.Text(8),
       #                     "Reward": spaces.Box(low=0.0,high=1.0, shape=(1,), dtype=np.float32), #spaces.Text(1),
       #                     "QueueSize": spaces.Box(low=1.0,high=np.inf, shape=(1,), dtype=np.float32), #spaces.Text(5),
       #                 })
       #         })
        

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # ToDo :: here dpending on the to_from mapping, set the right values
        # For now we set the values as below

        self._action_to_state = {
            Actions.RENEGE.value: self.requestObj.get_curr_obs_renege(),
            Actions.JOCKEY.value: self.requestObj.get_curr_obs_jockey(),
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
		
	    return {"ServerID": self.queue_id, "EndUtility": self.endutil, "Intensity": self.intensity, "Jockey": self.jockey,
                    "QueueSize": self.queuesize, "Renege": self.renege, "Reward": self.reward,  "ServRate": self.servrate, "Waited": self.waitingtime
               }
		
#        self.history = self.requestObj.get_history()
#        keys = []
#        values = []
#        observ = {}
#        print("\n ==============>> ", self.history)
#        for k, v in self.history.items():
#            if isinstance (v, OrderedDict):
#                for j, l in v.items():
#                    keys.append(keys, j)
#                    values.append(values, l[0])
#        
#        observ["ServerID"] = self.requestObj.get_curr_queue_id()
#        
#        observ["Status"] = dict(zip(keys, values))
#        
#        print("\n ******* ", observ)
#        
#        return observ # self.history

    
    def _get_info(self):

        return self._action_to_state 
            
    
    def reset(self, seed=None, options=None): # -> tuple[ObsType, dict[str, any]] :       
		
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # print("\n COMPARED STUFF: ", self.requestObj.get_history())
       
        #_dict = {"ServerID": spaces.Text(8), "EndUtility": 0.0, "Intensity": 0.0, "Jockey": False,
        #            "QueueSize": 0.0, "Renege": False, "Reward": 0.0,  "ServRate": 0.0, "Waited": 0.0
        #		}        
        
        #observation =  dict(random.choices(list(self.history.items()),k=1)) #_dict # self._get_obs() # [0]
        
        #if len(self.requestObj.get_curr_obs_jockey()) > 0:
        #    print("\n HISTORY JOCKEY =====> ", self.requestObj.get_curr_obs_jockey(), len(self.requestObj.get_curr_obs_jockey()))
        #    observation =  dict(random.choices(list(self.requestObj.get_curr_obs_jockey().items()),k=1)) 
        #elif len(self.requestObj.get_curr_obs_renege()) > 0:
        #    print("\n HISTORY RENEGED =====> ", self.requestObj.get_curr_obs_renege(), len(self.requestObj.get_curr_obs_renege()))
        #    observation =  dict(random.choices(list(self.requestObj.get_curr_obs_renege().items()),k=1))
			
        observation = random.choice(self.history) # self.requestObj.get_history()) # secrets
        info = self._get_info()
                 
        # print("\n OBSERVED: ",observation, "\n HISTORY: ", self.history)
               
        return observation, info              
    
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]

        hist = self.requestObj.get_history() # _curr

        counter = 1
        for act in action:

            result_from_action = self._action_to_state.get(act)
          
        # An episode is done iff the agent has reached the target
        # In our case, an episode is done if the customer has left the queue
        # ** or when a new batch of arrivals is being admitted to the queue **
        # ?? Decision for terminate: If time queue wait is greater than local wait
        # such that the tenant reneges -> terminate is true and a reward is given

        # terminated = np.array_equal(self._agent_location, self._target_location)
            if (hist[counter].get("EndUtility") < hist[counter].get("Waited")):
                # According to the network 1 is when terminated and 0 for still running simulation
                terminated = False #  True
            else:
                terminated = True # False

            reward = 1 if terminated else 0  # Binary sparse rewards
            observation = self._get_obs()
            info = self._get_info()
    
            if self.render_mode == "human":
                self._render_frame()
    
            counter = counter + 1


        return observation, reward, terminated, info  # False
    

