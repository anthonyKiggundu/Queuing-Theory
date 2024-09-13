import numpy as np
import random

import gymnasium as gym
import torch.optim as optim
import Request


LR = .01  # Learning rate
SEED = None  # Random seed for reproducibility
MAX_EPISODES = 10000  # Max number of episodes

# Init actor-critic agent
agent = A2C(gym.make('UngeduldigenKunden'), random_seed=SEED)

# Init optimizers
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

#
# Train
#


class ImpatientTenantEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, num_of_queues=2):
        #self.size = size  # The size of the square grid
        #self.window_size = 512  # The size of the PyGame window
        # Environment is made of the queues with the apparent xteristics ()
        self.dict_servers = {}
        self.num_of_queues = num_of_queues
        self.dict_queues = {}
        self.arrival_rates = [3,5,7,9,11,13,15]
        
        rand_idx = random.randrange(len(self.arrival_rates))
        self.sampled_arr_rate = self.arrival_rates[rand_idx]
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = {
                "currentQueue": "",
                "ServRate": float,
                "action": "",
                "reward": 0.0,
            }

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = ["jockey", "renege"]
        self.queue_state = None
        self.action = ""
        self.reward = 0.0

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_state = {
            "Reneged": [curr_queue, actual_latency, reward-1],
            "Jockeyed": [curr_queue, actual_latency, exp_latency, reward],
        }
        
        self.queue_setup_manager()
        self.generate_queues()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        #self.window = None
        #self.clock = None
        
    def queue_setup_manager(self):
                
        # deltalambda controls the difference between the service rate of either queues    
        deltaLambda=random.randint(1, 2)
        
        serv_rate_one=self.sampled_arr_rate + deltaLambda 
        serv_rate_two=self.sampled_arr_rate - deltaLambda

        _serv_rate_one=serv_rate_one / 2
        _serv_rate_two=serv_rate_two / 2
        
        self.dict_servers["Server1"] = _serv_rate_one
        self.dict_servers["Server2"] = _serv_rate_two
        
        print("\n Current Arrival Rate:", self.sampled_arr_rate, "ServerRate 1:", _serv_rate_one, "ServerRate 2:", _serv_rate_two) 
        
        return self.dict_servers        
    
    
    def generate_queues(self):
        
        for i in range(self.num_of_queues):
            code_string = "Server%01d" % (i+1)
            queue_object = np.array([])
            self.dict_queues.update({code_string: queue_object})

        return self.dict_queues
    
    '''
        need to compute observations both in reset and step
        method _get_obs that translates the environment’s state into an observation
    '''
    # The agent was in a given queue at a time t, then took some action and received some reward
    def _get_obs(self, customerID):
        action = get_action_taken(customerID)
        self.queue_state = get_current_sys_state(queueID)
        reward = get_reward_from_action()
        
        return {"Action": action, "State": queue_state, "Reward": reward}
    
    def get_current_sys_state(self):
        curr_sys_state = {}
        
        for key in self.dict_servers:
            serv_rate = self.dict_servers.values()
            queue_intensity = self.sampled_arr_rate / serv_rate
            estimated_latency = Request.estimateMarkovWaitingTime()
            self.curr_sys_state = {queueID:[self.action, queue_intensity, estimated_latency,self.reward],
                                   }
            
        return curr_sys_state
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        # Here we initialize the simulation with new arrival_rate and service_rates
        self.queue_setup_manager()
        # Choose the agent's location uniformly at random
        #self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
    
        # We will sample the target's location randomly until it does not coincide with the agent's location
        #self._target_location = self._agent_location
        #while np.array_equal(self._target_location, self._agent_location):
        #    self._target_location = self.np_random.integers(
        #        0, self.size, size=2, dtype=int
        #    )
    
        observation = self._get_obs()
        info = self._get_info()
    
        #if self.render_mode == "ansi":
        #   self._render_frame()
    
        return observation, info
    
    # ToDo:: Here action needs to be either renege or jockey
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
    
        if self.render_mode == "human":
            self._render_frame()
    
        return observation, reward, terminated, False, info
    
    

