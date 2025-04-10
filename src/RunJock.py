###############################################################################
# Author: anthony.kiggundu
###############################################################################
from collections import OrderedDict
import numpy as np
import pygame as pyg
import scipy.stats as stats
import uuid
import time
import math
import sys
import random
import schedule
import gymnasium as gym
import threading
from tqdm import tqdm
import MarkovStateMachine as msm
from ImpTenEnv import ImpatientTenantEnv
from a2c import ActorCritic, A2CAgent
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime
from termcolor import colored
###############################################################################

#from RenegeJockey import RequestQueue, Queues, Observations

# Configure logging
logging.basicConfig(
    filename="request_decisions.log",
    filemode="a",
    format="%(asctime)s - Request ID: %(request_id)s - Queue ID: %(queue_id)s - Action: %(action)s",
    level=logging.INFO
)

################################## Globals ####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ImpatientTenantEnv()

state_dim = len(env.observation_space)
action_dim = env.action_space
# print("\n => ", state_dim, " = ", action_dim)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
		
        # print("\n FORWARDED STATE: ", state)
        action_probs = self.actor(state)
        state_value = self.critic(state)
        
        # Check for NaN values in action_probs and handle them
        if torch.isnan(action_probs).any():
            print("NaN values detected in action_probs:", action_probs)
            action_probs = torch.where(torch.isnan(action_probs), torch.zeros_like(action_probs), action_probs)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # Normalize to ensure valid probabilities
            
        return action_probs, state_value
        

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
                
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.rewards = []
        

    def select_action(self, state):
        # print("\n -->> ", state)
        if isinstance(state, dict):
            #state = np.concatenate([state[key].flatten() for key in state.keys()])
            # state = np.concatenate([np.array(state[key]).flatten() if hasattr(state[key], 'flatten') else [state[key]] for key in state.keys()])
            state = np.concatenate([
                np.array(state[key]).flatten() if hasattr(state[key], 'flatten') else np.array([state[key]], dtype=float)
                for key in state.keys() if isinstance(state[key], (int, float, np.number))
            ])
            
        # If state is not a tensor, create one; otherwise, use it as is.
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
    
        # Ensure the state tensor does not contain NaN values
        if torch.isnan(state).any():
            print("NaN values detected in state tensor:", state)
            state = torch.where(torch.isnan(state), torch.zeros_like(state), state)
                       
        state = state.unsqueeze(0).to(device)
        action_probs, state_value = self.model(state)
    
        # Check for NaN values in action_probs and handle them
        if torch.isnan(action_probs).any():
            print("NaN values detected in action_probs:", action_probs)
            action_probs = torch.where(torch.isnan(action_probs), torch.zeros_like(action_probs), action_probs)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # Normalize to ensure valid probabilities
            #print("*** After *** :", action_probs)
            
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        self.log_probs.append(action_dist.log_prob(action))
        self.values.append(state_value)
        
        return action.item()
        

    def store_reward(self, reward):
        self.rewards.append(reward)
        

    def update(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(device)
        log_probs = torch.stack(self.log_probs).to(device)
        values = torch.cat(self.values).to(device)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.values = []
        self.rewards = []


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
        self.queueObj = Queues()
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
        # print("\n Observed: ", obs)
        return obs
        

    def _get_info(self):
		
        return self._action_to_state 
        

    def reset(self, seed=None, options=None):
        random.seed(seed)
        np.random.seed(seed)
        observation = self.Observations.get_obs() # self._get_obs()         
        info = self._get_info()                                 
        return observation, info


    def get_renege_action_outcome(self, queue_id):
        curr_state = self.requestObj.get_queue_state(queue_id) # get_queue_curr_state()
        srv = curr_state.get('ServerID')
        
        if srv == 1:
            if len(self.Observations.get_obs()) > 0: #get_curr_obs_renege(srv)) > 0:
                curr_state['apt_pose'] = self.queuesize - 1
                curr_state["reward"] = self.Observations.get_curr_obs_renege(srv)[0]['reward']
        else:
            if len(self.Observations.get_obs()) > 0: # get_curr_obs_renege(srv)) > 0:
                curr_state['apt_pose'] = self.queuesize - 1
                curr_state["reward"] = self.Observations.get_curr_obs_renege[0]['reward']
                
        return curr_state
        

    def get_jockey_action_outcome(self, queue_id):
        curr_state = self.requestObj.get_queue_state(queue_id) # get_queue_curr_state()
        srv = curr_state.get('ServerID')
        
        if srv == 1:
            if len(self.Observations.get_obs()) > 0: #get_curr_obs_jockey(srv)) > 0:
                curr_state['apt_pose'] = self.queuesize + 1
                curr_state["reward"] = self.Observations.get_curr_obs_jockey(srv)[0]['reward']
        else:
            if len(self.Observations.get_obs()) > 0: # get_curr_obs_jockey(srv)) > 0:
                curr_state['at_pose'] = self.queuesize + 1
                curr_state["reward"] = self.Observations.get_curr_obs_jockey(srv)[0]['reward']
                
        return curr_state
        

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


class Queues(object):
    def __init__(self):
        super().__init__()
        
        self.num_of_queues = 2
        self.dict_queues = {}
        self.dict_servers = {}
        self.arrival_rates = [3,5,7,9,11,13,15]
        rand_idx = random.randrange(len(self.arrival_rates))
        self.sampled_arr_rate = self.randomize_arrival_rate() #self.arrival_rates[rand_idx] 
        self.queueID = ""             
        
        self.dict_queues = self.generate_queues()
        #self.dict_servers = self.queue_setup_manager()

        self.capacity = 50 #np.inf
        
    def randomize_arrival_rate(self):
		
        return random.choice(self.arrival_rates)
        
        
    def queue_setup_manager(self):
                
        # deltalambda controls the difference between the service rate of either queues    
        deltaLambda=random.randint(1, 2)
        
        serv_rate_one=self.sampled_arr_rate + deltaLambda 
        serv_rate_two=self.sampled_arr_rate - deltaLambda

        _serv_rate_one=serv_rate_one / 2
        _serv_rate_two=serv_rate_two / 2
                
        self.dict_servers["1"] = _serv_rate_one # Server1
        self.dict_servers["2"] = _serv_rate_two # Server2
        
        #print("\n Current Arrival Rate:", self.sampled_arr_rate, "Server1:", _serv_rate_one, "Server2:", _serv_rate_two) 


    def get_dict_servers(self):

        self.queue_setup_manager()
        
        return self.dict_servers        


    def get_curr_preferred_queues (self):
        # queues = Queues()
        #self.all_queues = self.generate_queues() #queues.generate_queues()

        curr_queue = self.dict_queues.get("1") # Server1
        alter_queue = self.dict_queues.get("2") # Server2

        return (curr_queue, alter_queue)

    
    def generate_queues(self):
        
        for i in range(self.num_of_queues):
            code_string = "%01d" % (i+1) #"Server%01d" % (i+1)
            queue_object = np.array([])
            self.dict_queues.update({code_string: queue_object})

        return self.dict_queues
        

    def get_dict_queues(self):
        
        return self.dict_queues
        
        
    def get_number_of_queues(self):

        return len(self.dict_queues)
        

    def get_arrivals_rates(self):

        return self.sampled_arr_rate
        
    
    def update_queue_status(self, queue_id):
		
        pass
		
		
    def get_queue_capacity(self):
	
	    return self.capacity

    
class Request:

    LEARNING_MODES=['stochastic','transparent' ] # [ online','fixed_obs', 'truncation','preemption']
    APPROX_INF = 1000 # an integer for approximating infinite
    # pyg.time.get_ticks()

    def __init__(self,uses_nn, time_entrance,pos_in_queue=0,utility_basic=0.0,service_time=0.0,discount_coef=0.0, outage_risk=0.1, # =timer()
                 customerid="", learning_mode='online',min_amount_observations=1,time_res=1.0,markov_model=msm.StateMachine(orig=None),
                 exp_time_service_end=0.0, serv_rate=1.0, dist_local_delay=stats.expon,para_local_delay=[1.0,2.0,10.0], batchid=0):  #markov_model=a2c.A2C, 
        
        # self.id=id #uuid.uuid1()
        self.customerid = "Batch"+str(batchid)+"_Customer_"+str(pos_in_queue+1)
        ## self.customerid = self.set_customer_id()
        # time_entrance = self.estimateMarkovWaitingTime()
        self.time_entrance=time_entrance #[0] # ToDo:: still need to find out why this turns out to be an array
        # self.time_last_observation=float(time_entrance)
        self.pos_in_queue=int(pos_in_queue)
        self.utility_basic=float(utility_basic)
        self.discount_coef=float(discount_coef)
        self.certainty=1.0-float(outage_risk)
        self.exp_time_service_end = exp_time_service_end
        self.time_exit = None  # To be set when the request leaves the queue
        self.service_time = service_time
        self.uses_nn = uses_nn
        self.reneged = False
        self.jockeyed = False
        self.actor_critic = None # ""

        if (self.certainty<=0) or (self.certainty>=1):
            raise ValueError('Invalid outage risk threshold! Please select between (0,1)')
        #if Request.LEARNING_MODES.count(learning_mode)==0:
        #   raise ValueError('Invalid learning mode! Please select from '+str(Request.learning_modes))
        #else:
        #    self.learning_mode=str(learning_mode)
            
        self.min_amount_observations=int(min_amount_observations)
        self.time_res=float(exp_time_service_end)
        self.markov_model=msm.StateMachine(orig=markov_model) # markov_model #
        if learning_mode=='transparent':
           self.serv_rate=self.markov_model.feature
        else:
           self.serv_rate=float(serv_rate)
           
        queueObj = Queues()

        queue_srv_rates = queueObj.get_dict_servers()

        if queue_srv_rates.get("1"):# Server1
            self.serv_rate = queue_srv_rates.get("1") # Server1
        else:
            self.serv_rate = queue_srv_rates.get("2") # Server2

        self.dist_local_delay=dist_local_delay
        self.loc_local_delay=np.random.uniform(low=float(para_local_delay[0]),high=(para_local_delay[1]))
        self.scale_local_delay=float(para_local_delay[0]) #2
        self.max_local_delay=self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay)
        self.max_cloud_delay=float(queueObj.get_arrivals_rates()/self.serv_rate) # np.inf
       
        # print("\n ****** ",self.loc_local_delay, " ---- " , self.time_entrance-arr_prev_times[len(arr_prev_times)-1])
        self.observations=np.array([])
        self.error_loss=1
        self.optimal_learning_achieved=False

        return
    
    
    def renege(self):
        """Request leaves the queue before service."""
        self.reneged = True
        #self.log_action("Reneged")
        

    def jockey(self, new_queue):
        """Request switches to another queue."""
        self.jockeyed = True
        new_queue.enqueue(self)
        #self.log_action(f"Jockeyed to Queue {new_queue.queue_id}")
        

    # def learn(self,new_pos,new_time): 
    def generate_observations (self):
        steps_forward=self.pos_in_queue-int(new_pos)
        # self.time_last_observation=float(new_time)
        self.pos_in_queue=int(new_pos)
        self.observations=np.append(self.observations,(new_time-self.time_entrance-np.sum(self.observations))/steps_forward)
        
        if not self.makeRenegingDecision():
            self.makeJockeyingDecision()
            return 
        else:
            self.makeRenegingDecision()
            return 
            

    def estimateMarkovWaitingTime(self, pos_in_queue, features):
        # print("   Estimating Markov waiting time...")
        queue_indices=np.arange(pos_in_queue)+1 # self.pos_in_queue-1)+1
        samples=1
        start_belief=np.matrix(np.zeros(2).reshape(1, 2)[0], np.float32).T #np.matrix(np.zeros(self.markov_model.num_states).reshape(1,self.markov_model.num_states)[0],np.float64).T
        print("\n FIRST BELIEF: ", start_belief)
        start_belief[self.markov_model.current_state]=1.0
        # print("\n NEXT BELIEF: ", start_belief)
        cdf=0        
        while cdf<=self.certainty:
            eff_srv=self.markov_model.integratedEffectiveFeature(samples, start_belief, features)
            cdf=1-sum((eff_srv**i*np.exp(-eff_srv)/np.math.factorial(i) for i in queue_indices))
            # print([eff_srv,cdf])
            samples+=1
        return (samples-1)*self.time_res


        #OrderedDict

    def makeRenegingDecision(self):
        # print("   User making reneging decision...") ACTION
        decision=False
        if self.learning_mode=='transparent':
            self.max_cloud_delay=stats.erlang.ppf(self.certainty,a=self.pos_in_queue,loc=0,scale=1/self.serv_rate)
            #self.max_cloud_delay=self.estimateMarkovWaitingTime()
        else:
            num_observations=self.observations.size
            mean_interval=np.mean(self.observations) # unbiased estimation of 1/lambda where lambda is the service rate
            if np.isnan(mean_interval):
                mean_interval=0
            if mean_interval!=0:
                self.serv_rate=1/mean_interval
            k_erlang=self.pos_in_queue*num_observations
            scale_erlang=mean_interval*k_erlang
            #mean_wait_time=mean_interval*self.pos_in_queue
            if np.isnan(mean_interval):
                self.max_cloud_delay=np.Inf
            else:
                self.max_cloud_delay=stats.erlang.ppf(self.certainty,loc=0,scale=mean_interval,a=self.pos_in_queue)
        #if self.learning_mode=='truncation':
        #    decision=False
        # elif self.learning_mode=='preemption':
        #    decision=False
        #elif self.learning_mode=='transparent':
        #    decision=(self.max_local_delay<=self.max_cloud_delay)
        #elif self.learning_mode=='fixed_obs':
        #    decision=(self.max_local_delay<=self.max_cloud_delay) & (num_observations>=self.min_amount_observations)
        #elif scale_erlang==0:
        #    decision=False
        #else: # mode='learning' , scale_erlang>0
        
            if self.max_local_delay <= self.max_cloud_delay: # will choose to renege
                decision=True
               #print('choose to rng')
                temp=stats.erlang.cdf(np.arange(self.max_local_delay,step=self.time_res),k_erlang,scale=scale_erlang)
                error_loss=np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))-np.sum(np.append([temp[0]],np.diff(temp))*np.exp(-self.pos_in_queue/np.arange(self.max_local_delay,step=self.time_res)))
            else:   #will choose to wait and learn
                decision=False
                #print('choose to wait')
                temp=stats.erlang.cdf(np.arange(self.max_local_delay,self.APPROX_INF+self.time_res,step=self.time_res),k_erlang,scale=scale_erlang)
                error_loss=np.sum(np.diff(temp)*np.exp(-self.pos_in_queue/np.arange(self.max_local_delay+self.time_res,self.APPROX_INF+self.time_res,step=self.time_res)))-np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))
                
            dec_error_loss = self.error_loss - error_loss
            self.error_loss = error_loss
            
            if dec_error_loss > 1-np.exp(-mean_interval):
                decision = False
            #else:
                #self.optimal_learning_achieved=True
                #print(self.observations)
            #if (not self.optimal_learning_achieved):
                self.min_amount_observations=self.observations.size+1
                # print(self.min_amount_observations)
        return decision
        
    
    # Extensions for the Actor-Critic modeling
    def makeJockeyingDecision(self, req, curr_queue, alt_queue):
        # We make this decision if we have already joined the queue 
        # First we analyse our current state -> which server, server intensity and expected remaining latency
        # Then we get information about the state of the alternative queue 
        # Evaluate input from the actor-critic once we get in the alternative queue
        decision=False                            
        expectedJockeyWait = self.generateExpectedJockeyCloudDelay(req)
        
        if expectedJockeyWait < estimateMarkovWaitingTime():             
            np.delete(curr_queue, np.where(id_queue==req_id)[0][0])
            reward = 1.0 
            dest_queue = np.append( dest_queue, req)
            obs_entry = self.objObserve(False,True,self.time-req.time_entrance, self.end_utility, len(curr_queue))#reward,req.min_amount_observations)
            # self.history = np.append(self.history,obs_entry)
            decision = True
            
        # ToDo:: There is also the case of the customer willing to take the risk
        #        and jockey regardless of the the predicted loss -> Customer does not
        #        care anymore whether they incur a loss because they have already joined anyway
        #        such that reneging returns more loss than the jockeying decision
        
        else:
            decision = False
            # ToDo:: revisit this for the case of jockeying.
            #        Do not use the local cloud delay
            reward = -1.0
            obs_entry = self.objObserve(False,False,self.time-req.time_entrance, self.end_utility, len(curr_queue)) # reward, req.min_amount_observations)
            self.min_amount_observations=self.observations.size+1
        
        return decision
        

    def get_time_entrance(self):

        return self.time_entrance
        
        
    def set_customer_id(self):
		
        self.customerid = uuid.uuid4()
        
    
    def get_customer_id(self):
		
        return self.customerid
        
    
    def log_action(self, action):
        """Logs the request action to the file."""
        logging.info("", extra={"request_id": self.request_id, "queue_id": self.queue_id, "action": action}) # MATCH


class Observations:
    def __init__(self, reneged=False, curr_pose=0, intensity=0.0, jockeying_rate=0.0, reneging_rate=0.0, jockeyed=False, time_waited=0.0,end_utility=0.0, reward=0.0, long_avg_serv_time=0.0): # reward=0.0, 
        self.reneged=reneged
        #self.serv_rate = serv_rate
        self.queue_intensity = intensity
        self.jockeyed=jockeyed
        self.time_waited=float(time_waited)
        self.end_utility=float(end_utility)
        self.reward= reward # id_queue
        self.queue_size=curr_pose
        self.obs = OrderedDict() #{} # self.get_obs()  
        self.curr_obs_jockey = []
        self.curr_obs_renege = []
        self.long_avg_serv_time = long_avg_serv_time 
        self. jockeying_rate = jockeying_rate
        self.reneging_rate = reneging_rate

        return


    def set_obs (self, queue_id,  curr_pose, intensity, jockeying_rate, reneging_rate, time_in_serv, time_to_service_end, reward, activity, long_avg_serv_time): 
        		
        if queue_id == "1": # Server1
            _id_ = 1
        else:
            _id_ = 2
			
        self.obs = {
			        "ServerID": _id_, #queue_id,
                    "at_pose": curr_pose,
                    "rate_jockeyed": jockeying_rate,
                    "rate_reneged": reneging_rate,                    
                    "this_busy": intensity,
                    "expected_service_time":time_in_serv,
                    "time_service_took": time_to_service_end,
                    "reward": reward,
                    "action":activity,
                    "long_avg_serv_time": long_avg_serv_time
                }
              

    def get_obs (self):
        
        return self.obs
        
        
    #def set_renege_obs(self, curr_pose, queue_intensity, perc_reneged,time_local_service, time_to_service_end, reward, queueid, activity):		

    #    self.curr_obs_renege.append(
    #        {   
    #            "queue": queueid,
    #            "at_pose": curr_pose,
    #            "rate_jockeyed": jockeying_rate ,
    #            "rate_reneged": perc_reneged,
    #            "this_busy": queue_intensity,
    #            "expected_service_time":time_local_service,
    #            "time_service_took": time_to_service_end,
    #           "reward": reward,
    #            "action":activity
    #        }
    #    )
        
        
    def get_renege_obs(self, queueid, queue): # , intensity, pose): # get_curr_obs_renege
		
        renegs = sum(1 for req in queue if '_reneged' in req.customerid)        			       
	    
        return renegs #self.curr_obs_renege 
  
        
    #def set_jockey_obs(self, curr_pose, queue_intensity, perc_jockeyed, time_alt_queue, time_to_service_end, reward, queueid, activity):
        
    #    self.curr_obs_jockey.append(
    #        {
    #            "queue": queueid,
    #            "at_pose": curr_pose,
    #            "this_busy": queue_intensity,
    #            "perc_jockeyed": perc_jockeyed,
    #            "expected_service_time":time_alt_queue,
    #            "time_service_took": time_to_service_end,
    #            "reward": reward,
    #            "action":activity   			
    #        }
    #    )


class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, utility_basic, discount_coef, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon, exp_time_service_end=0.0,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, batchid=np.int16, uses_nn=False): # Dispatched
                 
        
        self.dispatch_data = {
            "server_1": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "long_avg_serv_time":[], "queue_intensity":[]},
            "server_2": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "long_avg_serv_time":[], "queue_intensity":[]}
        }
        self.markov_model=msm.StateMachine(orig=markov_model)
        self.customerid = customerid
        self.utility_basic=float(utility_basic)
        self.local_utility = 0.0
        self.compute_counter = 0
        self.avg_delay = 0.0
        self.batchid = batchid
        self.discount_coef=float(discount_coef)
        self.outage_risk=float(outage_risk)
        self.time=float(time)
        self.service_time  = 0
        self.init_time=self.time        
        self.learning_mode=str(learning_mode)
        self.alt_option=str(alt_option)
        self.min_amount_observations=int(min_amount_observations)
        self.dist_local_delay=dist_local_delay
        self.para_local_delay=list(para_local_delay)
        # (False, serv_rate, queue_intensity, False,self.time-time_entrance,self.generateLocalCompUtility(req), reward, req.min_amount_observations)       
        self.decision_rule=str(decision_rule)
        self.truncation_length=float(truncation_length)
        self.preempt_timeout=float(preempt_timeout)
        self.preempt_timer=self.preempt_timeout
        self.time_res=float(time_res)
        self.dict_queues_obj = {}
        self.dict_servers_info = {}
        self.history = [] 
        self.curr_obs_jockey = [] 
        self.curr_obs_renege = [] 
        self.uses_nn = uses_nn 
        self.long_avg_serv_time = 0.0 # intensity

        self.arr_prev_times = np.array([])
        self.queue_intensity = 0.0
        

        self.objQueues = Queues()
        self.objRequest = Request(self.uses_nn,time)
        self.objObserv = Observations()
        # self.env = ImpatientTenantEnv()
        #state_dim, action_dim = self.getStateActionDims()
        #self.actor_critic = ActorCritic(state_dim, action_dim).to(device)

        self.dict_queues_obj = self.objQueues.get_dict_queues()
        self.dict_servers_info = self.objQueues.get_dict_servers()
        self.jockey_threshold = 1
        self.renege_reward = 0.0
        self.jockey_reward = 0.0
        self.curr_state = {} # ["Busy","Empty"]

        self.arr_rate = 0.0 # self.objQueues.get_arrivals_rates()
        # self.arr_rate = self.objQueues.get_arrivals_rates()
        
        self.setActCritNet(state_dim, action_dim)
        self.actor_critic = self.getActCritNet()
        
        self.setAgent(state_dim, action_dim)
        self.agent = self.getAgent()
        
        self.all_times = []
        self.all_serv_times = []
        self.queueID = ""
        self.curr_req = ""

        # self.rng_pos_reg=np.array([])
        self.rng_counter=np.array([])
        if self.markov_model.feature!=None:
            self.srv_rate=self.markov_model.feature
        
        self.certainty=1.0-float(outage_risk)
        if (self.certainty<=0) or (self.certainty>=1):
            raise ValueError('Invalid outage risk threshold! Please select between (0,1)') # get_curr_obs_renege
        
        self.exp_time_service_end = exp_time_service_end
        #self.dist_local_delay=dist_local_delay
        self.loc_local_delay=np.random.uniform(low=float(para_local_delay[0]),high=(para_local_delay[1]))
        self.scale_local_delay=float(para_local_delay[2])
        self.max_local_delay=self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay)
        self.max_cloud_delay= np.inf  # float(self.arr_rate/self.serv_rate) #
        
        # self.observations=np.array([]) 
        self.error_loss=1
        
        self.capacity = self.objQueues.get_queue_capacity()
        self.total_served_requests_srv1 = 0
        self.total_served_requests_srv2 = 0
        
        self.queue = []
        self.jockeying_rate = 0.0
        self.reneging_rate = 0.0
        self.nn_subscribers = []  # Requests that use NN knowledge
        self.state_subscribers = []  # Requests that use raw queue state
        
        BROADCAST_INTERVAL = 5
        
        return
        
    
    #def enqueue(self, request, use_nn=False):
    #    if len(self.queue) < self.capacity:
    #        self.queue.append(request)
    #        if use_nn:
    #            self.nn_subscribers.append(request)
    #        else:
    #            self.state_subscribers.append(request)
    #        return True
    #    return False        
		
	
    def setActCritNet(self, state_dim, action_dim):
		
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        
        
    def setAgent(self, state_dim, action_dim):
		    
        self.agent =  A2CAgent(state_dim, action_dim) # .to(device)
        
		
    def getActCritNet(self):		       
        
        return self.actor_critic
        
    
    def getAgent(self):		       
        
        return self.agent 
        
                
    def setNormalReward(self, reward):
		
        self.reward = reward
		
		
    def setRewardJockey(self, reward):
		
        self.jockey_reward = reward
	
	
    def setRewardRenege(self, reward):
		
        self.renege_reward = reward
		
       
    def compute_reneging_rate(self, queue):
        """Compute the reneging rate for a given queue."""
        renegs = sum(1 for req in queue if '_reneged' in req.customerid)
        return renegs / len(queue) if len(queue) > 0 else 0
        

    def compute_jockeying_rate(self, queue):
        """Compute the jockeying rate for a given queue."""
        jockeys = sum(1 for req in queue if '_jockeyed' in req.customerid)
        return jockeys / len(queue) if len(queue) > 0 else 0
    
    
    def get_curr_request(self):
		
        return self.curr_req
        
    
    def get_matching_entries(self, queueid):
		
        lst_srv1 = []
        lst_srv2 = []        
        
        for hist in self.history:
            
            if str(hist.get('ServerID')) == str(queueid):
                lst_srv1.append(hist)                
                return lst_srv1
            else:
                lst_srv2.append(hist)                
                return lst_srv2
		
		   
    def get_queue_curr_state(self):
        matching_entries = self.get_matching_entries(self.queueID)
        print(f"Matching entries for queueID {self.queueID}: {matching_entries}")

        if not matching_entries:
            print("No matching entries found.")
            self.curr_state = {}
        else:
            self.curr_state = matching_entries[-1]
    
        return self.curr_state
		
		
    def get_customer_id(self):
		
        return self.customerid
		
        

    def estimateMarkovWaitingTimeVer2(self, pos_in_queue, queue_intensity, time_entered):
        """Calculate the amount after a certain time with exponential decay."""
                
        self.avg_delay = pos_in_queue * math.exp(-queue_intensity * time_entered)

        return self.avg_delay


    def get_times_entered(self):
                      
        return self.arr_prev_times


    # staticmethod
    def get_queue_sizes(self):
        q1_size = len(self.dict_queues_obj.get("1")) # Server1
        q2_size = len(self.dict_queues_obj.get("2")) # Server2

        return (q1_size, q2_size)
        

    def get_server_rates(self):
        srvrate1 = self.dict_servers_info.get("1")# Server1
        srvrate2 = self.dict_servers_info.get("2") # Server2

        return [srvrate1, srvrate2]


    def get_all_times(self):

        return self.all_times


    def get_curr_history(self):
        # We do the following to get rid of duplicate entries in the history
        seen = set()
        new_history = {} #[]
        for history in self.history:
            t = tuple(history.items())
            if t not in seen:
                seen.add(t)
                new_history.update({})
                #new_history.append(history)

        return new_history


    def run(self,duration, progress_bar=True,progress_log=False):
        steps=int(duration/self.time_res)
    
        if progress_bar!=None:
            loop=tqdm(range(steps),leave=False,desc='     Current run')
        else:
            loop=range(steps)                 
        
        self.arr_rate = self.objQueues.get_arrivals_rates()
        print("\n Arrival rate: ", self.arr_rate)
        
        for i in loop:            
			
            if progress_log:
                print("Step",i,"/",steps)
            # ToDo:: is line below the same as the update_parameters() in the a2c.py    
            self.markov_model.updateState()

            srv_1 = self.dict_queues_obj.get("1") # Server1
            srv_2 = self.dict_queues_obj.get("2") # Server2

            if len(srv_1) < len(srv_2):
                self.queue = srv_2
                self.srv_rate = self.dict_servers_info.get("2") # Server2

            else:            
                self.queue = srv_1
                self.srv_rate = self.dict_servers_info.get("1") # Server1
            
                  
            service_intervals=np.random.exponential(1/self.srv_rate,max(int(self.srv_rate*self.time_res*5),2)) # to ensure they exceed one sampling interval
            service_intervals=service_intervals[np.where(np.add.accumulate(service_intervals)<=self.time_res)[0]]
            service_intervals=service_intervals[0:np.min([len(service_intervals),self.queue.size])]
            arrival_intervals=np.random.exponential(1/self.arr_rate, max(int(self.arr_rate*self.time_res*5),2))

            arrival_intervals=arrival_intervals[np.where(np.add.accumulate(arrival_intervals)<=self.time_res)[0]]
            service_entries=np.array([[self.time+i,False] for i in service_intervals]) # False for service
            service_entries=service_entries.reshape(int(service_entries.size/2),2)
            time.sleep(2)
            arrival_entries=np.array([[self.time+i,True] for i in arrival_intervals]) # True for request
            # print("\n Arrived: ",arrival_entries) ####
            time.sleep(2)
            arrival_entries=arrival_entries.reshape(int(arrival_entries.size/2),2)
            # print(arrival_entries)
            time.sleep(2)
            all_entries=np.append(service_entries,arrival_entries,axis=0)
            all_entries=all_entries[np.argsort(all_entries[:,0])]
            self.all_times = all_entries
            # print("\n All Entered After: ",all_entries) ####
            serv_times = np.random.exponential(2, len(all_entries))
            serv_times = np.sort(serv_times)
            self.all_serv_times = serv_times
            # print("\n Times: ", np.random.exponential(2, len(all_entries)), "\n Arranged: ",serv_times)
            time.sleep(2)                      
            
            self.processEntries(all_entries, i) #, self.uses_nn)
            self.time+=self.time_res
            
            
            # Ensure dispatch data is updated at each step
            self.dispatch_all_queues() #dispatch_all_queues()
            #self.run_scheduler(duration)
            
            self.set_batch_id(i)
            
        return
    
    
    def set_batch_id(self, id):
		
        self.batchid = id
		
		
    def get_batch_id(self):
		
        return self.batchid
	
		
    def get_all_service_times(self):
        
        return self.all_serv_times  
        

    def processEntries(self,entries, batchid): #, uses_nn): # =np.int16 , actor_critic=actor_critic, =np.array([]), =np.int16 # else
        
        num_iterations = random.randint(1, 5)  # Random number of iterations between 1 and 5
        
        for entry in entries:           
            if entry[1]==True:
                # print("  Adding a new request into task queue...")                
                uses_nn = random.choice([True, False])
                req = self.addNewRequest(entry[0], batchid, uses_nn)
                self.arr_prev_times = np.append(self.arr_prev_times, entry[0])
                
            else:                
                q_selector = random.randint(1, 2)
                
                curr_queue = self.dict_queues_obj.get("1") if q_selector == "1" else self.dict_queues_obj.get("2")
                jockeying_rate = self.compute_jockeying_rate(curr_queue)
                reneging_rate = self.compute_jockeying_rate(curr_queue) # MATCH
                
                if q_selector == 1:					
                    self.queueID = "1" # Server1
                    
                    """
                         Run the serveOneRequest function a random number of times before continuing.
                    """
                        # Introduce a short delay to simulate processing time            
                    self.serveOneRequest(self.queueID, jockeying_rate, reneging_rate) # Server1 = self.dict_queues_obj["1"][0], entry[0],
                      
                                                                                
                    self.dispatch_queue_state(self.dict_queues_obj["1"], self.queueID) #, self.dict_queues_obj["2"]) #, req)
                    time.sleep(random.uniform(0.1, 0.5))  # Random delay between 0.1 and 0.5 seconds
                                               
                else:
                    self.queueID = "2"
                    self.serveOneRequest(self.queueID,  jockeying_rate, reneging_rate) # Server2 = self.dict_queues_obj["2"][0], entry[0],  
                    # self.initialize_queue_states(self.queueID,len(curr_queue), self.jockeying_rate, self.reneging_rate, req)                                     
                    self.dispatch_queue_state(self.dict_queues_obj["2"], self.queueID) #, self.dict_queues_obj["1"]) #, req)
                    time.sleep(random.uniform(0.1, 0.5))                                        
                    
                    
        return


    '''
        A mechanism to assess whether a reneging customer gets a reward 
        for the action or not. We take the computed localutility and compare 
        it to the general average, if this rep-emptive activity 
        took place and the localutility is less than the general moving average
        then a reward is given, else a penalty.get_queue_curr_state
    '''

    def getRenegeRewardPenalty (self, req, time_local_service, time_to_service_end):              
        
        if ((self.time+req.time_entrance) - time_to_service_end) > time_local_service:
            self.reward = 1
        else:
            self.reward = 0
     	 		
        return self.reward
        

    def generateLocalCompUtility(self, req):
        #req=Request(req)
        self.compute_counter = self.compute_counter + 1
        # local_delay=req.dist_local_delay.rvs(loc=req.loc_local_delay,scale=retime_to_service_endq.scale_local_delay)
        local_delay=req.dist_local_delay.rvs(loc=req.loc_local_delay,scale=2.0) #req.scale_local_delay)
        # print("\n Local :", local_delay, req.time_entrance, self.time)
        delay=float(self.time-req.time_entrance)+local_delay        
        self.local_utility = float(req.utility_basic*np.exp(-delay*req.discount_coef))

        self.avg_delay = (self.local_utility + self.avg_delay)/self.compute_counter

        return self.local_utility
    
    
    def generateExpectedJockeyCloudDelay (self, req, id_queue):
        #id_queue = np.array([req.id for req in self.queue]) get_queue_curr_state
        # req = self.queue[np.where(id_queue==req_id)[0][0]]

        total_jockey_delay = 0.0
        
        init_delay = float(self.time - req.time_entrance)
        
        if id_queue == "Server1":  
            curr_queue =self.dict_queues_obj["1"]  # Server1     
            alt_queue = self.dict_queues_obj["2"] # Server2
            pos_in_alt_queue = len(alt_queue)+1
            # And then compute the expected delay here using Little's Law
            expected_delay_in_alt_queue_pose = float(pos_in_alt_queue/self.arr_rate) #self.sampled_arr_rate)
            total_jockey_delay = expected_delay_in_alt_queue_pose + init_delay
        else:
            curr_queue =self.dict_queues_obj["2"]    # Server2    
            alt_queue = self.dict_queues_obj["1"]   # Server1
            pos_in_alt_queue = len(alt_queue)+1
            # And then compute the expected delay here using Little's Law
            expected_delay_in_alt_queue_pose = float(pos_in_alt_queue/self.arr_rate) # self.sampled_arr_rate)
            total_jockey_delay = expected_delay_in_alt_queue_pose + init_delay
            #self.queue= queue
            
        return total_jockey_delay
               

    def addNewRequest(self, expected_time_to_service_end, batchid, uses_nn): #, time_entered):
        # Join the shorter of either queues
               
        lengthQueOne = len(self.dict_queues_obj["1"]) # Server1
        lengthQueTwo = len(self.dict_queues_obj["2"]) # Server1 
        rate_srv1,rate_srv2 = self.get_server_rates()                

        if lengthQueOne < lengthQueTwo:
            time_entered = self.time   #self.estimateMarkovWaitingTime(lengthQueOne) ID
            pose = lengthQueOne+1
            server_id = "1" # Server1
            self.customerid = self.get_customer_id()
            self.customerid = "Batch"+str(self.get_batch_id())+"_"+self.customerid
            self.queue_intensity = self.arr_rate/rate_srv1
            #expected_time_to_service_end = self.estimateMarkovWaitingTime(float(pose)) # , queue_intensity, time_entered)
            #time_local_service = self.generateLocalCompUtility(req)
            

        else:
            pose = lengthQueTwo+1
            server_id = "2" # Server2
            self.customerid = self.get_customer_id()
            self.customerid = "Batch"+str(self.get_batch_id())+"_"+self.customerid
            time_entered = self.time #self.estimateMarkovWaitingTime(lengthQueTwo)
            self.queue_intensity = self.arr_rate/rate_srv2
            #expected_time_to_service_end = self.estimateMarkovWaitingTime(float(pose)) #, queue_intensity, time_entered)
            #time_local_service = self.generateLocalCompUtility(req)
            
            # self.long_avg_serv_time = self.get_long_run_avg_service_time(server_id)
                    
        
        if uses_nn:
            req=Request(uses_nn, time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid)
                    
            self.nn_subscribers.append(req)
            
        else:
            req=Request(uses_nn, time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid)
                    
            self.state_subscribers.append(req)  

        # print("\n LENGTHS => ", len(self.state_subscribers), " ==== ", len(self.nn_subscribers))   
  
        self.dict_queues_obj[server_id] = np.append(self.dict_queues_obj[server_id], req)
        
        self.queueID = server_id
        
        self.curr_req = req
        
        return #self.curr_req


    def getCustomerID(self):

        return self.customerid


    def setCurrQueueState(self, queueid):
		
        self.get_queue_curr_state()
		
        if queueid == "1": # Server1
            self.curr_state = {
                "ServerID": 1,
                "Intensity": self.arr_rate/get_server_rates()[0],
                "Pose":  self.get_queue_sizes([0])
                #"Wait": 
        }
        else:
            self.curr_state = {
                "ServerID":2,
                "Intensity": self.arr_rate/get_server_rates()[1],
                "Pose":  self.get_queue_sizes([1])
                #"Wait": 
        }
		
        return self.curr_state
        
    
    def get_queue_observation(self, queue_id, queue):
        """Returns queue observations formatted for Actor-Critic training."""
        
        state = self.get_queue_state(queue_id) # , queue)
        
        observation = {
			"server_id": state['queue_id'],
            "total_customers": state["total_customers"],
            "intensity": state["intensity"],
            "capacity": state["capacity"],
            "long_avg_serv_time": state["long_avg_serv_time"]
        }
        
        return observation

    
    def get_information_source(self):
    
        return self.uses_nn
        
    
    def dispatch_queue_state(self, curr_queue, curr_queue_id):
		
        rate_srv1,rate_srv2 = self.get_server_rates()
		
        if curr_queue_id == "1":
            alt_queue_id = "2"
            serv_rate = rate_srv1
        else:
            alt_queue_id = "1"
            serv_rate = rate_srv2

        curr_queue_state = self.get_queue_state(alt_queue_id) # , curr_queue)

        # Dispatch queue state to requests and allow them to act
        if not isinstance(None, type(curr_queue_state)):
            for req in curr_queue:
                if req.uses_nn:  # NN-based requests
                    
                    action = self.get_nn_optimized_decision(curr_queue_state) #self.actor_critic.select_action(curr_queue_state)
                    # print("\n ACT: ",action['action'], type(action['action']))
                    if action['action'] == 0: #action == 0:
                        print(f"ActorCriticInfo [RENEGE]: Server {curr_queue_id} in state:  {curr_queue_state}. Dispatching state to all {len(self.nn_subscribers)} requests  in server {alt_queue_id}")
                        self.makeRenegingDecision(req, curr_queue_id)                    
                    elif action['action'] ==  1: #action == 1:
                        print(f"ActorCriticInfo [JOCKEY]: Server {curr_queue_id} in state:  {curr_queue_state}. Dispatching state to all {len(self.nn_subscribers)} requests  in server {alt_queue_id}")
                        self.makeJockeyingDecision(req, curr_queue_id, alt_queue_id, req.customerid, serv_rate) # STATE
                else: 
                    print(f"Raw Markovian:  Server {curr_queue_id} in state {curr_queue_state}. Dispatching state to all {len(self.state_subscribers)} requests  in server {alt_queue_id}")
                    self.makeRenegingDecision(req, curr_queue_id)                                   
                    self.makeJockeyingDecision(req, curr_queue_id, alt_queue_id, req.customerid, serv_rate)
                    
             # of {len(curr_queue)}
             # of {len(curr_queue)}
        else:
            return

    
    def get_nn_optimized_decision(self, queue_state):		
        """Uses AI model to decide whether to renege or jockey."""
        
        # Convert queue_state values to a list of numeric values, filtering out non-numeric values
        numeric_values = [float(value) if isinstance(value, (int, float, np.number)) else 0.0 for value in queue_state.values()]
        
        state_tensor = torch.tensor(numeric_values, dtype=torch.float32).to(device)
        action = self.agent.select_action(state_tensor)  # self.actor_critic.select_action(state_tensor)# rate_reneged
        # print("\n Learned Action => ", "Jockey" if action == 1 else "Renege")
        return {"action": action, "nn_based": True}
    
    #def get_nn_optimized_decision(self, queue_state):
        
    #    if isinstance(queue_state, dict):         
    #        state = np.concatenate([
    #            np.array(queue_state[key]).flatten() if hasattr(queue_state[key], 'flatten') else np.array([queue_state[key]], dtype=float)
    #            for key in queue_state.keys() if isinstance(queue_state[key], (int, float, np.number))
    #        ])
            
        """Uses AI model to decide whether to renege or jockey."""
        #state_tensor = torch.tensor(state, dtype=torch.float32).to(device) # .values()
        
        # If state is not a tensor, create one; otherwise, use it as is.
    #    if not isinstance(state, torch.Tensor):
    #        state_tensor = torch.FloatTensor(state)
        
    #    if not isinstance(None, type(state_tensor)):
    #        action, _, _, _ = self.agent.select_action(state_tensor)  # self.actor_critic.select_action(state_tensor)                          
    #        return {"action": action.cpu().numpy(), "nn_based": True}
            
    #    else:			
    #        return     
    
    
    def dispatch(self, uses_nn): #  dispatch_data,   alt_queue_id
        """
        Dispatch state-action information or raw server status based on the use_nn flag.
        """
        if uses_nn:
            self.dispatch_nn_based_requests()
        else:
            self.dispatch_raw_server_status_requests()
            

    def dispatch_nn_based_requests(self): # dispatch_data,   alt_queue_id
        """
        Dispatch state-action information to NN-based subscribers.
        """
        
        for queue_id in ["1", "2"]:
            curr_queue = self.dict_queues_obj[queue_id]
            self.dispatch_queue_state( curr_queue, queue_id) # alt_queue,
        
        #for req in self.nn_subscribers:
        #    state_action_info = env.get_state_action_info()
            # Add logic to send state_action_info to the request
        #print(f"Dispatched the state-action info:  {state_action_info} about: {queueid} to all requests in {alt_queue_id}")
            

    def dispatch_raw_server_status_requests(self): # dispatch_data,
        """
        Dispatch raw server status information to state-based subscribers.
        """
        
        for queue_id in ["1", "2"]:
            curr_queue = self.dict_queues_obj[queue_id]
            self.dispatch_queue_state( curr_queue, queue_id)  #  alt_queue,
        
        #raw_server_status = self.get_queue_state(alt_queue_id) 
        
        #print("\n => ", raw_server_status) # MATCH
        
        #if not isinstance(None, type(raw_server_status)):
        #    for req in self.state_subscribers:

        #        if 'server_1' in raw_server_status.keys(): # dispatch_data                
                    # Add logic to send raw_server_status to the request
        #            print(f"Dispatching state: {raw_server_status} of server {alt_queue_id} to all requests in server {queueid}")
        #        else:
                    # Add logic to send raw_server_status to the request
        #            print(f"Dispatching state: {raw_server_status} of server {alt_queue_id} to all requests in server  {queueid}")
        #else:
        #    return 
       
    
    def dispatch_all_queues(self):
        """
        Dispatch the status of all queues and collect jockeying and reneging rates.
        """
        #for queue_id in ["1", "2"]:
        #    curr_queue = self.dict_queues_obj[queue_id]
        #    alt_queue_id = "2" if queue_id == "1" else "1"
        #    alt_queue = self.dict_queues_obj[alt_queue_id]
        #    # self.dispatch_queue_state(curr_queue, queue_id, alt_queue)
            #jockeying_rate = self.compute_jockeying_rate(curr_queue)
            #reneging_rate = self.compute_jockeying_rate(curr_queue)
        #    serv_rate = self.dict_servers_info[queue_id]
        #    num_requests = len(curr_queue)
            #long_avg_serv_time = self.get_long_run_avg_service_time(queue_id)
        #    queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate

        #    self.dispatch_data[f"server_{queue_id}"]["num_requests"].append(num_requests)
        #    self.dispatch_data[f"server_{queue_id}"]["jockeying_rate"].append(self.jockeying_rate)
        #    self.dispatch_data[f"server_{queue_id}"]["reneging_rate"].append(self.reneging_rate)
        #    self.dispatch_data[f"server_{queue_id}"]["queue_intensity"].append(queue_intensity)
        #    self.dispatch_data[f"server_{queue_id}"]["long_avg_serv_time"].append(self.long_avg_serv_time)
            # print(f"Server {queue_id} - Num requests: {num_requests}, Jockeying rate: {jockeying_rate}, Reneging rate: {reneging_rate}, Service rate: {serv_rate}, Long Avg Servtime: {self.long_avg_serv_time}")   
            
        self.use_nn=True
        self.dispatch(self.uses_nn) # self.dispatch_data, # Dispatch NN-based information   alt_queue_id
        
        self.use_nn=False
        self.dispatch(self.uses_nn) # self.dispatch_data, # Dispatch raw server status information   alt_queue_id             
            
        
    def setup_dispatch_intervals(self):
        """
        Set up the intervals for dispatching the queue status information.
        """
        
        #schedule.every(10).seconds.do(self.dispatch_all_queues)
        schedule.every(30).seconds.do(self.dispatch_all_queues)
        #schedule.every(60).seconds.do(self.dispatch_all_queues)       
            
            
    def plot_rates(self):
        """Plot the jockeying and reneging rates for NN-based vs. raw state subscribers."""
        
        num_requests_1 = self.dispatch_data["server_1"]["num_requests"]
        num_requests_2 = self.dispatch_data["server_2"]["num_requests"]

        plt.figure(figsize=(12, 5))

        # Plot Jockeying Rate Comparison
        plt.subplot(1, 2, 1)
        plt.plot(num_requests_1, self.dispatch_data["server_1"]["jockeying_rate"], label='Raw State - Server 1', linestyle='dashed')
        plt.plot(num_requests_1, self.dispatch_data["server_1"]["nn_jockeying_rate"], label='NN - Server 1')
        plt.plot(num_requests_2, self.dispatch_data["server_2"]["jockeying_rate"], label='Raw State - Server 2', linestyle='dashed')
        plt.plot(num_requests_2, self.dispatch_data["server_2"]["nn_jockeying_rate"], label='NN - Server 2')
        plt.xlabel("Number of Requests")
        plt.ylabel("Jockeying Rate")
        plt.legend()
        plt.title("Jockeying Rate Comparison")

        # Plot Reneging Rate Comparison
        plt.subplot(1, 2, 2)
        plt.plot(num_requests_1, self.dispatch_data["server_1"]["reneging_rate"], label='Raw State - Server 1', linestyle='dashed')
        plt.plot(num_requests_1, self.dispatch_data["server_1"]["nn_reneging_rate"], label='NN - Server 1')
        plt.plot(num_requests_2, self.dispatch_data["server_2"]["reneging_rate"], label='Raw State - Server 2', linestyle='dashed')
        plt.plot(num_requests_2, self.dispatch_data["server_2"]["nn_reneging_rate"], label='NN - Server 2')
        plt.xlabel("Number of Requests")
        plt.ylabel("Reneging Rate")
        plt.legend()
        plt.title("Reneging Rate Comparison")

        plt.show()  
        
        
    def adjust_service_rates(self):
        """
        Adjust the service rates of each queue to match the incoming request rate.
        """
        queue_sizes = {key: len(queue) for key, queue in self.dict_queues_obj.items()}
        total_requests = sum(queue_sizes.values())
        
        if total_requests == 0:
            print("No incoming requests, service rates remain unchanged.")
            return
        
        # Total arrival rate across all queues
        total_arrival_rate = self.sampled_arr_rate * len(self.dict_queues_obj)

        # Adjust service rates proportionally to the queue sizes
        for queue_id, queue_size in queue_sizes.items():
            proportion = queue_size / total_requests if total_requests > 0 else 0
            adjusted_service_rate = proportion * total_arrival_rate

            # Ensure a minimum service rate
            self.dict_servers[queue_id] = max(adjusted_service_rate, 1.0)

        print("Adjusted Service Rates:", self.dict_servers)          
            

    def run_scheduler(self): #, State):
        """
        Run the scheduler to dispatch queue status at different intervals.
        """
        self.setup_dispatch_intervals()
        while True: # duration: #
            schedule.run_pending()            
            time.sleep(1)
            
            #self.adjust_service_rates()
    
    
    def get_long_run_avg_service_time(self, queue_id):
		
        total_service_time = 0        
    
        if  "1" in queue_id:
            
            for req in self.dict_queues_obj["1"]:                
                total_service_time += req.service_time # exp_time_service_end                 
                             
            return total_service_time / self.total_served_requests_srv1
        else:
            
            for req in self.dict_queues_obj["2"]:                
                total_service_time += req.service_time  # exp_time_service_end                  
                              
            return total_service_time / self.total_served_requests_srv2
            
        if self.total_served_requests_srv1 == 0 or self.total_served_requests_srv2 == 0:
            return 0
    
    
    def initialize_queue_states(self, queueid, pose, jockeying_rate, reneging_rate, req, reward):
		
        self.history.append({
				    "ServerID": queueid, 
                    "at_pose": pose, 
                    "rate_jockeyed": jockeying_rate, 
                    "rate_reneged": reneging_rate, 
                    "expected_service_time": req.exp_time_service_end,               
                    "this_busy": self.queue_intensity, 
                    "long_avg_serv_time": self.long_avg_serv_time,
                    "time_service_took": req.time_exit-req.time_entrance,
                    "reward": reward, 
                    "action": "served",				 
			})	
    
                    
    def get_queue_state(self, queueid): # , queueobj): #, action):      
		
        observed = self.get_history()         
        
        if not isinstance(None, type(observed)):
            for hist in reversed(observed):               
                if queueid in str(hist['ServerID']):
                    # print("\n MATCH:", hist )
                    state = hist
                    return state
        else:			 
            self.history.append({
				    "ServerID": queueid, 
                    "at_pose": 0, 
                    "rate_jockeyed": 0.0, 
                    "rate_reneged": 0.0, 
                    "expected_service_time": 0.0,               
                    "this_busy": 0.0, 
                    "long_avg_serv_time": self.long_avg_serv_time,
                    "time_service_took": 0.0,
                    "reward": 0.0, 
                    "action":"served",				 
			})	
			            
            return self.history[0]  		  


    def serveOneRequest(self, queueID,  jockeying_rate, reneging_rate): # to_delete, serv_time, Dispatching
        #randomly select which queue to process at a time t+1
        q_selector = random.randint(1, 2)                            
        
        """Process a request and use the result for training the RL model."""
         
        if "1" in queueID:
            queue = self.dict_queues_obj["1"]            
            serv_rate = self.dict_servers_info["1"]
            #print("\n Length, service rates ", len(queue), " -- ", serv_rate)
        else:
            queue = self.dict_queues_obj["2"]
            serv_rate = self.dict_servers_info["2"]
            #print("\n Length, service rates ", len(queue), " -- ", serv_rate)

        if len(queue) == 0:
            return  # No request to process
        
        queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
        req = queue[0]  # Process the first request
        queue = queue[1:]  # Remove it from the queue
        req.time_exit = self.time

        # Compute reward based on actual waiting time vs. expected time
        time_in_queue = req.time_exit-req.time_entrance 
        reward = 1.0 if time_in_queue < req.service_time else -1.0
        
        # self.initialize_queue_states(queueID,len(queue), self.jockeying_rate, self.reneging_rate, req, reward)
 
        self.setNormalReward(reward)
        
        len_queue_1, len_queue_2 = self.get_queue_sizes()
        
        if "1" in queueID:
            self.total_served_requests_srv1+=1
            self.long_avg_serv_time = self.get_long_run_avg_service_time(queueID)
            #print("\n Setting observations in server: ", queueID, " ** Served: ", self.total_served_requests_srv1)
            self.objObserv.set_obs(queueID, len_queue_1, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time )
            self.history.append(self.objObserv.get_obs())  
        else:
            self.total_served_requests_srv2+=1
            self.long_avg_serv_time = self.get_long_run_avg_service_time(queueID)
            #print("\n Setting observations in server: ", queueID, " ** Served: ", self.total_served_requests_srv2)
            self.objObserv.set_obs(queueID, len_queue_2, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time) 
            self.history.append(self.objObserv.get_obs())                                           
        
        # Store the experience for RL training        
        state = self.get_queue_state(queueID) 
        # print("\n FORWARD: -> ", state)
        # Dispatch updated queue state
        self.dispatch_queue_state(queue, queueID)   
        
        if not isinstance(None, type(state)):
            action = self.agent.select_action(state)
            self.agent.store_reward(reward)

            # Train RL model after each request is processed Observed:
            self.agent.update()
                            
        return 	    
        
			
    def get_jockey_reward(self, req):
		
        reward = 0.0
        if not isinstance(req.customerid, type(None)):	
            if '_jockeyed' in req.customerid:
                if self.avg_delay+req.time_entrance < req.service_time: #exp_time_service_end:
                    reward = 1.0
                else:
                    reward = 0.0
                    
        return reward
        
    
    def get_history(self):

        return self.history   
    
    
    def get_curr_queue_id(self):
        
        return self.queueID
        
        
    def get_curr_queue(self):
		
        if "1" in  self.queueID:  # Server1
            self.queue = self.dict_queues.get("1")
               # find customer in queue by index
        # index = np.argwhere(self.queue==req_id)
        # req = self.queue[index]
        else:
            self.queue = self.dict_queues.get("2") # Server2
		
        return self.queue


    def getCurrentCustomerQueue(self, customer_id):

        for customer in self.dict_queues_obj["2"]: # Server2
            if customer_id in customer:
                curr_queue = self.dict_queues_obj["2"]

        for customer in self.dict_queues_obj["1"]: # Server1
            if customer_id in customer:
                curr_queue = self.dict_queues_obj["1"]

        return curr_queue
        
        
    def compare_wait_and_service_time(self, req):
        """
        Compare the time a request spent in the queue with the service time.

        :param req: The request object.
        :return: A dictionary with the tracked time, service time, and difference.
        """
        if req.time_exit is None:
            raise ValueError("Request has not exited the queue yet.")

        time_in_queue = req.time_exit - req.time_entrance
        service_time = req.service_time
        difference = time_in_queue - service_time

        return {
            "time_in_queue": time_in_queue,
            "service_time": service_time,
            "difference": difference
        }
        
        
    def get_remaining_time(self, queue_id, position):
        """
        Calculate the remaining time until a request at a given position is processed.
        
        :param queue_id: The ID of the queue (1 or 2).
        :param position: The position of the request in the queue (0-indexed).
        :return: Remaining time until the request is processed.
        """
        if queue_id == "1":
            serv_rate = self.dict_servers_info["1"]  # Server1
            queue = self.dict_queues_obj["1"]  # Queue1
        else:
            serv_rate = self.dict_servers_info["2"]  # Server2
            queue = self.dict_queues_obj["2"]  # Queue2

        queue_length = len(queue)
        
        if position < 0 or position >= queue_length:
            raise ValueError("Invalid position: Position must be within the bounds of the queue length.")
        
        # Calculate the remaining time based on the position and service rate
        remaining_time = sum(np.random.exponential(1 / serv_rate) for _ in range(position + 1))
        
        return remaining_time
        
        
    def calculate_max_cloud_delay(self, position, queue_intensity, req):
        """
        Calculate the max cloud delay based on the position in the queue and the current queue intensity.
        
        :param position: The position of the request in the queue (0-indexed).
        :param queue_intensity: The current queue intensity.
        :return: The max cloud delay.
        """
        base_delay = req.service_time #1.0  # Base delay for the first position
        position_factor = 0.01  # Incremental delay factor per position
        intensity_factor = 2.0  # Factor to adjust delay based on queue intensity

        # Calculate the position-dependent delay
        position_delay = base_delay + (position * position_factor)

        # Adjust for queue intensity
        max_cloud_delay = position_delay * (1 + (queue_intensity / intensity_factor))
        
        return max_cloud_delay
        
        
    def log_action(self, action, req, queueid):
        """Logs the request action to the file."""
        
        logging.info("", extra={"request_id": req.customerid, "queue_id": queueid, "action": action})
        
        
    def makeRenegingDecision(self, req, queueid):
        # print("   User making reneging decision...")
        decision=False  
        
        if queueid == "1":
            serv_rate = self.dict_servers_info["1"]
            queue =  self.dict_queues_obj["1"]         
        else:
            serv_rate = self.dict_servers_info["2"] 
            queue =  self.dict_queues_obj["2"]
        
        if self.learning_mode=='transparent':
            self.max_cloud_delay=stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=1/serv_rate) #(self.certainty,a=self.pos_in_queue,loc=0,scale=1/serv_rate) #self.serv_rate)

        else:			
            num_observations=min(self.objObserv.get_renege_obs(queueid, queue),len(self.history)) # if len(self.get_curr_obs_renege()) > 0 else 0 #self.history #self.observations.size
            mean_interval=np.mean(num_observations) # unbiased estimation of 1/lambda where lambda is the service rate
            if np.isnan(mean_interval):
                mean_interval=0
            if mean_interval!=0:
                self.serv_rate=1/mean_interval
            k_erlang=req.pos_in_queue*num_observations #self.pos_in_queue*num_observations
            scale_erlang=mean_interval*k_erlang
            # print("\n CHECKER -> ", np.arange(self.max_local_delay,self.APPROX_INF+self.time_res,step=self.time_res), " - - ", k_erlang, " ===== ",scale_erlang, " ===== ",num_observations)
            #mean_wait_time=mean_interval*self.pos_in_queue
            if np.isnan(mean_interval):
                self.max_cloud_delay=np.Inf
            else:
                self.max_local_delay = self.generateLocalCompUtility(req)
                queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
                self. max_cloud_delay = self.calculate_max_cloud_delay(req.pos_in_queue, queue_intensity, req)
                ## self.max_cloud_delay=stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=1/serv_rate)
                #self.max_cloud_delay=stats.erlang.ppf(self.certainty,loc=0,scale=mean_interval,a=req.pos_in_queue)                                            
            
            if "Server1" in queueid:
                self.queue = self.dict_queues_obj["1"]            
            else:
                self.queue = self.dict_queues_obj["2"] 
        
            if self.max_local_delay <= self.max_cloud_delay: # will choose to renege
                decision=True
                curr_pose = self.get_request_position(queueid, req.customerid)
        
                if curr_pose >= len(self.queue): #if curr_pose is None:
                    print(f"Request ID {req.customerid} not found in queue {queueid}. Continuing with processing...")
                    return 
                else:               
                    reward = self.reqRenege( req, queueid, curr_pose, serv_rate, self.queue_intensity, self.max_local_delay, req.customerid, req.service_time, decision, self.queue)
                                
                #temp=stats.erlang.cdf(np.arange(self.max_local_delay,step=self.time_res),k_erlang,scale=scale_erlang)
                '''
                    we still have a divide by zero error being thrown at this point - so blocking it out for now
                '''
                #error_loss=np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))-np.sum(np.append([temp[0]],np.diff(temp))*np.exp(-req.pos_in_queue/np.arange(self.max_local_delay,step=self.time_res)))
                
            else:   #will choose to wait and learn -> Can we use the actor-critic here??
                decision=False
                #print('choose to wait')
                # temp=stats.erlang.cdf(np.arange(self.max_local_delay,self.APPROX_INF+self.time_res,step=self.time_res),k_erlang,scale=scale_erlang)
                #error_loss=np.sum(np.diff(temp)*np.exp(-req.pos_in_queue/np.arange(self.max_local_delay+self.time_res,self.APPROX_INF+self.time_res,step=self.time_res)))-np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))
                
            ##dec_error_loss = self.error_loss - error_loss
            ##self.error_loss = error_loss
            
            ##if dec_error_loss > 1-np.exp(-mean_interval):
            ##    decision = False
            #else:
                #self.optimal_learning_achieved=True
                
            #if (not self.optimal_learning_achieved):
            ##    self.min_amount_observations=self.objObserv.get_renege_obs(queueid, queue) # self.observations.size+1
                
                
        self.curr_req = req
        
        return decision        


    def reqRenege(self, req, queueid, curr_pose, serv_rate, queue_intensity, time_local_service, customerid, time_to_service_end, decision, curr_queue):
        
        if "1" in queueid:
            self.queue = self.dict_queues_obj["1"]            
        else:
            self.queue = self.dict_queues_obj["2"] 
            
        if curr_pose >= len(curr_queue):
            return
            
        else:
            self.queue = np.delete(self.queue, curr_pose) # index)        
            self.queueID = queueid  
        
            req.customerid = req.customerid+"_reneged"
            
            self.log_action("Reneged", req, queueid)

        # In the case of reneging, you only get a reward if the time.entrance plus
        # the current time minus the time _to_service_end is greater than the time_local_service
        
            reward = self.getRenegeRewardPenalty(req, time_local_service, time_to_service_end)
            print(colored("%s", 'green') % (req.customerid) + " in Server %s" %(queueid) + " reneging now, to local processing " )
                                     
            self.reneging_rate = self.compute_reneging_rate(curr_queue)
            self.jockeying_rate = self.compute_jockeying_rate(curr_queue)
            self.long_avg_serv_time = self.get_long_run_avg_service_time(queueid)         
            
            self.objObserv.set_obs(queueid, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, time_local_service, time_to_service_end, reward, 0, self.long_avg_serv_time)
            
            self.history.append(self.objObserv.get_obs())  # Inside--    
        
            self.curr_req = req
        
            self.objQueues.update_queue_status(queueid)
            
            return reward


    def get_request_position(self, queue_id, request_id):
        """
        Get the position of a given request in the queue.
        
        :param queue_id: The ID of the queue (1 or 2).
        :param request_id: The ID of the request.
        :return: The position of the request in the queue (0-indexed).
        """
        if queue_id == "1":
            queue = self.dict_queues_obj["1"]  # Queue1
        else:
            queue = self.dict_queues_obj["2"]  # Queue2

        for position, req in enumerate(queue):            
            if req.customerid == request_id:
                return position

        return None
        
            
    def reqJockey(self, curr_queue_id, dest_queue_id, req, customerid, serv_rate, dest_queue, exp_delay, decision, curr_pose, curr_queue):		        
                
        if curr_pose >= len(curr_queue):
            return
            
        else:	
            np.delete(curr_queue, curr_pose) # np.where(id_queue==req_id)[0][0])
            reward = 1.0
            req.time_entrance = self.time # timer()
            dest_queue = np.append( dest_queue, req)
        
            self.queueID = curr_queue_id        
        
            req.customerid = req.customerid+"_jockeyed"
        
            if curr_queue_id == "1": # Server1
                queue_intensity = self.arr_rate/self.dict_servers_info["1"] # Server1
            
            else:
                queue_intensity = self.arr_rate/self.dict_servers_info["2"] # Server2
        
            reward = self.get_jockey_reward(req)
                  
            # print("\n Moving ", customerid," from Server ",curr_queue_id, " to Server ", dest_queue_id ) 
            print(colored("%s", 'green') % (req.customerid) + " in Server %s" %(curr_queue_id) + " jockeying now, to Server %s" % (colored(dest_queue_id,'green')))                      
            
            self.log_action(f"Jockeyed", req, dest_queue_id)
            
            self.reneging_rate = self.compute_reneging_rate(curr_queue)
            self.jockeying_rate = self.compute_jockeying_rate(curr_queue)     
            self.long_avg_serv_time = self.get_long_run_avg_service_time(curr_queue_id)                 
            
            self.objObserv.set_obs(curr_queue_id, curr_pose, self.queue_intensity, self.jockeying_rate, self.reneging_rate, exp_delay, req.exp_time_service_end, reward, 1, self.long_avg_serv_time) # time_alt_queue        
            
            self.history.append(self.objObserv.get_obs())
                                                  
            self.curr_req = req        
            self.objQueues.update_queue_status(curr_queue_id)# long_avg_serv_time
        
        return


    def makeJockeyingDecision(self, req, curr_queue_id, alt_queue_id, customerid, serv_rate):
        # We make this decision if we have already joined the queue
        # First we analyse our current state -> which server, server intensity and expected remaining latency
        # Then we get information about the state of the alternative queue
        # Evaluate input from the actor-critic once we get in the alternative queue
        
        if "Server1" in curr_queue_id:
            self.queue = self.dict_queues_obj["1"]  # Server1           
        else:
            self.queue = self.dict_queues_obj["2"]
                
        decision=False                
        #queue_intensity = self.arr_rate/self.dict_servers_info[alt_queue_id]
        curr_queue = self.dict_queues_obj.get(curr_queue_id)
        dest_queue = self.dict_queues_obj.get(alt_queue_id)

        self.avg_delay = self.generateExpectedJockeyCloudDelay ( req, curr_queue_id) 
        #self.objRequest.estimateMarkovWaitingTime(len(dest_queue)+1, features) #len(dest_queue)+1) #, queue_intensity, req.time_entrance)
        
        curr_pose = self.get_request_position(curr_queue_id, customerid)
        
        if curr_pose is None:
            print(f"Request ID {customerid} not found in queue {curr_queue_id}. Continuing with processing...")
            
        else:                                
            time_to_get_served = self.get_remaining_time(curr_queue_id, curr_pose)            
        
            '''
                I am at a position in server1 for example and the remaining
                time I will spend when I jockey to server2 is less than time
                left until I get served in the current queue, then jockey 
            '''
        
            if time_to_get_served > self.avg_delay:
                decision = True
                self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.queue)

        # ToDo:: There is also the case of the customer willing to take the risk
        #        and jockey regardless of the the predicted loss -> Customer does not
        #        care anymore whether they incur a loss because they have already joined anyway
        #        such that reneging returns more loss than the jockeying decision

        return decision
       

def main():       
	
    utility_basic = 1.0
    discount_coef = 0.1
    requestObj = RequestQueue(utility_basic, discount_coef)
    # env = ImpatientTenantEnv()
    duration = 20
    
    # Start the scheduler
    scheduler_thread = threading.Thread(target=requestObj.run(duration)) # requestObj.run_scheduler) # 
    scheduler_thread.start()
    
    # requestObj.run(duration)
    
    actor_critic = requestObj.getActCritNet() # Inside    
    agent = requestObj.getAgent()        
    
    #state_dim = len(env.observation_space)
    #action_dim = env.action_space            

    # agent = A2CAgent(state_dim, action_dim)    
    
    # Instantiate the ActorCritic class
    
    #actor_critic = ActorCritic(state_dim, action_dim).to(device)

    for episode in range(100):
        state, info = env.reset(seed=42)
        total_reward = 0

        for t in range(100):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.update()
        print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
