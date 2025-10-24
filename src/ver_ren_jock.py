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
from collections import defaultdict
# from ImpTenEnv import ImpatientTenantEnv
# from RenegeJockey import RequestQueue
# from a2c import ActorCritic, A2CAgent
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
import csv
import matplotlib.pyplot as plt
import pandas as pd
###############################################################################

#from RenegeJockey import RequestQueue, Queues, Observations

# Configure logging
#logging.basicConfig(
#    filename="request_decisions.log",
#    filemode="a",
#    format="%(asctime)s - Request ID: %(request_id)s - Queue ID: %(queue_id)s - Action: %(action)s",
#    level=logging.INFO
#)

################################## Globals ####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actions(Enum):
    RENEGE = 0
    JOCKEY = 1
    SERVED = -1
    

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


    def set_obs (self, queue_id,  curr_pose, intensity, jockeying_rate, reneging_rate, time_in_serv, time_to_service_end, reward, activity, long_avg_serv_time, uses_intensity_based): 
        		
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
                    "long_avg_serv_time": long_avg_serv_time,
                    "intensity_based_info": uses_intensity_based
                }
              

    def get_obs (self):
        
        return self.obs        
        
        
    def get_renege_obs(self): # , intensity, pose): # get_curr_obs_renege # , queueid, queue		                			      
	    
        return self.curr_obs_renege #  renegs
        
          
        
    def set_renege_obs(self, queueid, curr_pose, queue_intensity, jockeying_rate, reneging_rate ,time_local_service, req, reward,  activity, long_avg_serv_time, uses_intensity_based, diff_wait):		
        
        self.curr_obs_renege.append(
            {   
                "ServerID": queueid,
                "Request": req,
                "at_pose": curr_pose,
                "rate_jockeyed": jockeying_rate,
                "rate_reneged": reneging_rate,                
                "this_busy": queue_intensity,
                "expected_service_time":time_local_service,
                "time_service_took": req.exp_time_service_end,
                "reward": reward,
                "action":activity,
                "long_avg_serv_time": long_avg_serv_time,
                "intensity_based_info": uses_intensity_based,
                "waiting_time_diff": diff_wait
            }
        )
                
        
        
    def set_jockey_obs(self, queueid, curr_pose, queue_intensity, jockeying_rate, reneging_rate ,time_local_service, req, reward,  activity, long_avg_serv_time, uses_intensity_based, diff_wait):
        
        self.curr_obs_jockey.append(
            {
                "ServerID": queueid,
                "Request": req,
                "at_pose": curr_pose,
                "rate_jockeyed": jockeying_rate,
                "rate_reneged": reneging_rate,                
                "this_busy": queue_intensity,
                "expected_service_time":time_local_service,
                "time_service_took": req.exp_time_service_end,
                "reward": reward,
                "action":activity,
                "long_avg_serv_time": long_avg_serv_time,
                "intensity_based_info": uses_intensity_based,
                "waiting_time_diff": diff_wait			
            }
        )
        
    
    def get_jockey_obs(self):
		
        return self.curr_obs_jockey    


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
                
        self.dict_servers["1"] = _serv_rate_one 
        self.dict_servers["2"] = _serv_rate_two
        
        #print("\n Current Arrival Rate:", self.sampled_arr_rate, "Server1:", _serv_rate_one, "Server2:", _serv_rate_two) 


    def get_dict_servers(self):

        self.queue_setup_manager()
        
        return self.dict_servers        


    def get_curr_preferred_queues (self):        

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

    def __init__(self,uses_nn, uses_intensity_based, time_entrance,pos_in_queue=0,utility_basic=0.0,service_time=0.0,discount_coef=0.0, outage_risk=0.1, # =timer()
                 customerid="", learning_mode='online',min_amount_observations=1,time_res=1.0,markov_model=msm.StateMachine(orig=None), time_exit=0.0,
                 exp_time_service_end=0.0, serv_rate=1.0, dist_local_delay=stats.expon,para_local_delay=[1.0,2.0,10.0], batchid=0,  server_id=None):  #markov_model=a2c.A2C, 
        
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
        self.time_exit = time_exit  # To be set when the request leaves the queue
        self.service_time = service_time
        self.server_id = server_id
        self.uses_nn = uses_nn
        self.reneged = False
        self.jockeyed = False
        self.actor_critic = None # ""
        self.uses_intensity_based = uses_intensity_based
        

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
        self.scale_local_delay=float(para_local_delay[2]) #0
        self.max_local_delay=self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay)
        self.max_cloud_delay=float(queueObj.get_arrivals_rates()/self.serv_rate) # np.inf
       
        # print("\n ****** ",self.loc_local_delay, " ---- " , self.time_entrance-arr_prev_times[len(arr_prev_times)-1])
        self.observations=np.array([])
        self.error_loss=1
        self.optimal_learning_achieved=False
        
        #  utility accounting:
        self.served = False      # will be set True when request is served in serveOneRequest
        self.reneged = False     # set in reqRenege
        self.jockeyed = False    # set in reqJockey

        return
    
    
    def compute_observed_utility(self, c_wait=0.1, c_local=0.5, c_switch=0.2, v_local=0.0):
        """
        Compute observed utility for this request given realized outcomes.
        Parameters:
            c_wait: cost per unit waiting time
            c_local: cost/penalty incurred when reneging (local processing cost)
            c_switch: cost/penalty for jockeying
            v_local: value recovered from local processing if reneged
        Returns:
            utility (float)
        Uses fields:
            - self.utility_basic: value if served (v_i)
            - self.time_entrance, self.time_exit
            - self.served, self.reneged, self.jockeyed
        """
        # Ensure we can compute waiting time
        if (self.time_exit is None) or (self.time_entrance is None):
            waiting_time = 0.0
        else:
            waiting_time = float(self.time_exit - self.time_entrance)

        # Value if served
        served_value = float(self.utility_basic) if getattr(self, "served", False) else 0.0

        # Costs
        waiting_cost = c_wait * waiting_time
        local_cost = c_local if getattr(self, "reneged", False) else 0.0
        switch_cost = c_switch if getattr(self, "jockeyed", False) else 0.0
        local_value = v_local if getattr(self, "reneged", False) else 0.0

        utility = served_value - waiting_cost - local_cost - switch_cost + local_value
        return utility
    
    
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
        
        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.rewards = []
        

    def select_action(self, state):
        # print("\n -->> ", state)
        if isinstance(state, dict):
            #state = np.concatenate([state[key].flatten() for key in state.keys()])
            # state = np.concatenate([np.array(state[key]).flatten() if hasattr(state[key], 'flatten') else [state[key]] for key in state.keys()]) # Step
            state = np.concatenate([
                np.array(state[key]).flatten() if hasattr(state[key], 'flatten') else np.array([state[key]], dtype=float)
                for key in state.keys() if isinstance(state[key], (int, float, np.number))
            ])
            
        # If state is not a tensor, create one; otherwise, use it as is.
        #if not isinstance(state, torch.Tensor):
        #    state = torch.FloatTensor(state)
    
        # Ensure the state tensor does not contain NaN values
        #if torch.isnan(state).any():
        #    print("NaN values detected in state tensor:", state)
        #    state = torch.where(torch.isnan(state), torch.zeros_like(state), state)
                       
        #state = state.unsqueeze(0).to(device)
        #action_probs, state_value = self.model(state)
        
        # Ensure state is a numpy array
        # Ensure state is a numpy array
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32)

        # Pad or trim state to length 11
        expected_dim = 11
        
        if state.shape[0] < expected_dim:
            state = np.pad(state, (0, expected_dim - state.shape[0]), 'constant')
        elif state.shape[0] > expected_dim:
            state = state[:expected_dim]

        # Convert to torch tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Forward pass
        action_probs, state_value = self.model(state)
        # print("Action probabilities:", action_probs.detach().cpu().numpy())
        
        # Check for NaN values in action_probs and handle them
        if torch.isnan(action_probs).any():
            print("NaN values detected in action_probs:", action_probs)
            action_probs = torch.where(torch.isnan(action_probs), torch.zeros_like(action_probs), action_probs)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # Normalize to ensure valid probabilities            
            
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        self.log_probs.append(action_dist.log_prob(action))
        self.values.append(state_value)
        # print(f"log_probs: {self.log_probs}, values: {self.values}") #, rewards: {self.rewards}")
        #print("\n LOG: ", self.log_probs, "\n ACTION DIST: ", action_dist)
        return action.item()
    
    
    def select_action_with_entropy(self, state):
        """
        Select an action and calculate the entropy of the policy distribution.

        Args:
            state (np.ndarray or torch.Tensor): The current state or observation.

        Returns:
            tuple: (action, entropy) where:
                - action (int): The selected action.
                - entropy (float): The entropy of the policy distribution.
        """
        # Convert state to a tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Get action probabilities and state value from the model
        action_probs, state_value = self.model(state)

        # Create a categorical distribution based on action probabilities
        action_dist = torch.distributions.Categorical(action_probs)

        # Sample an action
        action = action_dist.sample()

        # Calculate entropy of the policy distribution
        entropy = action_dist.entropy().item()

        # Store log probability and state value for training
        self.log_probs.append(action_dist.log_prob(action))
        self.values.append(state_value)

        return action.item(), entropy
      

    def store_reward(self, reward):
        self.rewards.append(reward)
        # print(f"Stored reward: {reward}, Total rewards: {self.rewards}")
        

    def update_old(self):
        
        if not self.log_probs:
            log_probs = torch.zeros(1, device=device)  # Fallback to a zero tensor
            print("Warning: log_probs is empty. Initializing with zeros.")
        else:
            log_probs = torch.stack(self.log_probs).to(device)

        if not self.values:
            values = torch.zeros(1, device=device)  # Fallback to a zero tensor
            print("Warning: values is empty. Initializing with zeros.")
        else:
            values = torch.cat(self.values).to(device)

        if not self.rewards:
            print("Warning: rewards is empty. Skipping update.")
            return 0.0, 0.0, 0.0
            
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # print(f"Returns: {returns}")
        returns = torch.tensor(returns).to(device)            
            
        # values = torch.cat(self.values).to(device)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss
        
        # print("\n total loss of actor + critic:", loss) #loss.requires_grad)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear stored values
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        # Return scalar losses for logging
        return actor_loss.item(), critic_loss.item(), loss.item()
        
    def update(self):
        """
        Performs the A2C update using the accumulated log_probs, values, and rewards.
        Returns:
            actor_loss (float), critic_loss (float), total_loss (float)
        """
        if not self.log_probs or not self.values or not self.rewards:
            return 0.0, 0.0, 0.0

        # Convert lists to tensors
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze(-1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=values.device)

        # Compute returns (discounted rewards)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32, device=values.device)

        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate advantages
        advantages = returns - values

        # Actor loss (policy gradient)
        actor_loss = - (log_probs * advantages.detach()).mean()

        # Critic loss (value regression)
        critic_loss = advantages.pow(2).mean()

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss

        # Backprop and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear buffers for next episode
        self.log_probs = []
        self.values = []
        self.rewards = []

        # Return scalars (detached)
        return actor_loss.item(), critic_loss.item(), total_loss.item()


class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, state_dim, action_dim, utility_basic, discount_coef, actor_critic, agent, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon,  time_exit=0.0, exp_time_service_end=0.0,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, batchid=np.int16, uses_nn=False, uses_intensity_based = False): # Dispatched
                 
        self.request_position_history = defaultdict(list)  # NEW: key = request_id, value = list of (time, queue_id, position)
        
        # new runtime cost config (can be a single number or a (min,max) tuple for ranges)
        cost_wait_range=(0.05,0.2), #None,    # e.g. 0.1 or (0.05,0.2)
        cost_local_range=(0.3,0.8), #None,   # e.g. 0.5 or (0.3,0.8)
        cost_switch_range=(0.1,0.4), #None,  # e.g. 0.2 or (0.1,0.4)
        value_local_range=(0.0,0.5), #None,  # e.g. 0.0 or (0.0,0.5)
        randomize_costs=False,   # if True pick costs at construction
        randomize_per_episode=False,  # if True re-pick costs each episode during run()
        cost_seed=None           # RNG seed for reproducible draws
        
        self.dispatch_data = {
            "server_1": {
                "num_requests": [],
                "jockeying_rate_raw": [],  # Raw state jockeying rate
                "jockeying_rate_nn": [],   # NN-based jockeying rate
                "reneging_rate_raw": [],   # Raw state reneging rate
                "reneging_rate_nn": [],    # NN-based reneging rate
                "long_avg_serv_time": [],
                "queue_intensity": []
            },
            "server_2": {
                "num_requests": [],
                "jockeying_rate_raw": [],  # Raw state jockeying rate
                "jockeying_rate_nn": [],   # NN-based jockeying rate
                "reneging_rate_raw": [],   # Raw state reneging rate
                "reneging_rate_nn": [],    # NN-based reneging rate
                "long_avg_serv_time": [],
                "queue_intensity": []
            }
        }
        
        self.markov_model=msm.StateMachine(orig=markov_model) # dispatch_all_queues
        self.customerid = customerid
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_critic = actor_critic
        self.agent = agent
        self.env = ""
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
        self.time_exit = time_exit  # To be set when the request leaves the queue    
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
        self.srv1_history = []
        self.srv2_history = []
        self.history = [] 
        self.curr_obs_jockey = [] 
        self.curr_obs_renege = [] 
        self.uses_nn = uses_nn 
        self.long_avg_serv_time = 0.0 # intensity

        self.arr_prev_times = np.array([])
        self.queue_intensity = 0.0
        self.uses_intensity_based = uses_intensity_based

        self.objQueues = Queues()
        self.objRequest = Request(self.uses_nn, self.uses_intensity_based, time)
        self.objObserv = Observations()
        self.queueID = "1"  # Initialize with default queue ID "1" or any logic to choose a starting queue

        self.dict_queues_obj = self.objQueues.get_dict_queues()
        self.dict_servers_info = self.objQueues.get_dict_servers()
        
        if self.dict_queues_obj:
            self.queueID = str(max(self.dict_queues_obj, key=lambda q: len(self.dict_queues_obj[q])))
         
        # print("\n QUEUE ID: ", self.queueID)  
        self.renege_reward = 0.0
        self.jockey_reward = 0.0
        self.curr_state = {} # ["Busy","Empty"]

        self.arr_rate = 0.0 # self.objQueues.get_arrivals_rates()
        # self.arr_rate = self.objQueues.get_arrivals_rates()
        
        self.all_times = []
        self.all_serv_times = []        
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
        self.departure_dispatch_count = 0
        self.intensity_dispatch_count = 0 
        self.all_requests = []  # Stores all requests throughout the simulation            
        
        # existing initialization code (unchanged) ...
        # Add cost / local-value defaults (these can be tuned externally if you want)
        # Interpret single float as fixed range (min==max)
        # Helper to normalise a provided range-like or scalar into (min, max)
        def _norm_range(x, default):
            if x is None:
                return (default, default)
            # If single numeric value
            if isinstance(x, (int, float, np.floating, np.integer)):
                return (float(x), float(x))
            # If a sequence of two numbers
            try:
                seq = list(x)
                if len(seq) == 2:
                    return (float(seq[0]), float(seq[1]))
            except Exception:
                pass
            # Fallback
            return (default, default)

        # defaults (previous hard-coded params)
        default_cost_wait = 0.1
        default_cost_local = 0.5
        default_cost_switch = 0.2
        default_value_local = 0.0

        # normalize provided ranges
        self.cost_wait_range = _norm_range(cost_wait_range, default_cost_wait)
        self.cost_local_range = _norm_range(cost_local_range, default_cost_local)
        self.cost_switch_range = _norm_range(cost_switch_range, default_cost_switch)
        self.value_local_range = _norm_range(value_local_range, default_value_local)

        # randomization flags and seed
        self.randomize_costs = bool(randomize_costs)
        self.randomize_per_episode = bool(randomize_per_episode)
        self._cost_rng = np.random.default_rng(cost_seed) if cost_seed is not None else np.random.default_rng()

        # active cost params (initialized either to defaults or randomized below)
        self.cost_wait = default_cost_wait
        self.cost_local = default_cost_local
        self.cost_switch = default_cost_switch
        self.value_local = default_value_local

        # If randomize_costs True but not per-episode, do a single draw at construction
        if self.randomize_costs and not self.randomize_per_episode:
            # only randomize now if any range is non-degenerate
            if any(a != b for (a, b) in [self.cost_wait_range, self.cost_local_range, self.cost_switch_range, self.value_local_range]):
                self._randomize_costs(log_prefix="[init]")

        # Per-episode accounting (reset at start of each episode in run())
        self.current_episode_utilities = []   # observed utilities for requests in current episode
        self.episode_requests_count = 0
        self.episode_reneges = 0
        self.episode_jockeys = 0
        
        BROADCAST_INTERVAL = 5
        
        return               
		
	
    def _range_is_random(self, rng_tuple):
        """
        Return True if the tuple represents a non-degenerate range (min != max).
        Accepts None or (min, max).
        """
        try:
            a, b = float(rng_tuple[0]), float(rng_tuple[1])
            return a != b
        except Exception:
            return False


    # --- Replace or add this helper inside RequestQueue ---
    def _randomize_costs(self, log_prefix="[costs]"):
        """
        Randomize active cost parameters from the configured ranges using self._cost_rng.
        Uniform draw between min and max when min != max, otherwise use fixed value.
        Prints the ranges and chosen values for debugging.
        Returns chosen dict.
        """
        choices = {}
        a, b = self.cost_wait_range
        choices["cost_wait"] = float(self._cost_rng.uniform(a, b)) if a != b else float(a)

        a, b = self.cost_local_range
        choices["cost_local"] = float(self._cost_rng.uniform(a, b)) if a != b else float(a)

        a, b = self.cost_switch_range
        choices["cost_switch"] = float(self._cost_rng.uniform(a, b)) if a != b else float(a)

        a, b = self.value_local_range
        choices["value_local"] = float(self._cost_rng.uniform(a, b)) if a != b else float(a)

        # set active costs
        self.cost_wait = choices["cost_wait"]
        self.cost_local = choices["cost_local"]
        self.cost_switch = choices["cost_switch"]
        self.value_local = choices["value_local"]

        # debug print showing both ranges and chosen values
        ranges = {
            "cost_wait_range": self.cost_wait_range,
            "cost_local_range": self.cost_local_range,
            "cost_switch_range": self.cost_switch_range,
            "value_local_range": self.value_local_range
        }
        print(colored(f"{log_prefix} ranges: {ranges}", "cyan"))
        print(colored(f"{log_prefix} chosen: {choices}", "cyan"))

        return choices
    
	# NEW: Call this on every queue change
    def track_all_request_positions(self, queue_id):
        """
        Track the current positions of all requests in the specified queue.
        """
        queue = self.dict_queues_obj[queue_id]
        for position, req in enumerate(queue):
            self.request_position_history[req.customerid].append({
                "time": self.time,
                "queue_id": queue_id,
                "position": position
            })
        # Optionally track requests no longer present (e.g., mark as -1 or 'removed')
        # You can implement more logic if needed
	
	
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
		
    
    def compute_jockeying_rate(self, subscribers):
        """
        Compute the jockeying rate for a given list of subscribers.
        """
        if not subscribers:
            return 0.0
        jockeys = sum(1 for req in subscribers if getattr(req, 'jockeyed', False))
        
        return jockeys / len(subscribers)            
       
    
    def compute_reneging_rate(self, subscribers):
        """
        Compute the reneging rate for a given list of subscribers.
        """
        if not subscribers:
            return 0.0
        renegs = sum(1 for req in subscribers if getattr(req, 'reneged', False))
        
        return renegs / len(subscribers)
        
        
    def compute_reneging_rate_per_server(self, subscribers, server_id):
        """Compute the reneging rate for requests added to a specific server and using the correct subscriber list."""
        
        # Filter requests by server_id from the subscriber list
        server_requests = [req for req in subscribers if hasattr(req, 'server_id') and req.server_id == server_id]
        
        if len(server_requests) == 0:
            return 0
        
        renegs = sum(1 for req in server_requests if '_reneged' in req.customerid)
        
        return renegs / len(server_requests)


    def compute_jockeying_rate_per_server(self, subscribers, server_id):
        """Compute the jockeying rate for requests added to a specific server and using the correct subscriber list."""
        # Filter requests by server_id from the subscriber list
        server_requests = [req for req in subscribers if hasattr(req, 'server_id') and req.server_id == server_id]
        
        if len(server_requests) == 0:
            return 0
            
        jockeys = sum(1 for req in server_requests if '_jockeyed' in req.customerid)
        
        return jockeys / len(server_requests)
        
    
    def get_rates_summary_per_server_and_source(self):
        """Get a comprehensive summary of reneging and jockeying rates per server and information source."""
        summary = {
            "server_1": {
                "state_subscribers": {
                    "reneging_rate": self.compute_reneging_rate_per_server(self.state_subscribers, "1"),
                    "jockeying_rate": self.compute_jockeying_rate_per_server(self.state_subscribers, "1"),
                    "count": len([req for req in self.state_subscribers if hasattr(req, 'server_id') and req.server_id == "1"])
                },
                "nn_subscribers": {
                    "reneging_rate": self.compute_reneging_rate_per_server(self.nn_subscribers, "1"),
                    "jockeying_rate": self.compute_jockeying_rate_per_server(self.nn_subscribers, "1"),
                    "count": len([req for req in self.nn_subscribers if hasattr(req, 'server_id') and req.server_id == "1"])
                }
            },
            "server_2": {
                "state_subscribers": {
                    "reneging_rate": self.compute_reneging_rate_per_server(self.state_subscribers, "2"),
                    "jockeying_rate": self.compute_jockeying_rate_per_server(self.state_subscribers, "2"),
                    "count": len([req for req in self.state_subscribers if hasattr(req, 'server_id') and req.server_id == "2"])
                },
                "nn_subscribers": {
                    "reneging_rate": self.compute_reneging_rate_per_server(self.nn_subscribers, "2"),
                    "jockeying_rate": self.compute_jockeying_rate_per_server(self.nn_subscribers, "2"),
                    "count": len([req for req in self.nn_subscribers if hasattr(req, 'server_id') and req.server_id == "2"])
                }
            }
        }


        return summary
        
    ''' 
    def compute_reneging_rate(self, queue):
        """Compute the reneging rate for a given queue."""
        renegs = sum(1 for req in queue if '_reneged' in req.customerid)
        return renegs / len(queue) if len(queue) > 0 else 0
        

    def compute_jockeying_rate(self, queue):
        """Compute the jockeying rate for a given queue."""
        jockeys = sum(1 for req in queue if '_jockeyed' in req.customerid)
        return jockeys / len(queue) if len(queue) > 0 else 0
    '''
    
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
        print("\n ************ ", srvrate1, srvrate2, " ========== ",self.srvrates_1, self.srvrates_2)
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
        
        
    def initialize_queue_state(self, queueid, arr_rate):
		
        srv_rate = self.dict_servers_info.get(queueid)
		
        if "1" in queueid: 
            size_srv = len(self.dict_queues_obj.get("1"))			
            intensity = arr_rate/srv_rate
        else: 
            size_srv = len(self.dict_queues_obj.get("2"))           
            intensity = arr_rate/srv_rate
			
        return {
			"ServerID": queueid, 
            "at_pose": size_srv, 
            "rate_jockeyed": 0.0, 
            "rate_reneged": 0.0, 
            "expected_service_time": 0.0,               
            "this_busy": intensity, 
            "long_avg_serv_time": self.long_avg_serv_time,
            "time_service_took": 0.0,
            "reward": 0.0, 
            "action":"queued",	
            "intensity_based_info": self.uses_intensity_based,
        }


    def run(self, duration, env, adjust_service_rate, num_episodes=10, progress_bar=True, progress_log=True, save_to_file="simu_results.csv"):
        """
        Run the simulation with episodic training.

        Args:
            duration (int): The total duration of the simulation (e.g., seconds or steps).
            num_episodes (int): Number of episodes for training.
            progress_bar (bool): Whether to show a progress bar for steps.
            progress_log (bool): Whether to log progress for each step.

        Returns:
            None
        """
        
        self.env = env
        steps_per_episode = int(duration / self.time_res)
        metrics = []  # List to store metrics for all episodes
        save_to_file = "simu_results.csv"
        
        actor_losses = []
        critic_losses = []
        total_losses = []

        if progress_bar:
            step_loop = tqdm(range(steps_per_episode), leave=False, desc='     Current run')
        else:
            step_loop = range(steps_per_episode)                             
        
        # If user requested randomize_costs at construction but did not request per-episode randomization,
        # and ranges are non-degenerate, we still consider per-episode randomization to match user expectation.
        # (But we preserve the original option that allows a single random draw at construction time.)
        # Determine whether to enforce per-episode draws when ranges are non-degenerate
        enforce_per_episode_if_ranges = False
        if self.randomize_costs and not self.randomize_per_episode:
            if any(a != b for (a, b) in [self.cost_wait_range, self.cost_local_range, self.cost_switch_range, self.value_local_range]):
                enforce_per_episode_if_ranges = True          
        
        for episode in range(num_episodes):
            print(f"Starting Episode {episode + 1}/{num_episodes}")            
            
            # if randomization per episode requested, sample new costs here (reproducible via cost_seed)
            if self.randomize_per_episode or enforce_per_episode_if_ranges:
                chosen = self._randomize_costs(log_prefix=f"[episode {episode+1}]")
            elif self.randomize_costs and not self.randomize_per_episode:
                # previous behavior: randomize once at construction - keep backward compatibility
                # If costs were not randomized at construction time, do a one-off draw now and log it.
                if (self.cost_wait == 0.0 and self.cost_local == 0.0 and self.cost_switch == 0.0 and self.value_local == 0.0) or \
                   (self._range_is_random(self.cost_wait_range) or self._range_is_random(self.cost_local_range) or \
                    self._range_is_random(self.cost_switch_range) or self._range_is_random(self.value_local_range)):
                    chosen = self._randomize_costs()
                
            # Reset environment for the new episode
            state, info = self.env.reset(seed=42)
            total_reward = 0
            done = False  # Flag to track episode termination
            episode_start_time = time.time()
            i = 0
            episode_policy_entropy = 0  # Track total policy entropy for the episode
            
            # per-episode trackers already initialized in __init__, ensure cleared
            self.current_episode_utilities = []
            self.episode_requests_count = 0
            self.episode_reneges = 0
            self.episode_jockeys = 0
            
            #losses = {"actor_loss": 0, "critic_loss": 0, "total_loss": 0}
            srv1_jockeying_rates = []
            srv2_reneging_rates = []
            
            step_rewards = 0
            
            # Reset per-episode trackers
            self.current_episode_utilities = []
            self.episode_requests_count = 0
            self.episode_reneges = 0
            self.episode_jockeys = 0
        
            for i in step_loop: 
                self.arr_rate = self.objQueues.randomize_arrival_rate()  # Randomize arrival rate
                srv_1 = self.dict_queues_obj.get("1") # Server1
                srv_2 = self.dict_queues_obj.get("2") 
                
                deltaLambda=random.randint(1, 2)
                                
                #next_state, reward, done, info = self.env.step(action)
                
                if len(srv_1) < len(srv_2): #deltaLambda == 1:
                    serv_rate_one = self.arr_rate - deltaLambda 
                    serv_rate_two = self.arr_rate + deltaLambda
                else:
                    serv_rate_one = self.arr_rate + deltaLambda 
                    serv_rate_two = self.arr_rate - deltaLambda
        
                # serv_rate_one = self.arr_rate + deltaLambda 
                # serv_rate_two = self.arr_rate - deltaLambda

                self.srvrates_1 = serv_rate_one / 2                
                self.srvrates_2 = serv_rate_two / 2                                         
                
                print("\n Arrival rate: ", self.arr_rate, "Rates 1: ----", self.srvrates_1,  "Rates 2: ----", self.srvrates_2)  
                #if done:  # Break the loop if the episode ends
                #    break         
			
                if progress_log:
                    print("Step", i + 1, "/", steps_per_episode) # print("Step",i,"/",steps)
                        
                self.markov_model.updateState()
                
                if len(srv_1) < len(srv_2):
                    self.queue = srv_2
                    self.srv_rate = self.srvrates_1                   

                else:            
                    self.queue = srv_1
                    self.srv_rate = self.srvrates_2                                
                  
                service_intervals=np.random.exponential(1/self.srv_rate,max(int(self.srv_rate*self.time_res*5),2)) # to ensure they exceed one sampling interval
                service_intervals=service_intervals[np.where(np.add.accumulate(service_intervals)<=self.time_res)[0]]
                service_intervals=service_intervals[0:np.min([len(service_intervals),len(self.queue)])]
                arrival_intervals=np.random.exponential(1/self.arr_rate, max(int(self.arr_rate*self.time_res*5),2))

                arrival_intervals=arrival_intervals[np.where(np.add.accumulate(arrival_intervals)<=self.time_res)[0]]
                service_entries=np.array([[self.time+i,False] for i in service_intervals]) # False for service
                service_entries=service_entries.reshape(int(service_entries.size/2),2)
                time.sleep(1)
                arrival_entries=np.array([[self.time+i,True] for i in arrival_intervals]) # True for request
                # print("\n Arrived: ",arrival_entries) ####
                # time.sleep(1)
                arrival_entries=arrival_entries.reshape(int(arrival_entries.size/2),2)
                # print(arrival_entries)
                time.sleep(1)
                all_entries=np.append(service_entries,arrival_entries,axis=0)
                all_entries=all_entries[np.argsort(all_entries[:,0])]
                self.all_times = all_entries
                # print("\n All Entered After: ",all_entries) ####
                serv_times = np.random.exponential(2, len(all_entries))
                serv_times = np.sort(serv_times)
                self.all_serv_times = serv_times
                # print("\n Times: ", np.random.exponential(2, len(all_entries)), "\n Arranged: ",serv_times)
                time.sleep(1)                      
                
                # Step 1: Get action from the agent based on the current state
                action = self.agent.select_action(state)
                
                # Step 1: Get action from the agent based on the current state
                #action, policy_entropy = self.agent.select_action_with_entropy(state)
                #episode_policy_entropy += policy_entropy
                
                # Step 2: Apply the action to the environment and get feedback
                next_state, reward, done, info = self.env.step(action)                
                
                # Update queue state in the environment
                self.env.update_queue_state(next_state)                               

                # Step 3: Store the reward for training
                self.agent.store_reward(reward)
                step_rewards += reward
                #total_reward += reward

                # Step 4: Update the state for the next step
                state = next_state
                i += 1            
                              
                self.processEntries(all_entries, i) #, self.uses_nn)
                self.time+=self.time_res
            
                # Step 4 (Optional): Adjust service rates if enabled
                if adjust_service_rate:
                    self.adjust_service_rates()
                 
                
                # Ensure dispatch data is updated at each step
                self.dispatch_all_queues() #dispatch_all_queues()
                #self.run_scheduler(duration)
                
                # Optional: Log step-level progress (can be verbose)
                if progress_log:
                    print(f"Step {i + 1}: Action={action}, Reward={reward}, Total Reward={total_reward}")
            
                self.set_batch_id(i)
                
                total_reward += step_rewards
                
                print(f"Step {i + 1}: Action={action}, Reward={reward}, Step Total Reward={step_rewards}, Total Reward={total_reward}")
                
            # Update the RL agent at the end of each episode
            actor_loss, critic_loss, total_loss = self.agent.update()
                        
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            total_losses.append(total_loss)
            
            #print("\n ****** ", actor_losses, " **** ", critic_losses, " **** ", total_losses)
            # Calculate episode duration
            episode_duration = time.time() - episode_start_time
            
            # For Markov/raw subscribers
            srv1_reneging_rate_markov = self.compute_reneging_rate_per_server(self.state_subscribers, "1")
            srv2_reneging_rate_markov = self.compute_reneging_rate_per_server(self.state_subscribers, "2")
            srv1_jockeying_rate_markov = self.compute_jockeying_rate_per_server(self.state_subscribers, "1")
            srv2_jockeying_rate_markov = self.compute_jockeying_rate_per_server(self.state_subscribers, "2")

            # For NN subscribers
            srv1_reneging_rate_nn = self.compute_reneging_rate_per_server(self.nn_subscribers, "1")
            srv2_reneging_rate_nn = self.compute_reneging_rate_per_server(self.nn_subscribers, "2")
            srv1_jockeying_rate_nn = self.compute_jockeying_rate_per_server(self.nn_subscribers, "1")
            srv2_jockeying_rate_nn = self.compute_jockeying_rate_per_server(self.nn_subscribers, "2")

            # Log metrics for the episode
            episode_metrics = {
                "episode": episode + 1,
                "total_reward": total_reward,
                "average_reward": total_reward / i if i > 0 else 0,
                "steps": i,
                "duration": episode_duration,
                #"policy_entropy": episode_policy_entropy / i if i > 0 else 0,
                "actor_loss": np.mean(actor_losses), #sum(actor_losses) / len(actor_losses) if actor_losses else 0,
                "critic_loss": np.mean(critic_losses), # sum(critic_losses) / len(critic_losses) if critic_losses else 0,
                "total_loss": np.mean(total_losses), # sum(total_losses) / len(total_losses) if total_losses else 0,
                "srv1_reneging_rate_markov":srv1_reneging_rate_markov,
                "srv2_reneging_rate_markov": srv2_reneging_rate_markov,
                "srv1_jockeying_rate_markov": srv1_jockeying_rate_markov,
                "srv2_jockeying_rate_markov": srv2_jockeying_rate_markov,
                "srv1_reneging_rate_nn": srv1_reneging_rate_nn,
                "srv2_reneging_rate_nn": srv2_reneging_rate_nn,
                "srv1_jockeying_rate_nn": srv1_jockeying_rate_nn,
                "srv2_jockeying_rate_nn": srv2_jockeying_rate_nn
                #"srv1_average_jockeying_rate": sum(srv1_jockeying_rates) / len(srv1_jockeying_rates) if srv1_jockeying_rates else 0,
                #"srv1_average_reneging_rate": sum(srv1_reneging_rates) / len(srv1_reneging_rates) if srv1_reneging_rates else 0,
                #"srv2_average_jockeying_rate": sum(srv2_jockeying_rates) / len(srv2_jockeying_rates) if srv2_jockeying_rates else 0,
                #"srv2_average_reneging_rate": sum(srv2_reneging_rates) / len(srv2_reneging_rates) if srv2_reneging_rates else 0
            }
            
            # Episode-level utility/impatience logging
            if len(self.current_episode_utilities) > 0:
                avg_utility_episode = float(np.nanmean(self.current_episode_utilities))
            else:
                avg_utility_episode = 0.0

            total_impatience = float(self.episode_reneges + self.episode_jockeys)
            total_requests = float(self.episode_requests_count if self.episode_requests_count>0 else 1.0)
            impatience_rate_episode = total_impatience / total_requests

            # add to episode_metrics
            episode_metrics.update({
                "avg_tenant_utility": avg_utility_episode,
                "impatience_rate": impatience_rate_episode,
                "num_requests_episode": int(self.episode_requests_count),
                "num_reneges_episode": int(self.episode_reneges),
                "num_jockeys_episode": int(self.episode_jockeys),
                # log chosen cost parameters for this episode
                "cost_wait": float(self.cost_wait),
                "cost_local": float(self.cost_local),
                "cost_switch": float(self.cost_switch),
                "value_local": float(self.value_local)
            })
            
            metrics.append(episode_metrics)            
            
            # Reset per-episode trackers for next episode
            self.current_episode_utilities = []
            self.episode_requests_count = 0
            self.episode_reneges = 0
            self.episode_jockeys = 0
            
            # Print episode summary
            print(f"Episode {episode + 1} Summary: Total Reward={episode_metrics['total_reward']}, "
                f"Avg Reward={episode_metrics['average_reward']:.2f}, Steps={episode_metrics['steps']}, "      
                f"Actor Loss={episode_metrics['actor_loss']:.4f}, Critic Loss={episode_metrics['critic_loss']:.4f}, "
                f"Total Loss={episode_metrics['total_loss']:.4f}, "
                f"Duration={episode_metrics['duration']:.2f}s")

            print(f"Episode {episode + 1} finished with a total reward of {episode_metrics['total_reward']}")
            
            # Optional: Introduce a small delay between episodes for better monitoring
            time.sleep(0.1)
        
        # Save metrics to file
        # At the end of the simulation, save metrics to file
        save_adjusted_to_file = "adjusted_metrics.csv" if adjust_service_rate else "non_adjusted_metrics.csv"
        self.save_metrics(metrics, save_adjusted_to_file)
        self.save_metrics(metrics, save_to_file)
        print(f"Metrics saved to {save_to_file}")
        
        print("Training completed.")
           
        return
        
        
    def save_metrics(self, metrics, filename):
        """
        Save episode metrics to a CSV file.

        Args:
            metrics (list): A list of dictionaries with episode metrics.
            filename (str): Path to the CSV file to save metrics.

        Returns:
            None
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
            
        print(f"Metrics successfully saved to {filename}")
    
    
    def set_batch_id(self, id):
		
        self.batchid = id
		
		
    def get_batch_id(self):
		
        return self.batchid
	
		
    def get_all_service_times(self):
        
        return self.all_serv_times  
        

    def processEntries(self,entries, batchid): #, uses_nn): # =np.int16 , actor_critic=actor_critic, =np.array([]), =np.int16 # else
        
        
        num_iterations = random.randint(1, 5)  # Random number of iterations between 1 and 5
        
        for entry in entries: 
            #print("  Adding a new request into task queue...", entry, " ====== ENTRY[1]", entry[1])       
            #if entry[1]==True:
                # print("  Adding a new request into task queue...")                
            uses_nn = random.choice([True, False])
            req = self.addNewRequest(entry[0], batchid, uses_nn)
            #self.arr_prev_times = np.append(self.arr_prev_times, entry[0])
                
            #else:                
            q_selector = random.randint(1, 2)
                
            #print("\n Jumped into Processing now .....now serving ...", q_selector)
                  
            curr_queue = self.dict_queues_obj.get("1") if q_selector == "1" else self.dict_queues_obj.get("2")
            #jockeying_rate = self.compute_jockeying_rate_per_server(self.state_subscribers, queueid) # self.compute_jockeying_rate(curr_queue)
            #reneging_rate = self.compute_reneging_rate_per_server(self.state_subscribers, queueid) # self.compute_jockeying_rate(curr_queue) # MATCH
                
            if q_selector == 1:					
                self.queueID = "1" # Server1
                    
                """
                     Run the serveOneRequest function a random number of times before continuing.
                """
                if req.uses_nn: 
                    jockeying_rate = self.compute_jockeying_rate_per_server(self.nn_subscribers, self.queueID)
                    reneging_rate = self.compute_reneging_rate_per_server(self.nn_subscribers, self.queueID)           
                    # self.serveOneRequest(self.queueID, jockeying_rate, reneging_rate, entries) # Server1 = self.dict_queues_obj["1"][0], entry[0],
                else:
                    jockeying_rate = self.compute_jockeying_rate_per_server(self.state_subscribers, self.queueID)
                    reneging_rate = self.compute_reneging_rate_per_server(self.state_subscribers, self.queueID)           
                
                self.serveOneRequest(self.queueID, jockeying_rate, reneging_rate, entries)
                                                                                                                      
                time.sleep(random.uniform(0.1, 0.5))  # Random delay between 0.1 and 0.5 seconds
                                               
            else:
                self.queueID = "2"
                
                if req.uses_nn: 
                    jockeying_rate = self.compute_jockeying_rate_per_server(self.nn_subscribers, self.queueID)
                    reneging_rate = self.compute_reneging_rate_per_server(self.nn_subscribers, self.queueID)           
                    # self.serveOneRequest(self.queueID, jockeying_rate, reneging_rate, entries) # Server1 = self.dict_queues_obj["1"][0], entry[0],
                else:
                    jockeying_rate = self.compute_jockeying_rate_per_server(self.state_subscribers, self.queueID)
                    reneging_rate = self.compute_reneging_rate_per_server(self.state_subscribers, self.queueID)
                    
                self.serveOneRequest(self.queueID,  jockeying_rate, reneging_rate, entries) # Server2 = self.dict_queues_obj["2"][0], entry[0],                                                      
                
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
        
        total_time_spent = req.time_exit + req.time_entrance
        diff = total_time_spent - time_to_service_end

        if total_time_spent < time_local_service: #diff < time_local_service:
            reward = 1  # Reward
        else:
            reward = -0.5  # Penalty

        return reward
        

    def generateLocalCompUtility(self, req):
        #req=Request(req)
        self.compute_counter = self.compute_counter + 1
        
        local_delay=req.dist_local_delay.rvs(loc=req.loc_local_delay, scale=2.0) #scale=2.0) #req.scale_local_delay)
        
        delay=float(self.time-req.time_entrance)+local_delay        
        self.local_utility = float(req.utility_basic*np.exp(-delay*req.discount_coef))

        #self.avg_delay = (self.local_utility + self.avg_delay)/self.compute_counter

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
        rate_srv1,rate_srv2 = self.srvrates_1, self.srvrates_2 # self.get_server_rates()                

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
            req=Request(uses_nn, self.uses_intensity_based,time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid, server_id=server_id)
                    
            self.nn_subscribers.append(req)
            
        else:
            req=Request(uses_nn, self.uses_intensity_based, time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid, server_id=server_id)
                    
            self.state_subscribers.append(req)          
  
        self.dict_queues_obj[server_id] = np.append(self.dict_queues_obj[server_id], req)  
        self.track_all_request_positions(server_id)  # <--- track after add      
        self.queueID = server_id        
        self.curr_req = req        
        self.all_requests.append(req)
        # After creating req
        req.server_id = server_id
        
        # Per-episode accounting
        self.episode_requests_count += 1
        
        return req #self.curr_req


    def getCustomerID(self):

        return self.customerid


    def setCurrQueueState(self, queueid):
		
        self.get_queue_curr_state()
		
        if queueid == "1": # Server1
            self.curr_state = {
                "ServerID": 1,
                "Intensity": self.arr_rate/self.srvrates_1, # get_server_rates()[0],
                "Pose":  self.get_queue_sizes([0])
                #"Wait": 
        }
        else:
            self.curr_state = {
                "ServerID":2,
                "Intensity": self.arr_rate/ self.srvrates_2, # get_server_rates()[1],
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
        
    
    def dispatch_queue_state(self, curr_queue, curr_queue_id, uses_intensity_based):
		
        rate_srv1,rate_srv2 = self.srvrates_1, self.srvrates_2 # self.get_server_rates()
		
        if "1" in curr_queue_id:
            alt_queue_id = "2"
            serv_rate = rate_srv1
        else:
            alt_queue_id = "1"
            serv_rate = rate_srv2

        curr_queue_state = self.get_queue_state(alt_queue_id) # , curr_queue)
        # print("\n ****** Rate -> ", serv_rate, type(serv_rate))
        
        self.env.update_queue_state(curr_queue_state)

        # Dispatch queue state to requests and allow them to act
        
        if not isinstance(None, type(curr_queue_state)):
            #count = 1 # to track the position of the request
            for req in curr_queue:
                curr_pose = self.get_request_position( curr_queue_id, req.customerid)
                if req.uses_nn:  # NN-based requests
                    
                    action = self.get_nn_optimized_decision(curr_queue_state) 
                               
                    next_state, reward, done, _ = self.env.step(action['action'])  # Apply action
                    # print("\n That ACTION :", action, " in state: ",curr_queue_state," will land you in the STATE: ", next_state)
                    self.agent.store_reward(reward)  # Store the reward for training

                    # Train RL model after processing each request
                    if done:
                        self.agent.update()
                    
                    if action['action'] == 0: #action == 0:
                        #print(f"ActorCriticInfo [RENEGE]: Server {alt_queue_id} in state:  {curr_queue_state}. Dispatching {next_state} to all {len(self.nn_subscribers)} requests  in server {curr_queue_id}")
                        self.makeRenegingDecision(req, curr_queue_id, next_state, curr_queue_state) # req.customerid)  # uses_intensity_based                  
                    elif action['action'] ==  1: #action == 1:
                        #print(f"ActorCriticInfo [JOCKEY]: Server {alt_queue_id} in state:  {curr_queue_state}. Dispatching {next_state} to all {len(self.nn_subscribers)} requests  in server {curr_queue_id}")
                        self.makeJockeyingDecision(req, curr_queue_id, alt_queue_id, next_state, curr_queue_state,  serv_rate) # req.customerid, serv_rate, uses_intensity_based) # STATE
                else: 
                    # print(f"Raw Markovian:  Server {alt_queue_id} in state {curr_queue_state}. Dispatching state to all {len(self.state_subscribers)} requests  in server {curr_queue_id}")
                    #jockeying_rate = self.compute_jockeying_rate(curr_queue)
                    #reneging_rate = self.compute_jockeying_rate(curr_queue)
                    
                    reneging_rate = self.compute_reneging_rate_per_server(self.state_subscribers, curr_queue_id)
                    jockeying_rate = self.compute_jockeying_rate_per_server(self.state_subscribers, curr_queue_id)
                    reward_renege = self.generateLocalCompUtility(req) # get_renege_reward(req, remaining_time)
                    
                    next_state_renege = {
                        "ServerID": "Local", 
                        "at_pose": len(curr_queue), 
                        "rate_jockeyed": jockeying_rate, 
                        "rate_reneged": reneging_rate, 
                        "expected_service_time": req.exp_time_service_end,                                       
                        "long_avg_serv_time": self.get_long_run_avg_service_time(curr_queue_id),
                        "time_service_took": req.time_exit-req.time_entrance,
                        "reward": reward_renege, 
                        "action": "reneged",	
                        "intensity_based_info": req.uses_nn,
                    }
                    self.makeRenegingDecision(req, curr_queue_id, next_state_renege, curr_queue_state)
                    #self.makeRenegingDecision(req, curr_queue_id, uses_intensity_based, req.customerid)
                    remaining_time = self.get_remaining_time(curr_pose-1, serv_rate) # self.get_remaining_time(curr_queue_id, curr_pose)
                    reward_jockey = self.get_jockey_reward(req, remaining_time)
                    next_state_jockey = {
                        "ServerID": alt_queue_id, 
                        "at_pose": len(curr_queue), 
                        "rate_jockeyed": jockeying_rate, 
                        "rate_reneged": reneging_rate, 
                        "expected_service_time": req.exp_time_service_end,                                       
                        "long_avg_serv_time": self.get_long_run_avg_service_time(curr_queue_id),
                        "time_service_took": req.time_exit-req.time_entrance,
                        "reward": reward_jockey, 
                        "action": "jockeyed",	
                        "intensity_based_info": req.uses_nn,
                    }                                   
                    self.makeJockeyingDecision(req, curr_queue_id, alt_queue_id, next_state_jockey, curr_queue_state,  serv_rate)
                    #self.makeJockeyingDecision(req, curr_queue_id, alt_queue_id, req.customerid, serv_rate, uses_intensity_based)
                
                #count = count + 1
                        
        else:
            return

    
    def get_nn_optimized_decision(self, queue_state):		
        """Uses AI model to decide whether to renege or jockey."""
        
        # Convert queue_state values to a list of numeric values, filtering out non-numeric values
        # numeric_values = [float(value) if isinstance(value, (int, float, np.number)) else 0.0 for value in queue_state.values()]
        if isinstance(queue_state, dict):
            numeric_values = [float(value) if isinstance(value, (int, float, np.number)) else 0.0 for value in queue_state.values()]
        elif isinstance(queue_state, list):
            numeric_values = [float(value) if isinstance(value, (int, float, np.number)) else 0.0 for value in queue_state]
        else:
            raise TypeError(f"Unsupported type for queue_state: {type(queue_state)}")
        
        state_tensor = torch.tensor(numeric_values, dtype=torch.float32).to(device)
        action = self.agent.select_action(state_tensor)  # self.actor_critic.select_action(state_tensor)# rate_reneged
        # print("\n Learned Action => ", "Jockey" if action == 1 else "Renege")
        return {"action": action, "nn_based": True}
       
    #    if not isinstance(None, type(state_tensor)):
    #        action, _, _, _ = self.agent.select_action(state_tensor)  # self.actor_critic.select_action(state_tensor)                          
    #        return {"action": action.cpu().numpy(), "nn_based": True}
        

    def compare_rates_by_information_source(self):
        """
        Compare reneging and jockeying rates for intensity-based and departure-based requests.
        """
        # Separate requests based on their information source
        nn_based_requests = [req for req in self.nn_subscribers] # if req.uses_nn]  # Example flag
        markov_based_requests = [req for req in self.state_subscribers] # if req.uses_nn] # not

        # Calculate rates for intensity-based requests
        
        reneging_rate_nn = self.compute_reneging_rate(nn_based_requests)
        jockeying_rate_nn = self.compute_jockeying_rate(nn_based_requests)

        # Calculate rates for departure-based requests
        reneging_rate_markov = self.compute_reneging_rate(markov_based_requests)
        jockeying_rate_markov = self.compute_jockeying_rate(markov_based_requests)
        
        # Print and return the comparison
        comparison = {
            "nn_based": {
                "reneging_rate": reneging_rate_nn,
                "jockeying_rate": jockeying_rate_nn,
            },
            "markov_based": {
                "reneging_rate": reneging_rate_markov,
                "jockeying_rate": jockeying_rate_markov,
            },
        }       
        
        return comparison
        
        

        def plot_information_source_comparison(comparison):
            sources = ['NN-based', 'Markov-based']
            reneging_rates = [
                comparison['nn_based']['reneging_rate'],
                comparison['markov_based']['reneging_rate'],
            ]
            jockeying_rates = [
                comparison['nn_based']['jockeying_rate'],
                comparison['markov_based']['jockeying_rate'],
            ]

            x = range(len(sources))
            width = 0.35

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x, reneging_rates, width, label='Reneging Rate', color='red')
            ax.bar([i + width for i in x], jockeying_rates, width, label='Jockeying Rate', color='blue')

            ax.set_xlabel('Information Source')
            ax.set_ylabel('Rate')
            ax.set_title('Reneging and Jockeying Rates by Queue and Information Source')
            ax.set_xticks([i + width / 2 for i in x])
            ax.set_xticklabels(sources)
            ax.legend()
            plt.tight_layout()
            plt.show()
        
    
        comparison = compare_rates_by_information_source()
        plot_information_source_comparison(comparison)
   
    
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
            self.dispatch_queue_state( curr_queue, queue_id, self.uses_intensity_based) # alt_queue,
        
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
            self.dispatch_queue_state( curr_queue, queue_id, self.uses_intensity_based)  #  alt_queue,
        
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
        Dispatch the status of all queues and collect jockeying and reneging rates
        for both raw state and NN-based information sources.
        """
        for queue_id in ["1", "2"]:
            curr_queue = self.dict_queues_obj[queue_id]
            queue_size = len(curr_queue)
            serv_rate = self.dict_servers_info[queue_id]
            queue_intensity = self.objQueues.get_arrivals_rates() / serv_rate

            # Compute jockeying and reneging rates for raw state
            '''
            jockeying_rate_raw = self.compute_jockeying_rate(curr_queue)
            reneging_rate_raw = self.compute_reneging_rate(curr_queue)

            # Compute jockeying and reneging rates for NN-based information
            jockeying_rate_nn = jockeying_rate_raw * 1.1  # Example logic (adjust based on your simulation logic)
            reneging_rate_nn = reneging_rate_raw * 0.9    # Example logic (adjust based on your simulation logic)
            '''
            
            #if not uses_nn:
            #    reneging_rate_raw = self.compute_reneging_rate(self.state_subscribers)
            #    jockeying_rate_raw = self.compute_jockeying_rate(self.state_subscribers)                   
                
            #else:
			#	reneging_rate_nn = self.compute_reneging_rate(self.nn_subscribers)
			#	jockeying_rate_nn = self.compute_jockeying_rate(self.nn_subscribers)
			
			# For Markov/raw subscribers
            srv1_reneging_rate_markov = self.compute_reneging_rate_per_server(self.state_subscribers, "1")
            srv2_reneging_rate_markov = self.compute_reneging_rate_per_server(self.state_subscribers, "2")
            srv1_jockeying_rate_markov = self.compute_jockeying_rate_per_server(self.state_subscribers, "1")
            srv2_jockeying_rate_markov = self.compute_jockeying_rate_per_server(self.state_subscribers, "2")

            # For NN subscribers
            srv1_reneging_rate_nn = self.compute_reneging_rate_per_server(self.nn_subscribers, "1")
            srv2_reneging_rate_nn = self.compute_reneging_rate_per_server(self.nn_subscribers, "2")
            srv1_jockeying_rate_nn = self.compute_jockeying_rate_per_server(self.nn_subscribers, "1")
            srv2_jockeying_rate_nn = self.compute_jockeying_rate_per_server(self.nn_subscribers, "2")				 
				
            # Update dispatch data
            if "1" in queue_id:
                self.dispatch_data["server_1"]["num_requests"].append(queue_size)
                self.dispatch_data["server_1"]["jockeying_rate_raw"].append(srv1_jockeying_rate_markov)
                self.dispatch_data["server_1"]["jockeying_rate_nn"].append(srv1_jockeying_rate_nn)
                self.dispatch_data["server_1"]["reneging_rate_raw"].append(srv1_reneging_rate_markov)
                self.dispatch_data["server_1"]["reneging_rate_nn"].append(srv1_reneging_rate_nn)
                self.dispatch_data["server_1"]["queue_intensity"].append(queue_intensity)
            else:
                self.dispatch_data["server_2"]["num_requests"].append(queue_size)
                self.dispatch_data["server_2"]["jockeying_rate_raw"].append(srv2_jockeying_rate_markov)
                self.dispatch_data["server_2"]["jockeying_rate_nn"].append(srv2_jockeying_rate_nn)
                self.dispatch_data["server_2"]["reneging_rate_raw"].append(srv2_reneging_rate_markov)
                self.dispatch_data["server_2"]["reneging_rate_nn"].append(srv2_reneging_rate_nn)
                self.dispatch_data["server_2"]["queue_intensity"].append(queue_intensity)

        # Dispatch NN-based and raw server status information
        self.use_nn = True
        self.dispatch(self.uses_nn)  # Dispatch NN-based information
        self.use_nn = False
        self.dispatch(self.uses_nn)  # Dispatch raw server status information            
            
        
    def setup_dispatch_intervals(self):
        """
        Set up the intervals for dispatching the queue status information.
        """
        
        #schedule.every(10).seconds.do(self.dispatch_all_queues)
        schedule.every(30).seconds.do(self.dispatch_all_queues)
        #schedule.every(60).seconds.do(self.dispatch_all_queues)       
            
            
    def plot_rates_original(self):
        """
        Plot the comparison of jockeying and reneging rates
        for each queue for each individual information source subscribed to.
        """
        sources = ["Raw Markov State", "NN-based"]  # Example information sources
        queues = ["Server 1", "Server 2"]

        fig, axes = plt.subplots(len(queues), 2, figsize=(12, 10))
        fig.suptitle("Comparison of Jockeying and Reneging Rates by Queue and Information Source", fontsize=16)

        for i, queue in enumerate(queues):
            jockeying_rates = [
                # self.dispatch_data[f"server_{i + 1}"]["jockeying_rate"],
                #self.dispatch_data[f"server_{i + 1}"]["nn_jockeying_rate"]
                self.dispatch_data[f"server_{i + 1}"]["jockeying_rate_raw"],  # For raw state                
                self.dispatch_data[f"server_{i + 1}"]["jockeying_rate_nn"]  # For NN-based
            ]
            reneging_rates = [
                self.dispatch_data[f"server_{i + 1}"]["reneging_rate_raw"],
                self.dispatch_data[f"server_{i + 1}"]["reneging_rate_nn"]
            ]

            x = range(len(jockeying_rates[0]))  # Assumes equal length for all sources

            # Plot jockeying rates
            axes[i, 0].plot(x, jockeying_rates[0], label=f'{sources[0]}', linestyle='dashed')
            axes[i, 0].plot(x, jockeying_rates[1], label=f'{sources[1]}')
            axes[i, 0].set_title(f"Jockeying Rates - {queue}")
            axes[i, 0].set_xlabel("Number of Requests")
            axes[i, 0].set_ylabel("Jockeying Rate")
            axes[i, 0].legend()

            # Plot reneging rates
            axes[i, 1].plot(x, reneging_rates[0], label=f'{sources[0]}', linestyle='dashed')
            axes[i, 1].plot(x, reneging_rates[1], label=f'{sources[1]}')
            axes[i, 1].set_title(f"Reneging Rates - {queue}")
            axes[i, 1].set_xlabel("Number of Requests")
            axes[i, 1].set_ylabel("Reneging Rate")
            axes[i, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    def plot_rates(self):
        """
        Plot the comparison of jockeying and reneging rates
        for each queue for each individual information source subscribed to.

        Modifications:
        - Use a logarithmic y-axis on each subplot to improve visibility when rates are small.
        - Add clean major-grid only (no dense minor grid) for readability.
        - Return the matplotlib Figure object (so callers can save or modify it).
        """
        sources = ["Raw Markov State", "NN-based"]
        queues = ["Server 1", "Server 2"]

        def collect_rates(subscribers, server_id):
            jockeying_rates = []
            reneging_rates = []
            queue_sizes = []
            reqs = [req for req in subscribers if getattr(req, "server_id", None) == server_id]
            for i in range(1, len(reqs) + 1):
                current = reqs[:i]
                queue_sizes.append(i)
                num_jockeyed = sum(getattr(req, "jockeyed", False) for req in current)
                num_reneged = sum(getattr(req, "reneged", False) for req in current)
                jockeying_rates.append(num_jockeyed / i)
                reneging_rates.append(num_reneged / i)
            return queue_sizes, jockeying_rates, reneging_rates

        fig, axes = plt.subplots(len(queues), 2, figsize=(12, 10))
        eps = 1e-3 #1e-9

        for i, server_id in enumerate(["1", "2"]):
            queue_sizes_state, jockeying_state, reneging_state = collect_rates(self.state_subscribers, server_id)
            queue_sizes_nn, jockeying_nn, reneging_nn = collect_rates(self.nn_subscribers, server_id)

            # convert and add eps to avoid zeros
            jockeying_state_arr = (np.array(jockeying_state) if len(jockeying_state) > 0 else np.array([])) + eps
            jockeying_nn_arr = (np.array(jockeying_nn) if len(jockeying_nn) > 0 else np.array([])) + eps
            reneging_state_arr = (np.array(reneging_state) if len(reneging_state) > 0 else np.array([])) + eps
            reneging_nn_arr = (np.array(reneging_nn) if len(reneging_nn) > 0 else np.array([])) + eps

            ax_jock = axes[i, 0]
            if len(queue_sizes_state) > 0:
                ax_jock.plot(queue_sizes_state, jockeying_state_arr, label=f"{sources[0]}", linestyle="dashed", color="blue")
            if len(queue_sizes_nn) > 0:
                ax_jock.plot(queue_sizes_nn, jockeying_nn_arr, label=f"{sources[1]}", color="orange")
            ax_jock.set_title(f"Jockeying Rates - {queues[i]}")
            ax_jock.set_xlabel("Number of Requests")
            ax_jock.set_ylabel("Jockeying Rate (log scale)")
            ax_jock.set_yscale("log")
            ax_jock.set_ylim(bottom=eps)
            ax_jock.legend(loc="best", fontsize=12, ncol=2)
            ax_jock.grid(which="major", linestyle="--", alpha=0.7)
            ax_jock.minorticks_off()

            ax_renege = axes[i, 1]
            if len(queue_sizes_state) > 0:
                ax_renege.plot(queue_sizes_state, reneging_state_arr, label=f"{sources[0]}", linestyle="dashed", color="blue")
            if len(queue_sizes_nn) > 0:
                ax_renege.plot(queue_sizes_nn, reneging_nn_arr, label=f"{sources[1]}", color="orange")
            ax_renege.set_title(f"Reneging Rates - {queues[i]}")
            ax_renege.set_xlabel("Number of Requests")
            ax_renege.set_ylabel("Reneging Rate (log scale)")
            ax_renege.set_yscale("log")
            ax_renege.set_ylim(bottom=eps)
            # ax_renege.legend()
            ax_renege.legend(loc="best", fontsize=12)
            ax_renege.grid(which="major", linestyle="--", alpha=0.7)
            ax_renege.minorticks_off()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("logarithmic_rates_perqueue_length.pdf", format="pdf", dpi=300, bbox_inches="tight")
        # plt.show()
        return fig

    
    def plot_rates_old(self):
        """
        Plot the comparison of jockeying and reneging rates
        for each queue for each individual information source subscribed to.
        This version uses self.nn_subscribers and self.state_subscribers
        instead of dispatch_data.
        """
    
        sources = ["Raw Markov State", "NN-based"]  # Information sources
        queues = ["Server 1", "Server 2"]

        # Helper function to collect rates for each queue and info source
        def collect_rates(subscribers, server_id):
            jockeying_rates = []
            reneging_rates = []
            queue_sizes = []
            # Get requests for this server
            reqs = [req for req in subscribers if getattr(req, "server_id", None) == server_id]
            for i in range(1, len(reqs)+1):
                current = reqs[:i]
                queue_sizes.append(i)
                num_jockeyed = sum(getattr(req, 'jockeyed', False) for req in current)
                num_reneged = sum(getattr(req, 'reneged', False) for req in current)
                jockeying_rates.append(num_jockeyed / i)
                reneging_rates.append(num_reneged / i)
            return queue_sizes, jockeying_rates, reneging_rates

        fig, axes = plt.subplots(len(queues), 2, figsize=(12, 10))
        # fig.suptitle("Comparison of Jockeying and Reneging Rates by Queue and Information Source", fontsize=16)

        for i, server_id in enumerate(["1", "2"]):
            # Markov (raw state)
            queue_sizes_state, jockeying_state, reneging_state = collect_rates(self.state_subscribers, server_id)
            # NN-based
            queue_sizes_nn, jockeying_nn, reneging_nn = collect_rates(self.nn_subscribers, server_id)

            # Plot jockeying rates
            axes[i, 0].plot(queue_sizes_state, jockeying_state, label=f'{sources[0]}', linestyle='dashed', color='blue')
            axes[i, 0].plot(queue_sizes_nn, jockeying_nn, label=f'{sources[1]}', color='orange')
            axes[i, 0].set_title(f"Jockeying Rates - {queues[i]}")
            axes[i, 0].set_xlabel("Number of Requests")
            axes[i, 0].set_ylabel("Jockeying Rate")
            axes[i, 0].legend()

            # Plot reneging rates
            axes[i, 1].plot(queue_sizes_state, reneging_state, label=f'{sources[0]}', linestyle='dashed', color='blue')
            axes[i, 1].plot(queue_sizes_nn, reneging_nn, label=f'{sources[1]}', color='orange')
            axes[i, 1].set_title(f"Reneging Rates - {queues[i]}")
            axes[i, 1].set_xlabel("Number of Requests")
            axes[i, 1].set_ylabel("Reneging Rate")
            axes[i, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
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
        
        if self.total_served_requests_srv1 == 0 or self.total_served_requests_srv2 == 0:
            return total_service_time      
    
        if  "1" in queue_id:
            
            for req in self.dict_queues_obj["1"]:                
                total_service_time += req.service_time # exp_time_service_end                 
                             
            return total_service_time / self.total_served_requests_srv1
        else:
            
            for req in self.dict_queues_obj["2"]:                
                total_service_time += req.service_time  # exp_time_service_end                  
                              
            return total_service_time / self.total_served_requests_srv2                   
    
    
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
                    "intensity_based_info": self.uses_intensity_based,			 
			})	
    
                    
    def get_queue_state(self, queueid): # , queueobj): #, action):      
		
        if "1" in queueid:	
            if len(self.srv1_history) > 0:
                latest_state = self.srv1_history[-1]
                return latest_state
                             
            else:		 
                self.srv1_history.append({
				    "ServerID": queueid, 
                    "at_pose": 0, 
                    "rate_jockeyed": 0.0, 
                    "rate_reneged": 0.0, 
                    "expected_service_time": 0.0,                                   
                    "long_avg_serv_time": self.long_avg_serv_time,
                    "time_service_took": 0.0,
                    "reward": 0.0, 
                    "action":"served",	
                    "intensity_based_info": self.uses_intensity_based,			 
			    })	
                return self.srv1_history
        else:
            if len(self.srv2_history) > 0:
                latest_state = self.srv2_history[-1]
                return latest_state
                             
            else:	
                self.srv2_history.append({
                    "ServerID": queueid, 
                    "at_pose": 0, 
                    "rate_jockeyed": 0.0, 
                    "rate_reneged": 0.0, 
                    "expected_service_time": 0.0,                                   
                    "long_avg_serv_time": self.long_avg_serv_time,
                    "time_service_took": 0.0,
                    "reward": 0.0, 
                    "action":"served",
                     "intensity_based_info": self.uses_intensity_based,				 
                })	               
                          
                return self.srv2_history  # self.history[0]                      
    

    def serveOneRequest(self, queueID,  jockeying_rate, reneging_rate, arrived_now): # dispatch_all_queues
        #randomly select which queue to process at a time t+1
        q_selector = random.randint(1, 2)                           
        
        """Process a request and use the result for training the RL model."""             
         
        if "1" in queueID:
            queue = self.dict_queues_obj["1"]            
            serv_rate = self.dict_servers_info["1"]
            self.long_avg_serv_time = self.get_long_run_avg_service_time(queueID)
            
        else:
            queue = self.dict_queues_obj["2"]
            serv_rate = self.dict_servers_info["2"]
            self.long_avg_serv_time = self.get_long_run_avg_service_time(queueID)            

        if len(queue) == 0:
            return  # No request to process
        
        queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
        req = queue[0]  # Process the first request
        queue = queue[1:]  # Remove it from the queue
        req.time_exit = self.time

        # Compute reward based on actual waiting time vs. expected time
        time_in_queue = req.time_exit-req.time_entrance 
        reward = 1.0 if time_in_queue < req.service_time else 0.0
        
        len_queue_1, len_queue_2 = self.get_queue_sizes()
        
        # Mark as served for utility accounting
        req.served = True

        # Compute observed utility for request and store in episode list
        util = req.compute_observed_utility(c_wait=self.cost_wait, c_local=self.cost_local, c_switch=self.cost_switch, v_local=self.value_local)
        self.current_episode_utilities.append(util)
                              
        
        if "1" in queueID:
            self.total_served_requests_srv1+=1
            self.objObserv.set_obs(queueID, len_queue_1, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time,  req.uses_nn) # self.uses_intensity_based
            self.srv1_history.append(self.objObserv.get_obs())
            curr_intensity =  self.srv1_history[-1]
            
            if self.total_served_requests_srv1 >= abs(len(arrived_now)/2): # and "1" in queueID: 
                self.departure_dispatch_count += 1
                self.uses_intensity_based = False                              
                
            if curr_intensity['this_busy'] <= 2.0:
                self.intensity_dispatch_count += 1
                self.uses_intensity_based = True
                                       
            self.history.append(self.objObserv.get_obs())
            #self.objObserv.set_obs(queueID, len_queue_1, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time, self.uses_intensity_based )
            self.dispatch_queue_state(queue, queueID, self.uses_intensity_based)  
        else:
            self.total_served_requests_srv2+=1
            self.objObserv.set_obs(queueID, len_queue_2, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time, req.uses_nn) #self.uses_intensity_based )
            self.srv2_history.append(self.objObserv.get_obs())
            curr_intensity =  self.srv2_history[-1]
            
            if self.total_served_requests_srv2 >= abs(len(arrived_now)/2): # and "2" in queueID:
                self.departure_dispatch_count += 1
                self.uses_intensity_based = False
            
            self.history.append(self.objObserv.get_obs())
              
            self.dispatch_queue_state(queue, queueID, self.uses_intensity_based)                                        
        
        # Store the experience for RL training        
        state = self.get_queue_state(queueID) 
        
        
        '''
            Dispatch updated queue state - measure the value of what information
            1. Dispatch after the number of departures equal to the number of new arrivals at that run
            2. Dispatch only when the queue intensity is less than or equal to 1 
        '''               
        
        if not isinstance(None, type(state)):
            action = self.agent.select_action(state)
            self.agent.store_reward(reward)

            # Train RL model after each request is processed Observed:
            self.agent.update()
                            
        return 	    
        
			
    def get_jockey_reward_old(self, req):
		
        reward = 0.0
        if not isinstance(req.customerid, type(None)):	
            # print("\n Current Request: ", req)
            if '_jockeyed' in req.customerid:
                if self.avg_delay+req.time_entrance < req.service_time: #exp_time_service_end: That ACTION
                    reward = 1.0
                else:
                    reward = 0.0
                    
        return reward
        
    
    def get_jockey_reward(self, req, prev_remaining_time):
        """
        Compute reward for a jockeyed event.
    
        Reward is 1 if total time spent in the system (from entry to exit)
        is less than the time that was left to get served in the previous queue.
        Penalize with -1 otherwise.

        :param req: The request object.
        :param prev_remaining_time: Time left to get served in the previous queue before jockeying.
        :return: 1 (reward) or -1 (penalty)
        """
        # Total time spent from entry to exit
        total_time_spent = req.time_exit - req.time_entrance

        # Reward logic
        if total_time_spent < prev_remaining_time:
            return 1
        else:
            return -0.5
            
            
    def get_renege_reward(self, req, local_processing):
        """
        Compute reward for a reneging event.
    
        Reward is 1 if you renege and the remaining time
        before getting served is less than the time to process locally 

        :param req: The request object.
        :param prev_remaining_time: Time left to get served in the previous queue before jockeying.
        :return: 1 (reward) or -1 (penalty)
        """
        # Total time spent from entry to exit
        total_time_spent = req.time_exit - req.time_entrance

        # Reward logic
        if total_time_spent < local_processing:
            return 1
        else:
            return -0.5
        
    
    def get_history(self, queueid):
		
        if "1" in queueid:
            return self.srv1_history
        else:
            return self.srv2_history   
    
    
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
        
        
    def get_remaining_time_old(self, queue_id, position): # plot_rates
        """
        Calculate the remaining time until a request at a given position is processed.
        
        :param queue_id: The ID of the queue (1 or 2).
        :param position: The position of the request in the queue (0-indexed).
        :return: Remaining time until the request is processed.
        """
        if "1" in queue_id:
            serv_rate = self.dict_servers_info["1"]  
            queue = self.dict_queues_obj["1"]  
        else:
            serv_rate = self.dict_servers_info["2"]  
            queue = self.dict_queues_obj["2"] 

        queue_length = len(queue)
       
        if position < 0 or position > queue_length: #=
            print("\n Pose: ", position, queue_length) 
            raise ValueError("Invalid position: Position must be within the bounds of the queue length.")
        
        # Calculate the remaining time based on the position and service rate
        remaining_time = sum(np.random.exponential(1 / serv_rate) for _ in range(position + 1))
        
        return remaining_time
        
  
    def get_remaining_time(self, k, mu_i):
        """
        Returns the total remaining time for a pending request with k requests ahead,
        where each request ahead has an iid exponential service time with rate mu_i.
    
        Parameters:
            k (int): Number of requests ahead in the queue.
            mu_i (float): Service rate (requests per unit time).
    
        Returns:
            float: Total remaining time (sum of k exponential random variables).
        """
        # Each X_{i,t} ~ Exp(mu_i), so sum of k exponentials
        if k <= 0 or mu_i <= 0:
            return 0.0
            
        return np.sum(np.random.exponential(scale=1/mu_i, size=k))
        
        
    def calculate_max_cloud_delay_old(self, position, queue_intensity, req):
        """
        Calculate the max cloud delay based on the position in the queue and the current queue intensity.
        
        :param position: The position of the request in the queue (0-indexed).
        :param queue_intensity: The current queue intensity.
        :return: The max cloud delay.
        """
        if "1" in req.server_id:
            srv_rate = self.srvrates_1
        else:
            srv_rate = self.srvrates_2
			
        if position is None:          
            '''            
            k₀: initial position at entry
            μ: service rate (requests per unit time)
            Δt: time elapsed since entry
            '''         
			    
            position = max(1, req.pos_in_queue - int(srv_rate * self.time-req.time_entrance))
                
        base_delay = 1.0  # Base delay for the first position # req.service_time 
        position_factor = 0.01  # Incremental delay factor per position
        intensity_factor = 2.0  # Factor to adjust delay based on queue intensity

        # Calculate the position-dependent delay
        position_delay = base_delay + (position * position_factor)

        # Adjust for queue intensity
        max_cloud_delay = position_delay * (1 + (queue_intensity / intensity_factor))
        
        return max_cloud_delay
        
    
    def calculate_max_cloud_delay(self, position, queue_intensity, req):
        # Use M/M/1 expected wait time as the cloud delay
        if "1" in req.server_id:
            srv_rate = self.srvrates_1
        else:
            srv_rate = self.srvrates_2

        if position is None:
            position = req.pos_in_queue

        # Expected waiting time: sum of k exponentials = k / mu
        if srv_rate > 0 and position >= 0:
            max_cloud_delay = position / srv_rate
        else:
            max_cloud_delay = 0.0

        # print(f"DEBUG calculate_max_cloud_delay: position={position}, srv_rate={srv_rate}, max_cloud_delay={max_cloud_delay}")
        return max_cloud_delay
        
        
    def calculate_max_cloud_delay_erring(self, position, mu):
        import scipy.linalg as la
        from scipy.integrate import quad
        
        def build_q_matrix(k: int, mu: float) -> np.ndarray:
            """
            Build the (k+2)x(k+2) generator Q for states i=0,...,k+1,
            where state i->i-1 at rate 2*mu for i>=2, and at rate mu for i==1.
            State 0 is absorbing.            
            """
            
            if k is None:
                # Option 1: Return a default value (e.g., 0 or some defined constant)
                return 0
                
            size = k+2
            Q = np.zeros((size, size))
            for i in range(1, size):
                rate = 2*mu if i >= 2 else mu
                Q[i, i-1] = rate
                Q[i, i]   = -rate
            
            return Q

        """
        Solve for E[T] starting from state k+1 by using fundamental matrix.
          E = -Q^{-1} * 1  restricted to non-absorbing states.
        """
        Q = build_q_matrix(position, mu)
        # Extract transient states 1..k+1
        Q_tt = Q[1:, 1:]
        # Fundamental matrix N = -inv(Q_tt)
        N = -la.inv(Q_tt)
        # Expected absorption time vector: t = N @ 1
        t_vec = N.dot(np.ones(position+1))
        # We start in state k+1 => index k  in t_vec
        return t_vec[position]
    
		        
    def log_action(self, action, req, queueid):
        """Logs the request action to the file."""
        
        logging.info("", extra={"request_id": req.customerid, "queue_id": queueid, "action": action})
        
        
    def makeRenegingDecision(self, req, queueid, next_state, curr_queue_state): # req, curr_queue_id, next_state, 
        
        decision=False                  
        
        if "1" in queueid:
            serv_rate = self.srvrates_1 # self.dict_servers_info["1"]
            queue =  self.dict_queues_obj["1"]  
            dest_queue = self.dict_queues_obj["2"]    
        else:
            serv_rate = self.srvrates_2 # self.dict_servers_info["2"] 
            queue =  self.dict_queues_obj["2"]
            dest_queue = self.dict_queues_obj["1"] 
        
        if self.learning_mode=='transparent':
            self.max_cloud_delay=stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=1/serv_rate) #(self.certainty,a=self.pos_in_queue,loc=0,scale=1/serv_rate) #self.serv_rate)

        else:	
            curr_pose = self.get_request_position( queueid, req.customerid)		
            num_observations=min(len(self.objObserv.get_renege_obs()), len(self.history)) # queueid, queue),len(self.history)
            mean_interval=np.mean(num_observations) # unbiased estimation of 1/lambda where lambda is the service rate
            if np.isnan(mean_interval):
                mean_interval=0
            if mean_interval!=0:
                self.serv_rate=1/mean_interval
            # k_erlang=req.pos_in_queue*num_observations #self.pos_in_queue*num_observations
            # scale_erlang=mean_interval*k_erlang
            
            if np.isnan(mean_interval):
                self.max_cloud_delay=np.Inf
            else:
                self.max_local_delay = self.generateLocalCompUtility(req)
                queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
                self. max_cloud_delay = self.calculate_max_cloud_delay(curr_pose, queue_intensity, req) #self.calculate_max_cloud_delay(curr_pose, serv_rate)                                                           
            
            #if "1" in queueid:
            #    self.queue = self.dict_queues_obj["1"] 
            #    serv_rate = self.srvrates_1           
            #else:
            #    self.queue = self.dict_queues_obj["2"] 
            #    serv_rate = self.srvrates_2
             
            self.avg_delay = self.calculate_max_cloud_delay(len(dest_queue)+1, queue_intensity, req)    
            # Get the relevant queue
            queue = self.dict_queues_obj[queueid]
           
            found = False                              
            # Compute remaining waiting time (number of requests ahead * average service time)
            # For M/M/1, expected remaining time = position * 1/service_rate
            # remaining_wait_time = pos * (1.0 / serv_rate) if serv_rate > 0 else 1e4  # avoid zero division
                
            #if customer_id in req.customerid: # _in_queue
                
            remaining_wait_time = self.get_remaining_time(curr_pose-1, serv_rate) # get_remaining_time(queueid, curr_pose)  # pos) #               
            renege = (remaining_wait_time > self.max_local_delay) #T_local)
                                    
            if renege:
                decision = True
                diff_wait = abs(remaining_wait_time - self.avg_delay) # jockey_total_expected_time_to_service
                reward = self.reqRenege(req, queueid, curr_pose, serv_rate, queue_intensity, self.max_local_delay, req.customerid, req.service_time, decision, queue, req.uses_nn, diff_wait)
                #self.reqRenege(req_in_queue, queueid, pos, serv_rate, T_local, req_in_queue.customerid, req_in_queue.service_time, decision, queue, anchor)
                found = True
                #return 
                    # break

                if not found:
                    print(f" Request ID {req.customerid} not found in queue {queueid}. Continuing with processing...")
                    return # False    
                       
        self.curr_req = req                


    def reqRenege(self, req, queueid, curr_pose, serv_rate, queue_intensity, time_local_service, customerid, time_to_service_end, decision, curr_queue, uses_nn, diff_wait):
        
        if decision:
            decison = 1
        else:
            decision = 0

        if curr_pose >= len(curr_queue):
            return
            
        else:
            # Get the queue as a list
            queue = list(self.dict_queues_obj[queueid])
            # Remove the request at curr_pose if it exists
            if 0 <= curr_pose < len(queue):
                queue.pop(curr_pose)
            # Update the main queue object
            self.dict_queues_obj[queueid] = queue
            self.track_all_request_positions(queueid)
            self.queueID = queueid  
        
            # req.customerid = req.customerid+"_reneged"
            if not req.customerid.endswith("_reneged"):
                req.customerid = req.customerid+"_reneged"
                
            req.reneged = True
            req.time_exit = self.time  
            
            self.log_action("Reneged", req, queueid)
            
            # compute and record utility for this reneged request
            util = req.compute_observed_utility(c_wait=self.cost_wait, c_local=self.cost_local, c_switch=self.cost_switch, v_local=self.value_local)
            self.current_episode_utilities.append(util)
            self.episode_reneges += 1
        
            # reward = self.getRenegeRewardPenalty(req, time_local_service, time_to_service_end)
            reward = self.get_renege_reward( req, time_local_service)
            print(colored("%s", 'green') % (req.customerid) + " in Server %s" %(queueid) + " reneging now, to local processing with reward %f "%(reward) )
                                     
            #self.reneging_rate = self.compute_reneging_rate(curr_queue)
            #self.jockeying_rate = self.compute_jockeying_rate(curr_queue)
            self.long_avg_serv_time = self.get_long_run_avg_service_time(queueid)  
            
            if not uses_nn:
                self.reneging_rate = self.compute_reneging_rate_per_server(self.state_subscribers, queueid) # self.compute_reneging_rate(self.state_subscribers)
                self.jockeying_rate = self.compute_jockeying_rate_per_server(self.state_subscribers, queueid) # self.compute_jockeying_rate(self.state_subscribers)                   
                self.objObserv.set_renege_obs(queueid, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, time_local_service, req, reward, decision, self.long_avg_serv_time, uses_nn, diff_wait)
                #self.curr_obs_renege.append(self.objObserv.get_renege_obs())
            else:
                self.reneging_rate = self.compute_reneging_rate_per_server(self.nn_subscribers, queueid) # self.compute_reneging_rate(self.nn_subscribers)
                self.jockeying_rate = self.compute_jockeying_rate_per_server(self.nn_subscribers, queueid) # self.compute_jockeying_rate(self.nn_subscribers)
                self.objObserv.set_renege_obs(queueid, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, time_local_service, req, reward, decision, self.long_avg_serv_time, uses_nn, diff_wait) 
                #self.curr_obs_renege.append(self.objObserv.get_renege_obs())            
            
            self.curr_req = req
            
            return reward
            
            
    def get_current_renege_count(self):
		
        return self.objObserv.get_renege_obs()


    def get_request_position_old(self, queue_id, request_id):
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
        
        
    def get_request_position(self, queue_id, request_id):
        """
        Get the position of a given request in the queue, update its position history,
        and also update 'at_pose' in self.curr_obs_jockey for the matching request.
        """
        if "1" in queue_id:
            queue = self.dict_queues_obj["1"]
        else:
            queue = self.dict_queues_obj["2"]

        found_position = None
        for position, req in enumerate(queue):
            if req.customerid == request_id:
                found_position = position
                # Track the position history
                self.request_position_history[request_id].append({
                    "time": self.time,
                    "queue_id": queue_id,
                    "position": position
                })
                break

        if found_position is None:
            # Not found: track position as -1
            self.request_position_history[request_id].append({
                "time": self.time,
                "queue_id": queue_id,
                "position": -1
            })

        # Update self.curr_obs_jockey
        for obs in self.curr_obs_jockey:
            req = obs.get("Request")
            if req is not None and request_id in req.customerid:
                obs["at_pose"] = found_position if found_position is not None else -1

        return found_position
        
        
    def get_request_position_history(self, request_id):
        """
        Get the recorded position history of the specified request.
        :param request_id: The ID of the request.
        :return: List of dicts with keys 'time', 'queue_id', 'position'
        """
        return self.request_position_history.get(request_id, [])
        
            
    def reqJockey(self, curr_queue_id, dest_queue_id, req, customerid, serv_rate, remaining_time, exp_delay, decision, curr_pose, curr_queue, uses_nn, diff_wait):	
		# Convert here the decision from bool to numeric - true = 1, false = 0
		
		# Do not allow jockeying for requests that have already reneged
        if '_reneged' in req.customerid:
            return  # Ignore this request
			        
        if decision:
            decision = 1
        else:
            decision = 0
			
        if curr_pose >= len(curr_queue):
            return                   
            
        else:	               
            # Remove from current queue
            queue = list(self.dict_queues_obj[curr_queue_id])
            if 0 <= curr_pose < len(queue):
                queue.pop(curr_pose)
            self.dict_queues_obj[curr_queue_id] = queue
            self.track_all_request_positions(curr_queue_id)  # <--- track after add

            # Add to destination queue
            dest_queue_list = list(self.dict_queues_obj[dest_queue_id])
            dest_queue_list.append(req)
            self.dict_queues_obj[dest_queue_id] = dest_queue_list
            self.track_all_request_positions(dest_queue_id) 
            
            if not req.customerid.endswith("_jockeyed"):
                req.customerid = req.customerid+"_jockeyed"
                
            req.jockeyed = True
            req.time_exit = self.time  
            
            # compute and record utility for this jockeyed request (note: jockeying often means they may re-enter another queue;
            # if you treat jockeying as immediate exit for utility accounting, keep this. Otherwise, adapt to compute only on final exit)
            util = req.compute_observed_utility(c_wait=self.cost_wait, c_local=self.cost_local, c_switch=self.cost_switch, v_local=self.value_local)
            self.current_episode_utilities.append(util)
            self.episode_jockeys += 1
        
            if  "1" in curr_queue_id: # Server1
                queue_intensity = self.arr_rate/self.dict_servers_info["1"] # Server1
            
            else:
                queue_intensity = self.arr_rate/self.dict_servers_info["2"] # Server2
        
            reward = self.get_jockey_reward(req, remaining_time)
                              
            print(colored("%s", 'green') % (req.customerid) + " in Server %s" %(curr_queue_id) + " jockeying now, to Server %s" % (colored(dest_queue_id,'green'))+ " with reward %f"%(reward))                      
            
            self.log_action(f"Jockeyed", req, dest_queue_id)
            
            #self.reneging_rate = self.compute_reneging_rate(queue)
            #self.jockeying_rate = self.compute_jockeying_rate(queue)     
            self.long_avg_serv_time = self.get_long_run_avg_service_time(curr_queue_id)
            
            if not uses_nn:
                self.reneging_rate = self.compute_reneging_rate_per_server(self.state_subscribers, curr_queue_id) # self.compute_reneging_rate(self.state_subscribers)
                self.jockeying_rate = self.compute_jockeying_rate_per_server(self.state_subscribers, curr_queue_id) # self.compute_jockeying_rate(self.state_subscribers)                   
                self.objObserv.set_jockey_obs(curr_queue_id, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, exp_delay, req, reward, decision, self.long_avg_serv_time , uses_nn, diff_wait) #req.exp_time_service_end
                #self.curr_obs_jockey.append(self.objObserv.get_jockey_obs())
            else:
                self.reneging_rate = self.compute_reneging_rate_per_server(self.nn_subscribers, curr_queue_id) # self.compute_reneging_rate(self.nn_subscribers)
                self.jockeying_rate = self.compute_jockeying_rate_per_server(self.nn_subscribers, curr_queue_id) # self.compute_jockeying_rate(self.nn_subscribers)
                self.objObserv.set_jockey_obs(curr_queue_id, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, exp_delay, req, reward, decision, self.long_avg_serv_time , uses_nn, diff_wait) #req.exp_time_service_end
                #self.curr_obs_jockey.append(self.objObserv.get_jockey_obs())
            #if not uses_nn:
	        #	self.reneging_rate = self.compute_reneging_rate(self.state_subscribers)
            #    self.jockeying_rate = self.compute_jockeying_rate(self.state_subscribers)                   
            #    self.objObserv.set_renege_obs(queueid, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, time_local_service, time_to_service_end, reward, decision, self.long_avg_serv_time, uses_nn)
            #else:
            #	self.reneging_rate = self.compute_reneging_rate(self.nn_subscribers)
			#	self.jockeying_rate = self.compute_jockeying_rate(self.nn_subscribers)
			#	self.objObserv.set_renege_obs(queueid, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, time_local_service, time_to_service_end, reward, decision, self.long_avg_serv_time, uses_nn)                         
            
            #self.objObserv.set_jockey_obs(curr_queue_id, curr_pose, self.queue_intensity, self.jockeying_rate, self.reneging_rate, exp_delay, req.exp_time_service_end, reward, decision, self.long_avg_serv_time, uses_intensity_based) # time_alt_queue                                
                                               
            self.curr_req = req                    
        
        return reward
        
    
    def get_current_jockey_observations(self):
		
        #print("OBS -> ", self.objObserv.get_jockey_obs())
        return self.objObserv.get_jockey_obs()


    def makeJockeyingDecision(self, req, curr_queue_id, alt_queue_id, next_state, curr_queue_state,  serv_rate): # req, curr_queue_id, alt_queue_id, next_state, curr_queue_state,  serv_rate
                
        decision=False  
        found = False
        
        if "1" in curr_queue_id:
            serv_rate = self.srvrates_1
        else:  
            serv_rate = self.srvrates_2               
        
        curr_queue = self.dict_queues_obj.get(curr_queue_id)
        dest_queue = self.dict_queues_obj.get(alt_queue_id)
        curr_pose = self.get_request_position( curr_queue_id, req.customerid)       
        queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate                
        
        self.avg_delay = self.calculate_max_cloud_delay(len(dest_queue)+1, queue_intensity, req) # self.calculate_max_cloud_delay(curr_pose, serv_rate) 
        # self.generateExpectedJockeyCloudDelay ( req, alt_queue_id)                 
        # curr_pose = req.pos_in_queue # self.get_request_position(curr_queue_id, customerid)
        
        if curr_pose is None:
            # print(f" CurrPose is None -> Request ID {req.customerid} not found in queue {curr_queue_id}. Continuing with processing...")
            return
            
        else: 
            # print(f" Found request at {curr_pose} ")                               
            found = True
            remaining_wait_time = self.get_remaining_time(curr_pose-1, serv_rate) # self.get_remaining_time(curr_queue_id, curr_pose) # pos) #
            # time_already_spent_in_curr_queue = self.time - req.time_entrance
            
            # The time expected to spent in the other queue is what has already been 
            # spent plus the new time when the jockey lands at a particular position in the other queue
            jockey_total_expected_time_to_service = remaining_wait_time + self.avg_delay # remaining_wait_time
             
            # print("\n --- jockey expected", jockey_total_expected_time_to_service, " --- until service ",remaining_wait_time)
            # if remaining_wait_time > jockey_total_expected_time_to_service: # self.avg_delay:
            # margin = 0.5  # or any value you want
            if remaining_wait_time > self.avg_delay: #jockey_total_expected_time_to_service - margin: # or random.random() < 0.1:                
                decision = True
                diff_wait = remaining_wait_time - self.avg_delay # jockey_total_expected_time_to_service
                reward = self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, remaining_wait_time, self.avg_delay, decision, curr_pose, curr_queue, req.uses_nn, diff_wait)

                if not found:
                    # print(f"Request ID {req.customerid} not found in queue {curr_queue_id}. Continuing with processing...")
                    return False
                    
    
    def compute_reneging_rate_by_info_source(self, info_src_requests):
        """Compute the reneging rate for a specific info source."""
        renegs = len(info_src_requests)  
        return renegs / len(self.get_current_renege_count()) 
        
      
    def compute_jockeying_rate_by_info_source(self, info_src_requests):
        """Compute the reneging rate for a specific info source."""
        jockeys = len(info_src_requests) 
        return jockeys / len(self.get_current_jockey_observations())
    
        
    def plot_reneging_time_vs_queue_length(self):
		
        # print("\n **** ", self.curr_obs_renege)
        nn_x, nn_y = [], []
        state_x, state_y = [], []
        
        for obs in self.curr_obs_renege:  
            if not isinstance(obs, dict):
                continue
            req = obs.get("Request")
            if req is None:
                continue
            
            #req = obs["Request"]
            # Ensure necessary fields are present and valid
            time_entrance = getattr(req, 'time_entrance', None)
            time_exit = getattr(req, 'time_exit', None)
            pos_in_queue = obs.get("at_pose")
            uses_nn = obs.get("intensity_based_info")
            if time_exit is not None and time_entrance is not None and pos_in_queue is not None:
                waiting_time = time_exit - time_entrance
                if uses_nn:
                    nn_x.append(pos_in_queue)
                    nn_y.append(waiting_time)
                else:
                    state_x.append(pos_in_queue)
                    state_y.append(waiting_time)
        # Sort for line plot
        if nn_x and nn_y:
            nn_sorted = sorted(zip(nn_x, nn_y))
            nn_x, nn_y = zip(*nn_sorted)
        if state_x and state_y:
            state_sorted = sorted(zip(state_x, state_y))
            state_x, state_y = zip(*state_sorted)
    
        plt.figure(figsize=(10, 6))
        if nn_x:
            plt.plot(nn_x, nn_y, '-o', color='orange', label='NN-based Reneged')
        if state_x:
            plt.plot(state_x, state_y, '-o', color='blue', label='State-based Reneged')
        plt.xlabel("Queue Length on Arrival")
        plt.ylabel("Waiting Time")
        plt.title("Reneging Requests: Waiting Time vs Queue Length")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        
    def plot_jockeying_time_vs_queue_length(self):
        nn_x, nn_y = [], []
        state_x, state_y = [], []
        
        for obs in self.curr_obs_jockey:
            if not isinstance(obs, dict):
                continue
            req = obs.get("Request")
            if req is None:
                continue
            #req = obs["Request"]
            time_entrance = getattr(req, 'time_entrance', None)
            time_exit = getattr(req, 'time_exit', None)
            pos_in_queue = obs.get("at_pose")
            uses_nn = obs.get("intensity_based_info")
            if time_exit is not None and time_entrance is not None and pos_in_queue is not None:
                waiting_time = time_exit - time_entrance
                if uses_nn:
                    nn_x.append(pos_in_queue)
                    nn_y.append(waiting_time)
                else:
                    state_x.append(pos_in_queue)
                    state_y.append(waiting_time)
        # Sort for line plot
        if nn_x and nn_y:
            nn_sorted = sorted(zip(nn_x, nn_y))
            nn_x, nn_y = zip(*nn_sorted)
        if state_x and state_y:
            state_sorted = sorted(zip(state_x, state_y))
            state_x, state_y = zip(*state_sorted)
    
        plt.figure(figsize=(10, 6))
        if nn_x:
            plt.plot(nn_x, nn_y, '-o', color='orange', label='NN-based Jockeyed')
        if state_x:
            plt.plot(state_x, state_y, '-o', color='blue', label='State-based Jockeyed')
        plt.xlabel("Queue Length on Arrival")
        plt.ylabel("Waiting Time")
        plt.title("Jockeying Requests: Waiting Time vs Queue Length")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    def compare_behavior_rates_by_information_how_often(self):
        """
        Compare the percentage of jockeyed versus reneged requests based on
        intensity-based information versus departure-based information over all episodes.
        """
        # Separate observations based on decision source
        jockeyed_intensity_based_requests = [d for d in self.objObserv.get_jockey_obs() if d.get("intensity_based_info", False)]
        jockeyed_departure_based_requests = [d for d in self.objObserv.get_jockey_obs() if not d.get("intensity_based_info", False)]
    
        reneged_intensity_based_requests = [d for d in self.objObserv.get_renege_obs() if d.get("intensity_based_info", False)]
        reneged_departure_based_requests = [d for d in self.objObserv.get_renege_obs() if not d.get("intensity_based_info", False)]

        # Compute total counts for each category
        total_jockeyed_intensity = len(jockeyed_intensity_based_requests)
        total_jockeyed_departure = len(jockeyed_departure_based_requests)
        total_reneged_intensity = len(reneged_intensity_based_requests)
        total_reneged_departure = len(reneged_departure_based_requests)

        # Compute percentages
        total_jockeyed = total_jockeyed_intensity + total_jockeyed_departure
        total_reneged = total_reneged_intensity + total_reneged_departure

        jockeyed_intensity_percent = (total_jockeyed_intensity / total_jockeyed) * 100 if total_jockeyed > 0 else 0
        jockeyed_departure_percent = (total_jockeyed_departure / total_jockeyed) * 100 if total_jockeyed > 0 else 0
        reneged_intensity_percent = (total_reneged_intensity / total_reneged) * 100 if total_reneged > 0 else 0
        reneged_departure_percent = (total_reneged_departure / total_reneged) * 100 if total_reneged > 0 else 0

        # Create a bar chart
        categories = ["Jockeyed", "Reneged"]
        intensity_based_percents = [jockeyed_intensity_percent, reneged_intensity_percent]
        departure_based_percents = [jockeyed_departure_percent, reneged_departure_percent]

        x = range(len(categories))  # Indices for categories

        plt.figure(figsize=(10, 6))
        plt.bar(x, intensity_based_percents, width=0.4, label="Intensity-Based", color="blue", align="center")
        plt.bar(
            [i + 0.4 for i in x], departure_based_percents, width=0.4, label="Departure-Based", color="orange", align="center"
        )

        # Add labels and titles
        plt.xlabel("Categories")
        plt.ylabel("Percentage")
        #plt.title("Percentage of Jockeyed vs Reneged Requests by Decision Source")
        plt.xticks([i + 0.2 for i in x], categories)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7) 
        plt.tight_layout()

        # Display the plot
        plt.show()
        

    def compare_rates_by_information_source_with_graph(self):
        """
        Compare reneging and jockeying rates for intensity-based and departure-based requests,
        and plot the comparison as a bar chart.
        """
        # Separate requests based on their information source
        intensity_based_requests = [req for req in self.state_subscribers if req.uses_nn]  # Example flag
        departure_based_requests = [req for req in self.state_subscribers if not req.uses_nn]

        # Calculate rates for intensity-based requests
        reneging_rate_intensity = self.compute_reneging_rate(intensity_based_requests)
        jockeying_rate_intensity = self.compute_jockeying_rate(intensity_based_requests)

        # Calculate rates for departure-based requests
        reneging_rate_departure = self.compute_reneging_rate(departure_based_requests)
        jockeying_rate_departure = self.compute_jockeying_rate(departure_based_requests)

        # Plot comparison
        categories = ['Reneging Rate', 'Jockeying Rate']
        intensity_rates = [reneging_rate_intensity, jockeying_rate_intensity]
        departure_rates = [reneging_rate_departure, jockeying_rate_departure]

        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(categories, intensity_rates, marker='o', label='Intensity-Based', color='blue', linestyle='-')
        plt.plot(categories, departure_rates, marker='o', label='Departure-Based', color='orange', linestyle='--')

        # Add labels and title
        plt.xlabel("Categories")
        plt.ylabel("Rate")
        #plt.title("Comparison of Reneging and Jockeying Rates")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Show the plot
        plt.show()
        
        
    def set_equal_service_rates(self, rate):
        self.dict_servers_info["1"] = rate
        self.dict_servers_info["2"] = rate
        
        
    import numpy as np

    def compute_success_table(self, time_limit=10.0):
        """
        Compute success rates and standard errors for jockeying and reneging, per server and info source,
        and format as LaTeX table rows.
        """
        # Helper
        def compute_stats(obs_list, info_flag, server_id):
            count = 0
            success = 0
            for obs in obs_list:
                req = obs.get("Request")
                if req is None:
                    continue
                uses_nn = obs.get("intensity_based_info", False)
                srv = str(obs.get("ServerID"))
                if (uses_nn == info_flag) and (srv == server_id):
                    time_entrance = getattr(req, 'time_entrance', None)
                    time_exit = getattr(req, 'time_exit', None)
                    if time_entrance is not None and time_exit is not None:
                        count += 1
                        duration = time_exit - time_entrance
                        if duration <= time_limit:
                            success += 1
            p = success / count if count > 0 else 0.0
            se = np.sqrt(p * (1 - p) / count) if count > 0 else 0.0
            return p, se, count

        jockey_obs = self.objObserv.get_jockey_obs()
        renege_obs = self.objObserv.get_renege_obs()

        # Prepare result rows
        rows = []
        for server_id, queue_label in zip(["1", "2"], ["Srv1", "Srv2"]):
            # Markov (False), then NN (True)
            for info_flag, info_label in zip([False, True], ["Markov", "NN"]):
                p_renege, se_renege, n_renege = compute_stats(renege_obs, info_flag, server_id)
                p_jockey, se_jockey, n_jockey = compute_stats(jockey_obs, info_flag, server_id)
                n = max(n_renege, n_jockey)
                rows.append((
                    queue_label, info_label,
                    f"${p_renege:.3f}\\pm{se_renege:.4f}$",
                    f"${p_jockey:.4f}\\pm{se_jockey:.4f}$",
                    f"{n}"
                ))

        # Format for LaTeX table
        print("\\begin{tabular}{llccc}")
        print("\\toprule")
        print("Queue & Info & Reneging $p\\pm$SE & Jockey $p\\pm$SE & $N$ \\\\")
        print("\\midrule")
        for r in rows:
            print("    {} & {} & {} & {} & {} \\\\".format(*r))
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\vspace{2pt}")
        print("\\footnotesize")
        print("Binomial standard error (SE) computed as $\\sqrt{p(1-p)/N}$.")
        
        
    def plot_reneging_and_jockeying_rates_vs_queue_size_by_type(self):
        """
        Plots the reneging and jockeying rates as queue sizes grow for raw-state and NN-based subscribers.
        """
   

        types = [("Raw Markov", self.state_subscribers), ("Actor-Critic (NN)", self.nn_subscribers)]
        rate_results = {}

        for label, subscribers in types:
            reneging_counts = []
            jockeying_counts = []
            queue_sizes = []

            # Step through time (or just end state if you don't track queue size progression)
            # If you want to plot vs. queue size, use range(len(subscribers)), or store sizes as requests are added
            for i in range(1, len(subscribers) + 1):
                current_subs = subscribers[:i]
                queue_sizes.append(i)
                # Reneging and jockeying up to this point
                num_reneged = sum(1 for req in current_subs if getattr(req, 'reneged', False))
                num_jockeyed = sum(1 for req in current_subs if getattr(req, 'jockeyed', False))
                reneging_counts.append(num_reneged / i)
                jockeying_counts.append(num_jockeyed / i)

            rate_results[label] = (queue_sizes, reneging_counts, jockeying_counts)

        plt.figure(figsize=(12, 6))
        for label, (queue_sizes, reneging_rates, jockeying_rates) in rate_results.items():
            plt.plot(queue_sizes, reneging_rates, '-o', label=f'{label} Reneging Rate', markersize=1)
            plt.plot(queue_sizes, jockeying_rates, '-x', label=f'{label} Jockeying Rate', markersize=1)

        plt.xlabel("Queue Size (Number of Requests Entered)")
        plt.ylabel("Rate")
        plt.title("Reneging and Jockeying Rates vs Queue Size by Information Type")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_jockeying_reneging_rates_vs_waiting_time_difference(self, bin_width=0.2):
        # Convert lists to dicts using id(Request) as unique key
        jockey_events = {}
        for obs in self.objObserv.get_jockey_obs():
            req = obs.get("Request")
            if req is not None:
                jockey_events[id(req)] = obs

        renege_events = {}
        for obs in self.objObserv.get_renege_obs():
            req = obs.get("Request")
            if req is not None:
                renege_events[id(req)] = obs

        jockey_x, jockey_y = [], []
        renege_x, renege_y = [], []

        # Jockeying events
        for obs in jockey_events.values():
            req = obs.get("Request")
            #time_entrance = getattr(req, 'time_entrance', None)
            #time_exit = getattr(req, 'time_exit', None)
            #expected_service_time = obs.get("expected_service_time")
            diff_wait = obs.get("waiting_time_diff")
            jockeying_rate = obs.get("rate_jockeyed")
            if None not in (diff_wait, jockeying_rate): # time_entrance, time_exit, expected_service_time,
                #waiting_time = time_exit - time_entrance
                #diff = waiting_time - expected_service_time
                jockey_x.append(diff_wait)
                jockey_y.append(jockeying_rate)

        # Reneging events
        for obs in renege_events.values():
            req = obs.get("Request")
            #time_entrance = getattr(req, 'time_entrance', None)
            #time_exit = getattr(req, 'time_exit', None)
            #expected_service_time = obs.get("expected_service_time")
            diff_wait = obs.get("waiting_time_diff")
            reneging_rate = obs.get("rate_reneged")
            if None not in (diff_wait, reneging_rate): # time_entrance, time_exit, expected_service_time,
                #waiting_time = time_exit - time_entrance
                #diff = waiting_time - expected_service_time
                renege_x.append(diff_wait)
                renege_y.append(reneging_rate)

        # Bin and average to remove zigzagging
        def bin_and_average(x, y, bin_width):
            if not x:
                return [], []
            bins = np.arange(min(x), max(x) + bin_width, bin_width)
            bin_indices = np.digitize(x, bins)
            bin_sums = defaultdict(float)
            bin_counts = defaultdict(int)
            for xi, yi, bi in zip(x, y, bin_indices):
                bin_sums[bi] += yi
                bin_counts[bi] += 1
            bin_means_x = []
            bin_means_y = []
            for bi in sorted(bin_sums):
                bin_center = bins[bi-1] if bi-1 < len(bins) else bins[-1]
                bin_means_x.append(bin_center)
                bin_means_y.append(bin_sums[bi] / bin_counts[bi])
            return bin_means_x, bin_means_y

        jockey_x_smooth, jockey_y_smooth = bin_and_average(jockey_x, jockey_y, bin_width)
        renege_x_smooth, renege_y_smooth = bin_and_average(renege_x, renege_y, bin_width)

        plt.figure(figsize=(10, 6))
        if jockey_x_smooth and jockey_y_smooth:
            plt.plot(jockey_x_smooth, jockey_y_smooth, color='orange', label='Jockeying Rate', marker='o')
        if renege_x_smooth and renege_y_smooth:
            plt.plot(renege_x_smooth, renege_y_smooth, color='blue', label='Reneging Rate', marker='x')

        plt.xlabel('Waiting Time - Expected Service Time (at event)')
        plt.ylabel('Event Rate (Jockeying/Reneging)')
        plt.title('Jockeying and Reneging Rates vs Waiting Time Difference')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
    
    def plot_asymptotic_behavior_of_jockeying_and_reneging(self, max_queue_length=100, bin_width=2):
        """
        Plot the probability of beneficial jockeying and reneging as queue length increases,
        for both NN-based and Markov information sources. Demonstrates asymptotic convergence
        to extrema (0 or 1) and negligible difference between information sources.
        """

        # Gather events from observations
        jockey_obs = self.objObserv.get_jockey_obs()
        renege_obs = self.objObserv.get_renege_obs()

        # Bin queue lengths
        bins = np.arange(1, max_queue_length+bin_width, bin_width)
        
        def bin_by_length(obs_list, reward_key='reward'):
            count_nn = np.zeros(len(bins))
            total_nn = np.zeros(len(bins))
            count_state = np.zeros(len(bins))
            total_state = np.zeros(len(bins))
            for obs in obs_list:
                if not isinstance(obs, dict):
                    continue
                pos = obs.get("at_pose")
                uses_nn = obs.get("intensity_based_info", False)
                reward = obs.get(reward_key, 0)
                if pos is None:
                    continue
                # Find bin index
                idx = np.digitize(pos, bins) - 1
                if idx < 0 or idx >= len(bins):
                    continue
                if uses_nn:
                    total_nn[idx] += 1
                    if reward > 0:
                        count_nn[idx] += 1
                else:
                    total_state[idx] += 1
                    if reward > 0:
                        count_state[idx] += 1
            # Probability per bin
            prob_nn = np.divide(count_nn, total_nn, out=np.zeros_like(count_nn), where=total_nn>0)
            prob_state = np.divide(count_state, total_state, out=np.zeros_like(count_state), where=total_state>0)
            return prob_nn, prob_state

        # Compute probabilities for jockeying and reneging
        jockey_nn, jockey_state = bin_by_length(jockey_obs)
        renege_nn, renege_state = bin_by_length(renege_obs)

        plt.figure(figsize=(12, 6))
        plt.plot(bins, jockey_nn, '-o', color='orange', label='Jockeying (NN-based)',markersize=1)
        plt.plot(bins, jockey_state, '-o', color='red', label='Jockeying (Markov-based)', markersize=1)
        plt.plot(bins, renege_nn, '-x', color='blue', label='Reneging (NN-based)', markersize=1)
        plt.plot(bins, renege_state, '-x', color='green', label='Reneging (Markov-based)', markersize=1)
        plt.xlabel("Queue Length")
        plt.ylabel("Probability of Beneficial Action")
        plt.title("Asymptotic Behavior of Jockeying and Reneging vs Queue Length")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Validation printout for theorem
        print("Asymptotic regime validation: For large queue lengths, probabilities approach extrema (0 or 1), and curves for NN-based and Markov-based decisions become indistinguishable.")
        
        
    def plot_decision_probability_vs_queue_length(self, bin_width=1, min_count_per_n=1):
        """
        Improved: Plot Pr{estimator declares switch} vs queue length n for Markov, actor-critic, and oracle.
        Only plot bins with at least min_count_per_bin observations.
        """

        obs_jockey = self.objObserv.get_jockey_obs()
        # Gather all observed queue lengths
        queue_lengths = [obs.get("at_pose") for obs in obs_jockey if isinstance(obs, dict) and obs.get("at_pose") is not None]
        unique_ns = sorted(set(queue_lengths))
        print("Observed queue lengths (n):", unique_ns)

        # Prepare data per queue length
        stats = { "Markov": {}, "NN": {}, "Oracle": {} }
        for n in unique_ns:
            stats["Markov"][n] = {"switch": 0, "total": 0}
            stats["NN"][n] = {"switch": 0, "total": 0}
            stats["Oracle"][n] = {"switch": 0, "total": 0}

        def oracle_decision(obs):
            req = obs.get("Request")
            if req is None:
                return False
            waiting_time = getattr(req, 'time_exit', 0) - getattr(req, 'time_entrance', 0)
            try:
                true_switch = waiting_time > obs.get("expected_service_time", 0)
            except Exception:
                true_switch = False
            return true_switch

        for obs in obs_jockey:
            if not isinstance(obs, dict):
                continue
            n = obs.get("at_pose")
            if n is None:
                continue
            uses_nn = obs.get("intensity_based_info", False)
            declared_switch = bool(obs.get("action", 0))
            oracle_switch = oracle_decision(obs)
            if uses_nn:
                stats["NN"][n]["total"] += 1
                if declared_switch:
                    stats["NN"][n]["switch"] += 1
            else:
                stats["Markov"][n]["total"] += 1
                if declared_switch:
                    stats["Markov"][n]["switch"] += 1
            stats["Oracle"][n]["total"] += 1
            if oracle_switch:
                stats["Oracle"][n]["switch"] += 1

        # Prepare data for plotting: only n values with enough data
        plot_ns = []
        markov_probs = []
        nn_probs = []
        oracle_probs = []
        for n in unique_ns:
            t_markov = stats["Markov"][n]["total"]
            t_nn = stats["NN"][n]["total"]
            t_oracle = stats["Oracle"][n]["total"]
            if (t_markov + t_nn + t_oracle) >= min_count_per_n:
                plot_ns.append(n)
                markov_probs.append(stats["Markov"][n]["switch"] / t_markov if t_markov else np.nan)
                nn_probs.append(stats["NN"][n]["switch"] / t_nn if t_nn else np.nan)
                oracle_probs.append(stats["Oracle"][n]["switch"] / t_oracle if t_oracle else np.nan)

        plt.figure(figsize=(10,6))
        plt.plot(plot_ns, markov_probs, '-o', label="Markov estimator")
        plt.plot(plot_ns, nn_probs, '-x', label="Actor-Critic estimator")
        plt.plot(plot_ns, oracle_probs, '--s', label="Oracle (true)")
        plt.xlabel("Queue Length (n)")
        plt.ylabel("Pr{Declares Switch}")
        plt.title("Decision Probability vs Queue Length (integer n)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

    def plot_diagnostics_error_and_misclassification(self, min_count_per_n=1):
        """
        Plot normalized estimation error and misclassification rate vs actual integer queue length n for each model.
        Only plot queue lengths with at least min_count_per_n observations.
        """


        obs_jockey = self.objObserv.get_jockey_obs()
        #for obs in obs_jockey:
        #    print("\n ************ ", obs)
            
        queue_lengths = [obs.get("at_pose") for obs in obs_jockey if isinstance(obs, dict) and obs.get("at_pose") is not None]
        unique_ns = sorted(set(queue_lengths))
        print("Observed queue lengths (n):", unique_ns)

        error_markov = {n: [] for n in unique_ns}
        error_nn = {n: [] for n in unique_ns}
        misclass_markov = {n: [] for n in unique_ns}
        misclass_nn = {n: [] for n in unique_ns}

        def oracle_decision(obs):
            req = obs.get("Request")
            if req is None:
                return False
            waiting_time = getattr(req, 'time_exit', 0) - getattr(req, 'time_entrance', 0)
            try:
                true_switch = waiting_time > obs.get("expected_service_time", 0)
            except Exception:
                true_switch = False
            return true_switch

        for obs in obs_jockey:
            if not isinstance(obs, dict):
                continue
            n = obs.get("at_pose")
            uses_nn = obs.get("intensity_based_info", False)
            req = obs.get("Request")
            estimator_wait = obs.get("expected_service_time", 0)
            true_wait = getattr(req, 'time_exit', 0) - getattr(req, 'time_entrance', 0)
            declared_switch = bool(obs.get("action"))
            oracle_switch = oracle_decision(obs)
            if n is None or true_wait is None:
                continue
            norm_error = abs(estimator_wait - true_wait) / n if n > 0 else 0
            misclass = int(declared_switch != oracle_switch)
            if uses_nn:
                error_nn[n].append(norm_error)
                misclass_nn[n].append(misclass)
            else:
                error_markov[n].append(norm_error)
                misclass_markov[n].append(misclass)

        # Prepare data for plotting
        plot_ns = []
        err_markov_plot = []
        err_markov_std_plot = []
        err_nn_plot = []
        err_nn_std_plot = []
        misclass_markov_plot = []
        misclass_nn_plot = []

        for n in unique_ns:
            # Only plot if enough data
            if len(error_markov[n]) + len(error_nn[n]) >= min_count_per_n:
                plot_ns.append(n)
                err_markov_plot.append(np.mean(error_markov[n]) if error_markov[n] else np.nan)
                err_markov_std_plot.append(np.std(error_markov[n]) if error_markov[n] else np.nan)
                err_nn_plot.append(np.mean(error_nn[n]) if error_nn[n] else np.nan)
                err_nn_std_plot.append(np.std(error_nn[n]) if error_nn[n] else np.nan)
                misclass_markov_plot.append(np.mean(misclass_markov[n]) if misclass_markov[n] else np.nan)
                misclass_nn_plot.append(np.mean(misclass_nn[n]) if misclass_nn[n] else np.nan)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        axs[0].errorbar(plot_ns, err_markov_plot, yerr=err_markov_std_plot, fmt='-o', label="Markov")
        axs[0].errorbar(plot_ns, err_nn_plot, yerr=err_nn_std_plot, fmt='-x', label="Actor-Critic")
        axs[0].set_xlabel("Queue Length (n)")
        axs[0].set_ylabel("Normalized Error: E[|cWi(n)-Wi(n)|/n]")
        axs[0].set_title("Normalized Estimation Error vs n")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(plot_ns, misclass_markov_plot, '-o', label="Markov")
        axs[1].plot(plot_ns, misclass_nn_plot, '-x', label="Actor-Critic")
        axs[1].set_xlabel("Queue Length (n)")
        axs[1].set_ylabel("Misclassification Rate")
        axs[1].set_title("Misclassification Rate vs n")
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()
        
        
    def plot_successful_jockey_probability_vs_queue_length(self, T=None, min_count_per_n=1):
        """
        Plot Pr{switch & complete before T} vs actual integer queue length n for Markov, NN, and Oracle.
        Only plot queue lengths with at least min_count_per_n observations.
        """

        obs_jockey = self.objObserv.get_jockey_obs()
        queue_lengths = [obs.get("at_pose") for obs in obs_jockey if isinstance(obs, dict) and obs.get("at_pose") is not None]
        unique_ns = sorted(set(queue_lengths))
        print("Observed queue lengths (n):", unique_ns)

        stats = { "Markov": {}, "NN": {}, "Oracle": {} }
        for n in unique_ns:
            stats["Markov"][n] = {"success": 0, "total": 0}
            stats["NN"][n] = {"success": 0, "total": 0}
            stats["Oracle"][n] = {"success": 0, "total": 0}

        def oracle_decision(obs, T):
            req = obs.get("Request")
            if req is None:
                return False
            time_entrance = getattr(req, 'time_entrance', None)
            time_exit = getattr(req, 'time_exit', None)
            completion_time = time_exit - time_entrance if time_exit is not None and time_entrance is not None else None
            try:
                true_switch = completion_time > obs.get("expected_service_time", 0)
            except Exception:
                true_switch = False
            threshold_T = T if T is not None else obs.get("expected_service_time", None)
            success_oracle = true_switch and (completion_time is not None and threshold_T is not None and completion_time < threshold_T)
            return success_oracle

        for obs in obs_jockey:
            if not isinstance(obs, dict):
                continue
            n = obs.get("at_pose")
            if n is None:
                continue
            uses_nn = obs.get("intensity_based_info", False)
            declared_switch = bool(obs.get("action", 0))
            req = obs.get("Request")
            time_entrance = getattr(req, 'time_entrance', None)
            time_exit = getattr(req, 'time_exit', None)
            completion_time = time_exit - time_entrance if time_exit is not None and time_entrance is not None else None
            threshold_T = T if T is not None else obs.get("expected_service_time", None)
            success = declared_switch and (completion_time is not None and threshold_T is not None and completion_time < threshold_T)
            success_oracle = oracle_decision(obs, T)
            if uses_nn:
                stats["NN"][n]["total"] += 1
                if success:
                    stats["NN"][n]["success"] += 1
            else:
                stats["Markov"][n]["total"] += 1
                if success:
                    stats["Markov"][n]["success"] += 1
            stats["Oracle"][n]["total"] += 1
            if success_oracle:
                stats["Oracle"][n]["success"] += 1

        plot_ns = []
        markov_probs = []
        nn_probs = []
        oracle_probs = []
        for n in unique_ns:
            t_markov = stats["Markov"][n]["total"]
            t_nn = stats["NN"][n]["total"]
            t_oracle = stats["Oracle"][n]["total"]
            if (t_markov + t_nn + t_oracle) >= min_count_per_n:
                plot_ns.append(n)
                markov_probs.append(stats["Markov"][n]["success"] / t_markov if t_markov else np.nan)
                nn_probs.append(stats["NN"][n]["success"] / t_nn if t_nn else np.nan)
                oracle_probs.append(stats["Oracle"][n]["success"] / t_oracle if t_oracle else np.nan)

        plt.figure(figsize=(10,6))
        plt.plot(plot_ns, markov_probs, '-o', label="Markov estimator")
        plt.plot(plot_ns, nn_probs, '-x', label="Actor-Critic estimator")
        plt.plot(plot_ns, oracle_probs, '--s', label="Oracle (true)")
        plt.xlabel("Queue Length (n)")
        plt.ylabel("Pr{Switch & Complete before T}")
        plt.title("Successful Jockey Probability vs Queue Length (integer n)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    
    def run_equal_service_rate_experiment(self, arrival_rates_even, duration, env, num_episodes=10):
        """
        For each value in arrival_rates_even, set μ₁ = μ₂ = λ/2 and run the simulation, plotting results.
        """
        for arrival_rate in arrival_rates_even:
            service_rate = arrival_rate / 2.0
            self.set_equal_service_rates(service_rate)
            self.objQueues.sampled_arr_rate = arrival_rate  # Update the sampled arrival rate in Queues
            print(f"\nRunning for arrival_rate={arrival_rate}, service_rate={service_rate}")
            self.run(
                duration,
                env,
                adjust_service_rate=False,
                num_episodes=num_episodes,
                save_to_file=f"equal_service_rate_metrics_lambda_{arrival_rate}.csv",
                arrival_rates_even=arrival_rates_even
            )
            self.plot_rates()
            # plot_reneging_and_jockeying_rates_vs_queue_size()
            plot_reneging_and_jockeying_rates_vs_queue_size_by_type()
            #self.plot_reneging_time_vs_queue_length()
            #self.plot_jockeying_time_vs_queue_length()
            # self.plot_queue_intensity_vs_requests()
            
            
    def compute_success_table_with_max_local_utility(self):
        """
        Computes the LaTeX table for success rates using the maximum value of generateLocalCompUtility as the time limit.
        """
        # Gather all generateLocalCompUtility values for requests in jockey and renege observations
        jockey_obs = self.objObserv.get_jockey_obs()
        renege_obs = self.objObserv.get_renege_obs()
        all_obs = jockey_obs + renege_obs

        comp_utils = []
        for obs in all_obs:
            req = obs.get("Request")
            if req is not None:
                try:
                    val = self.generateLocalCompUtility(req)
                    comp_utils.append(val)
                except Exception:
                    continue  # If for any reason utility cannot be computed, skip

        if not comp_utils:
            print("No generateLocalCompUtility values found.")
            return

        time_limit = max(comp_utils)
        print(f"Using max(generateLocalCompUtility) as time_limit: {time_limit:.4f}")

        # Now call your existing success table function with this time limit
        self.compute_success_table(time_limit=time_limit)
       

class ImpatientTenantEnv:
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.queue_state = {}
        self.action = ""  
        self.utility_basic = 1.0
        self.discount_coef = 0.1
        self.history = {}
        self.objQueues = Queues()
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
        
        self.current_step = 0
        self.max_steps = 1000  # Or set from outside, or pass as parameter
                
        self.action_space = 2  # Number of discrete actions
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
            "intensity_based_info": (True, False)
        }
        # Reneged
        
        self.state_dim = len(self.observation_space)
        self.action_dim = self.action_space
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim)
        self.agent = A2CAgent(self.state_dim, self.action_dim)     
        # self.requestObj = RequestQueue(self.utility_basic, self.discount_coef)
        self.requestObj = RequestQueue(self.state_dim, self.action_dim, self.utility_basic, self.discount_coef, self.actor_critic, self.agent)       
        self.queue_id = self.requestObj.get_curr_queue_id()  
        arr_rate = self.objQueues.get_arrivals_rates()    
        
        self.initialize_env_states()                               
        self._action_to_state = {}
        #    Actions.RENEGE.value: self.get_renege_action_outcome(self.queue_id), 
        #    Actions.JOCKEY.value: self.get_jockey_action_outcome(self.queue_id)
        #}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    
    def initialize_env_states(self):
		
        self.queue_state = {
            "ServerID": "1",
            "rate_jockeyed": 0.0,            
            "this_busy": 0.0,
            "expected_service_time": 0.0,
            "time_service_took": 0.0,
            "rate_reneged": 0.0,
            "reward": 0,
            "at_pose": 0,
            "long_avg_serv_time": 0.0,
            "action": "served",
            "intensity_based_info": False
        }
        
        #return self.queue_state
    
    
    def update_queue_state(self, next_queue_state): #queue_id, curr_queue, dest_queue, queue_sizes, service_rates, total_reward, action):
        """
        Updates the `self.queue_state` variable with the current state of the queue and environment.

        Args:
            queue_id (str): The ID of the current queue.
            curr_queue (list): The current queue's state.
            dest_queue (list): The destination queue's state (if applicable).
            queue_sizes (tuple): A tuple containing the sizes of the queues.
            service_rates (tuple): A tuple containing the service rates of the servers.
            total_reward (float): The total accumulated reward.
            action (str): The last action taken in the environment.
        """
        self.queue_state = next_queue_state
        
        # print("Queue state updated:", self.queue_state)
         
    
    def _get_obs(self):
        obs = {key: np.zeros(1) for key in self.observation_space.keys()}
        
        return obs
        

    def _get_info(self):
		
        return self._action_to_state 
        

    def reset(self, seed=None, options=None):
        random.seed(seed)
        np.random.seed(seed)
        self.current_step = 0
        observation = [0.0] * self.state_dim 
        info = self._get_info()     
                                    
        return observation, info


    def get_renege_action_outcome_original(self, queue_id):
        """
        Compute the state of the queue after a renege action.

        Args:
            queue_id (str): The ID of the queue where the renege action occurs.

        Returns:
            dict: The resulting state of the queue.
        """
        srv = self.queue_state.get('ServerID')  # Get the current server ID from the queue state
        
        # Check if the current server has any reneged requests
        if "1" in srv:  # Server 1
            #print("\n --> ID ", srv, type(srv))
            if len(self.requestObj.get_current_renege_count()) > 0:
                # Retrieve the latest reneged observation
                last_renege = self.requestObj.get_current_renege_count()[-1]
                #print("\n LAST RENEGE EVENT: ", last_renege)
                self.queue_state['at_pose'] = max(0, last_renege['at_pose'] - 1)  # Decrease the position in the queue
                self.queue_state["reward"] = last_renege["reward"]  # Update the reward based on the latest renege
            elif len(self.requestObj.get_history(queue_id)) > 0:
                # Fall back to the last known state in the history if no reneged requests exist
                self.queue_state = self.requestObj.get_history(queue_id)[-1]
            else:
                # Default state if no renege or history exists
                self.queue_state['at_pose'] = 0
                self.queue_state["reward"] = 0.0

        elif "2" in srv:  # Server 2
            #print("\n --> ID ", srv, type(srv))
            if len(self.requestObj.get_current_renege_count()) > 0:
                # Retrieve the latest reneged observation
                last_renege = self.requestObj.get_current_renege_count()[-1]
                #print("\n LAST RENEGE EVENT: ", last_renege)
                self.queue_state['at_pose'] = max(0, last_renege['at_pose'] - 1)  # Decrease the position in the queue
                self.queue_state["reward"] = last_renege["reward"]  # Update the reward based on the latest renege
            elif len(self.requestObj.get_history(queue_id)) > 0:
                # Fall back to the last known state in the history if no reneged requests exist
                self.queue_state = self.requestObj.get_history(queue_id)[-1]
            else:
                # Default state if no renege or history exists
                self.queue_state['at_pose'] = 0
                self.queue_state["reward"] = 0.0

        else:
            # Handle unexpected server IDs
            raise ValueError(f"Invalid Server ID: {srv}")

        # Log the resulting state and return it
        print(f"New state after renege action on Server {srv}: {self.queue_state}")
        return self.queue_state
    
    
    def get_renege_action_outcome(self, queue_id):
        """
        Compute the state of the queue after a renege action.

        Args:
            queue_id (str): The ID of the queue where the renege action occurs.

        Returns:
            dict: The resulting state of the queue.
        """
        
        srv = self.queue_state.get('ServerID') # curr_state.get('ServerID')
        
        if srv == 1:
            
            if len(self.requestObj.get_current_renege_count()) > 0: #len(self.Observations.get_renege_obs()) > 0: 
                last_renege = self.requestObj.get_current_renege_count()[-1]
                self.queue_state['at_pose'] = max(0, last_renege['at_pose'] - 1) # self.requestObj.get_current_renege_count()[-1]['at_pose'] - 1 # self.Observations.get_renege_obs()[0]['at_pose'] - 1
                self.queue_state["reward"] = last_renege["reward"] # self.requestObj.get_current_renege_count()[-1]['reward'] # self.Observations.get_renege_obs()[0]['reward']                                
                return self.queue_state
            else:
                
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]                    
                    return self.queue_state
                else:                    
                    return self.queue_state
        else:
            if len(self.requestObj.get_current_renege_count()) > 0: # get_curr_obs_renege(srv)) > 0:
                last_renege = self.requestObj.get_current_renege_count()[-1]
                self.queue_state['at_pose'] = max(0, last_renege['at_pose'] - 1) # int(self.requestObj.get_current_renege_count()[-1]['at_pose']) - 1 #self.Observations.get_renege_obs()[0]['at_pose'] - 1
                self.queue_state["reward"] = last_renege["reward"] #self.requestObj.get_current_renege_count()[-1]['reward'] #self.Observations.get_renege_obs()[0]['reward']
                                
                return self.queue_state
            else:
                
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]                    
                    return self.queue_state
                else:                     
                    return self.queue_state
        

    def get_jockey_action_outcome(self, queue_id):
        """
        Compute the state of the queue after a jockey action.

        Args:
            queue_id (str): The ID of the source queue for the jockey action.

        Returns:
            dict: The resulting state of the queue.
        """
        
        srv = self.queue_state.get('ServerID')  # Current server ID
        if srv == 1:
            if len(self.requestObj.get_current_jockey_observations()) > 0:
                # Access the last jockeying event to calculate expected processing time
                jockey_event = self.requestObj.get_current_jockey_observations()[-1]
                #print("\n LAST JOCKEYED EVENT: ", jockey_event)
                expected_processing_time = jockey_event['expected_service_time']

                # If the previous jockeying event was rewarded, and the expected processing time is favorable
                if jockey_event['reward'] > 0 and expected_processing_time < self.queue_state['long_avg_serv_time']:
                    self.queue_state['at_pose'] += 1  # Jockey to the alternative queue
                    self.queue_state['reward'] = jockey_event['reward']

                    # Update the history to reflect the jockeyed state
                    self.requestObj.srv1_history.append({
                        "ServerID": queue_id,
                        "at_pose": self.queue_state['at_pose'],
                        "rate_jockeyed": self.queue_state['rate_jockeyed'],
                        "rate_reneged": self.queue_state['rate_reneged'],
                        "expected_service_time": expected_processing_time,
                        "this_busy": self.queue_state['this_busy'],
                        "long_avg_serv_time": self.queue_state['long_avg_serv_time'],
                        "time_service_took": expected_processing_time,
                        "reward": jockey_event['reward'],
                        "action": "jockeyed",
                        "intensity_based_info": self.queue_state['intensity_based_info']
                    })
                    return self.queue_state
            else:
                # Default to the last known state if no jockeying occurred
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]
                return self.queue_state
        else:
            if len(self.requestObj.get_current_jockey_observations()) > 0:
                jockey_event = self.requestObj.get_current_jockey_observations()[-1]
                #print("\n LAST JOCKEYED EVENT: ", jockey_event)
                expected_processing_time = jockey_event['expected_service_time']

                if jockey_event['reward'] > 0 and expected_processing_time < self.queue_state['long_avg_serv_time']:
                    self.queue_state['at_pose'] += 1
                    self.queue_state['reward'] = jockey_event['reward']

                    # Update the history to reflect the jockeyed state
                    self.requestObj.srv2_history.append({
                        "ServerID": queue_id,
                        "at_pose": self.queue_state['at_pose'],
                        "rate_jockeyed": self.queue_state['rate_jockeyed'],
                        "rate_reneged": self.queue_state['rate_reneged'],
                        "expected_service_time": expected_processing_time,
                        "this_busy": self.queue_state['this_busy'],
                        "long_avg_serv_time": self.queue_state['long_avg_serv_time'],
                        "time_service_took": expected_processing_time,
                        "reward": jockey_event['reward'],
                        "action": "jockeyed",
                        "intensity_based_info": self.queue_state['intensity_based_info']
                    })
                    return self.queue_state
            else:
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]
                return self.queue_state        


    def get_jockey_action_outcome_original(self, queue_id):
        """
        Compute the state of the queue after a jockey action.

        Args:
            queue_id (str): The ID of the source queue for the jockey action.

        Returns:
            dict: The resulting state of the queue.
        """
        
        srv = self.queue_state.get('ServerID') # curr_state.get('ServerID')
        
        if srv == 1:
            if len(self.requestObj.get_current_jockey_observations()) > 0:                 
                self.queue_state['at_pose'] = int(self.requestObj.get_current_jockey_observations()[-1]['at_pose']) + 1 # self.Observations.get_jockey_obs()[0]['at_pose'] + 1
                self.queue_state["reward"] = self.requestObj.get_current_jockey_observations()[-1]['reward'] # self.Observations.get_jockey_obs()[0]['reward']
                #print("\n ===== Reward Jockey ==== ", srv,self.queue_state["reward"])
                return self.queue_state
            else:                
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]
                    #print("\n ---- Reward Jockey---- ", srv,self.queue_state["reward"])
                    return self.queue_state
                else:    
                    #print("\n ---- Reward Jockey---- ", srv, self.queue_state["reward"])
                    return self.queue_state
        else:
            if len(self.requestObj.get_current_jockey_observations()) > 0: 
                
                self.queue_state['at_pose'] = int(self.requestObj.get_current_jockey_observations()[-1]['at_pose']) + 1 
                self.queue_state["reward"] = self.requestObj.get_current_jockey_observations()[-1]['reward']
                #print("\n ***** Reward Jockey**** ", srv,self.queue_state["reward"]) 
                return self.queue_state
            else:
                
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]  
                    #print("\n ***** Reward Jockey**** ", srv,self.queue_state["reward"])              
                    return self.queue_state
                else: 
                    #print("\n ***** Reward Jockey**** ", srv,self.queue_state["reward"])   
                    return self.queue_state
                             

    def get_action_to_state(self):
		
        self._action_to_state = {
            Actions.RENEGE.value: self.get_renege_action_outcome(self.queue_id), 
            Actions.JOCKEY.value: self.get_jockey_action_outcome(self.queue_id)
        }
        return self._action_to_state
        
        
    def step(self, action):
        """
        Execute a step in the environment based on the given action.

        Args:
            action (int): The action to perform (RENEGE or JOCKEY).

        Returns:
            tuple: (observation, reward, terminated, info)
        """
        new_state = self.get_action_to_state()  # Map actions to states

        # Initialize variables
        terminated = False
        reward = 0
        
        # Increase step count
        self.current_step += 1

        # Check if the action exists in the new_state mapping
        if action in new_state:
            value = new_state[action]

            # Ensure value is not None before accessing its keys
            if value is not None and isinstance(value, dict):
                #terminated = value.get("at_pose", 0) <= 0  # Fallback to 0 if "at_pose" is missing
                queue_empty = value.get("at_pose", 0) <= 0
                reward = value.get("reward", 0)  # Fallback to 0 if "reward" is missing
                self.queue_state = value
            else:
                queue_empty = False
                #print(f"Warning: Action {action} does not map to a valid state. Skipping.")
        else:
            queue_empty = False
            #print(f"Warning: Action {action} is not recognized. Skipping.")
            
        # Terminate if queue is empty or max_steps reached
        terminated = queue_empty or (self.current_step >= self.max_steps)

        # Get observation and info
        observation = self.queue_state
        info = self._get_info()

        return observation, reward, terminated, info

 
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
              

def visualize_results(metrics_file="simu_results.csv", window=5):
    """
    Visualize simulation results recorded in the metrics file.

    - Single combined rates plot showing both reneging and jockeying for each server.
    - Logarithmic y-axis with only major grid lines (clean look).
    - Distinct colors for metric types, line style differentiates servers.
    - Small epsilon added before log-scaling to avoid zeros.
    Returns the matplotlib Figure object.
    """
    import matplotlib.ticker as mticker

    metrics = pd.read_csv(metrics_file)

    # moving average helper
    def moving_average(x, w):
        if w < 2:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # Smoothed average rewards
    episodes = metrics["episode"].to_numpy()
    avg_rewards = metrics.get("average_reward", pd.Series(np.zeros(len(metrics)))).to_numpy()
    smoothed_rewards = moving_average(avg_rewards, window)
    episodes_smoothed = episodes[: len(smoothed_rewards)]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_smoothed, smoothed_rewards, label=f"Avg Reward (window={window})", marker="o")
    plt.title("Smoothed Average Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Loss metrics
    plt.figure(figsize=(10, 5))
    if "actor_loss" in metrics.columns:
        plt.plot(metrics["episode"], metrics["actor_loss"], label="Actor Loss", marker="o", color="red")
    if "critic_loss" in metrics.columns:
        plt.plot(metrics["episode"], metrics["critic_loss"], label="Critic Loss", marker="o", color="blue")
    if "total_loss" in metrics.columns:
        plt.plot(metrics["episode"], metrics["total_loss"], label="Total Loss", marker="o", color="purple")
    plt.title("Loss Metrics per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Helper to safely fetch a series (returns zeros if missing)
    def get_series(df, col):
        return df[col].to_numpy() if col in df.columns else np.zeros(len(df))

    # Server-specific columns
    srv1_renege_markov = get_series(metrics, "srv1_reneging_rate_markov")
    srv2_renege_markov = get_series(metrics, "srv2_reneging_rate_markov")
    srv1_renege_nn = get_series(metrics, "srv1_reneging_rate_nn")
    srv2_renege_nn = get_series(metrics, "srv2_reneging_rate_nn")

    srv1_jock_markov = get_series(metrics, "srv1_jockeying_rate_markov")
    srv2_jock_markov = get_series(metrics, "srv2_jockeying_rate_markov")
    srv1_jock_nn = get_series(metrics, "srv1_jockeying_rate_nn")
    srv2_jock_nn = get_series(metrics, "srv2_jockeying_rate_nn")

    # Add small epsilon so log scale works even when values are zero
    eps = 1e-3 #1e-9
    srv1_renege_markov = srv1_renege_markov + eps
    srv2_renege_markov = srv2_renege_markov + eps
    srv1_renege_nn = srv1_renege_nn + eps
    srv2_renege_nn = srv2_renege_nn + eps
    srv1_jock_markov = srv1_jock_markov + eps
    srv2_jock_markov = srv2_jock_markov + eps
    srv1_jock_nn = srv1_jock_nn + eps
    srv2_jock_nn = srv2_jock_nn + eps

    # Single combined plot: lines grouped by metric-type color, server distinguished by linestyle
    fig, ax = plt.subplots(figsize=(14, 7))

    # Colors grouped by metric type:
    # - Reneging: blue (markov) / red (nn) color family
    # - Jockeying: purple / cyan (avoids green/yellow)
    # Server differentiation: solid = srv1, dotted = srv2
    ax.plot(metrics["episode"], srv1_renege_markov, label="Srv1 Reneging Markov", color="purple", linestyle="-", linewidth=1.5) # tab:
    ax.plot(metrics["episode"], srv2_renege_markov, label="Srv2 Reneging Markov", color="purple", linestyle=":", linewidth=1.5) # tab:
    ax.plot(metrics["episode"], srv1_renege_nn, label="Srv1 Reneging NN", color="tab:red", linestyle="-", linewidth=1.5)
    ax.plot(metrics["episode"], srv2_renege_nn, label="Srv2 Reneging NN", color="tab:red", linestyle=":", linewidth=1.5)

    #ax.plot(metrics["episode"], srv1_jock_markov, label="Srv1 Jockeying Markov", color="tab:purple", linestyle="-", linewidth=1.5)
    #ax.plot(metrics["episode"], srv2_jock_markov, label="Srv2 Jockeying Markov", color="tab:purple", linestyle=":", linewidth=1.5)
    #ax.plot(metrics["episode"], srv1_jock_nn, label="Srv1 Jockeying NN", color="tab:cyan", linestyle="-", linewidth=1.5)
    #ax.plot(metrics["episode"], srv2_jock_nn, label="Srv2 Jockeying NN", color="tab:cyan", linestyle=":", linewidth=1.5)
    # JOCKEYING lines - DARKER colors (replaced cyan/purple with navy and darkslategray)
    # navy = dark blue (distinct from tab:blue by linestyle/linewidth), darkslategray = dark gray
    ax.plot(metrics["episode"], srv1_jock_markov, label="Srv1 Jockeying Markov", color="navy", linestyle="-", linewidth=1.5)
    ax.plot(metrics["episode"], srv2_jock_markov, label="Srv2 Jockeying Markov", color="navy", linestyle=":", linewidth=1.5)
    ax.plot(metrics["episode"], srv1_jock_nn, label="Srv1 Jockeying NN", color="green", linestyle="-", linewidth=1.5) # darkslategray
    ax.plot(metrics["episode"], srv2_jock_nn, label="Srv2 Jockeying NN", color="green", linestyle=":", linewidth=1.5) # darkslategray

    # Log scale, grid (major only), and tick formatting
    ax.set_yscale("log")
    ax.set_ylim(bottom=eps)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (log scale)")
    ax.set_title("Jockeying and Reneging Rates per Server (log y-axis)")
    ax.grid(which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.7)
    ax.minorticks_off()

    # Use LogLocator for nicer major ticks and scientific formatting
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))

    # ax.legend(loc="best", fontsize="small", ncol=2)
    ax.legend(loc="best", fontsize=12, ncol=2)
    plt.tight_layout()
    plt.savefig("logarithmic_rates_episodic.pdf", format="pdf", dpi=300, bbox_inches="tight")
    # plt.show()

    return fig


def plot_utility_vs_impatience(metrics_file="non_adjusted_metrics.csv", bins=5, annotate=True, annotate_every=1):
    """
    Read per-episode metrics CSV and plot average tenant utility vs impatience rate.
    Points are colored by the cost_wait parameter used that episode (binned into 'bins' intervals).
    Options:
      - bins: number of discrete cost intervals (use bins=1 for continuous colormap)
      - annotate: whether to draw a small text label near each point with the cost_wait value
      - annotate_every: annotate every N-th point (useful when many episodes)
    Returns:
      (fig, ax)
    """
    import matplotlib as mpl

    metrics = pd.read_csv(metrics_file)
    required = {"impatience_rate", "avg_tenant_utility", "cost_wait"}
    if not required.issubset(set(metrics.columns)):
        raise ValueError(f"Metrics file must contain columns: {required}. Found: {list(metrics.columns)}")

    x = metrics["impatience_rate"].to_numpy(dtype=float)
    y = metrics["avg_tenant_utility"].to_numpy(dtype=float)
    c = metrics["cost_wait"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    # If all cost_wait values identical, use a single color and avoid colorbar issues
    if np.allclose(c, c[0]):
        sc = ax.scatter(x, y, c="tab:blue", edgecolor="k", s=70)
        cbar = None
    else:
        cmap = plt.get_cmap("viridis")
        if bins is None or bins <= 1:
            # continuous colormap
            sc = ax.scatter(x, y, c=c, cmap=cmap, edgecolor="k", s=70)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("cost_wait")
        else:
            # discrete bins with boundary normalization
            bounds = np.linspace(c.min(), c.max(), bins + 1)
            norm = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
            sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, edgecolor="k", s=70)
            cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, boundaries=bounds)
            # label tick values with bin centers for readability
            tick_locs = 0.5 * (bounds[:-1] + bounds[1:])
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels([f"{t:.3f}" for t in tick_locs])
            cbar.set_label("cost_wait (binned)")

    # Annotate points with cost_wait value (small font), but avoid overcrowding by annotate_every
    if annotate:
        for idx, (xx, yy, cc) in enumerate(zip(x, y, c)):
            if idx % annotate_every == 0:
                ax.annotate(f"{cc:.2f}", (xx, yy),
                            textcoords="offset points", xytext=(5, 3), ha="left", fontsize=8, alpha=0.9)

    # regression line (optional) + robust plotting niceties
    try:
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ax.plot(xs, np.polyval(coeffs, xs), color="tab:orange", linestyle="--", label=f"Linear fit (slope={coeffs[0]:.3f})")
    except Exception:
        pass

    ax.set_xlabel("Impatience Rate (reneges + jockeys) / requests")
    ax.set_ylabel("Average Tenant Utility (per episode)")
    ax.set_title("Average Tenant Utility vs Impatience Rate\n(points colored by cost_wait)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=12, ncol=2)
    plt.tight_layout()
    plt.savefig("impatience_utility_episodic.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    return fig, ax


def visualize_results_broken(metrics_file="simu_results.csv", window=5):
    """
    Visualize simulation results recorded in the metrics file.

    Changes in this version:
    - Plot reneging and jockeying rates separately for each server (srv1, srv2)
      instead of plotting aggregated means.
    - Use a logarithmic y-axis for the rate plots.
    - Use major-grid only (cleaner) to avoid the dense dotted grid on log scale.
    - Add a tiny epsilon before applying log scale so zero values don't cause errors.
    """
    import matplotlib.ticker as mticker

    metrics = pd.read_csv(metrics_file)

    # moving average helper
    def moving_average(x, w):
        if w < 2:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # Smoothed average rewards (same as before)
    avg_rewards = metrics["average_reward"].to_numpy()
    smoothed_rewards = moving_average(avg_rewards, window)
    episodes = metrics["episode"].to_numpy()
    episodes_smoothed = episodes[: len(smoothed_rewards)]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_smoothed, smoothed_rewards, label=f"Avg Reward (window={window})", marker="o")
    plt.title("Smoothed Average Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Loss metrics (unchanged)
    plt.figure(figsize=(10, 6))
    if "actor_loss" in metrics.columns:
        plt.plot(metrics["episode"], metrics["actor_loss"], label="Actor Loss", marker="o", color="red")
    if "critic_loss" in metrics.columns:
        plt.plot(metrics["episode"], metrics["critic_loss"], label="Critic Loss", marker="o", color="blue")
    if "total_loss" in metrics.columns:
        plt.plot(metrics["episode"], metrics["total_loss"], label="Total Loss", marker="o", color="purple")
    plt.title("Loss Metrics per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Rates per server (log y-axis) ---
    # helper to safely get series (returns zeros if missing)
    def get_series(df, col):
        if col in df.columns:
            return df[col].to_numpy()
        else:
            return np.zeros(len(df))

    # read server-specific columns
    srv1_renege_markov = get_series(metrics, "srv1_reneging_rate_markov")
    srv2_renege_markov = get_series(metrics, "srv2_reneging_rate_markov")
    srv1_renege_nn = get_series(metrics, "srv1_reneging_rate_nn")
    srv2_renege_nn = get_series(metrics, "srv2_reneging_rate_nn")

    srv1_jock_markov = get_series(metrics, "srv1_jockeying_rate_markov")
    srv2_jock_markov = get_series(metrics, "srv2_jockeying_rate_markov")
    srv1_jock_nn = get_series(metrics, "srv1_jockeying_rate_nn")
    srv2_jock_nn = get_series(metrics, "srv2_jockeying_rate_nn")

    # epsilon to avoid zeros on log scale
    eps = 1e-9
    srv1_renege_markov = srv1_renege_markov + eps
    srv2_renege_markov = srv2_renege_markov + eps
    srv1_renege_nn = srv1_renege_nn + eps
    srv2_renege_nn = srv2_renege_nn + eps
    srv1_jock_markov = srv1_jock_markov + eps
    srv2_jock_markov = srv2_jock_markov + eps
    srv1_jock_nn = srv1_jock_nn + eps
    srv2_jock_nn = srv2_jock_nn + eps

    # Create two subplots: one for reneging (srv1 & srv2), one for jockeying (srv1 & srv2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    ax_r, ax_j = axes

    # RENEGING: server 1 and server 2, markov vs nn
    ax_r.plot(metrics["episode"], srv1_renege_markov, label="Srv1 Reneging Markov", color="tab:blue", linestyle="-")
    ax_r.plot(metrics["episode"], srv1_renege_nn, label="Srv1 Reneging NN", color="tab:red", linestyle="--")
    ax_r.plot(metrics["episode"], srv2_renege_markov, label="Srv2 Reneging Markov", color="tab:green", linestyle="-")
    ax_r.plot(metrics["episode"], srv2_renege_nn, label="Srv2 Reneging NN", color="tab:orange", linestyle="--")
    ax_r.set_yscale("log")
    ax_r.set_ylim(bottom=eps)
    ax_r.set_xlabel("Episode")
    ax_r.set_ylabel("Reneging Rate (log scale)")
    ax_r.set_title("Reneging Rates per Server (log y-axis)")
    # Major grid only for clarity
    ax_r.grid(which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.7)
    ax_r.minorticks_off()
    ax_r.legend()

    # JOCKEYING: server 1 and server 2, markov vs nn
    ax_j.plot(metrics["episode"], srv1_jock_markov, label="Srv1 Jockeying Markov", color="tab:purple", linestyle="-")
    ax_j.plot(metrics["episode"], srv1_jock_nn, label="Srv1 Jockeying NN", color="tab:cyan", linestyle="--")
    ax_j.plot(metrics["episode"], srv2_jock_markov, label="Srv2 Jockeying Markov", color="tab:brown", linestyle="-")
    ax_j.plot(metrics["episode"], srv2_jock_nn, label="Srv2 Jockeying NN", color="tab:pink", linestyle="--")
    ax_j.set_yscale("log")
    ax_j.set_ylim(bottom=eps)
    ax_j.set_xlabel("Episode")
    ax_j.set_ylabel("Jockeying Rate (log scale)")
    ax_j.set_title("Jockeying Rates per Server (log y-axis)")
    ax_j.grid(which="major", linestyle="--", linewidth=0.6, color="gray", alpha=0.7)
    ax_j.minorticks_off()
    ax_j.legend()

    # Improve tick formatting on y axis (show powers of 10)
    import matplotlib.ticker as mticker
    ax_r.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax_j.yaxis.set_major_locator(mticker.LogLocator(base=10.0))

    plt.suptitle("Reneging and Jockeying Rates per Server (log y-axis)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    

def visualize_results_original(metrics_file="simu_results.csv", window=5):
    """
    Visualize simulation results recorded in the metrics file.

    Args:
        metrics_file (str): Path to the CSV file containing episode metrics.

    Returns:
        None
    """
    # Load metrics from the CSV file
    metrics = pd.read_csv(metrics_file)

    # Get average rewards
    avg_rewards = metrics['average_reward'].to_numpy()

    # Compute moving average for smoothing
    def moving_average(x, w):
        if w < 2:
            return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    smoothed_rewards = moving_average(avg_rewards, window)
    # Adjust episode numbers to match smoothed length
    episodes = metrics['episode'].to_numpy()
    episodes_smoothed = episodes[:len(smoothed_rewards)]

    # Plot Smoothed Average Rewards per Episode
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_smoothed, smoothed_rewards, label=f'Avg Reward (window={window})', marker='o')
    plt.title("Smoothed Average Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot Average Rewards per Episode
    #plt.figure(figsize=(10, 6))
    #plt.plot(metrics['episode'], metrics['average_reward'], label='Average Reward', marker='o', color='orange')
    #plt.title("Average Rewards per Episode")
    #plt.xlabel("Episode")
    #plt.ylabel("Average Reward")
    #plt.grid()
    #plt.legend()
    #plt.show()

    # Plot Policy Entropy
    #plt.figure(figsize=(10, 6))
    #plt.plot(metrics['episode'], metrics['policy_entropy'], label='Policy Entropy', marker='o', color='green')
    #plt.title("Policy Entropy per Episode")
    #plt.xlabel("Episode")
    #plt.ylabel("Policy Entropy")
    #plt.grid()
    #plt.legend()
    #plt.show()

    # Plot Loss Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['actor_loss'], label='Actor Loss', marker='o', color='red')
    plt.plot(metrics['episode'], metrics['critic_loss'], label='Critic Loss', marker='o', color='blue')
    plt.plot(metrics['episode'], metrics['total_loss'], label='Total Loss', marker='o', color='purple')
    plt.title("Loss Metrics per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot Jockeying and Reneging Rates
    # Plot Reneging Rates
    plt.figure(figsize=(12,7))

    # Reneging rates comparison
    plt.plot(metrics['episode'], metrics['srv1_reneging_rate_markov'], label='Srv1 Reneging Markov', color='blue', linestyle='-')
    plt.plot(metrics['episode'], metrics['srv2_reneging_rate_markov'], label='Srv2 Reneging Markov', color='blue', linestyle='--')
    plt.plot(metrics['episode'], metrics['srv1_reneging_rate_nn'], label='Srv1 Reneging NN', color='red', linestyle='-')
    plt.plot(metrics['episode'], metrics['srv2_reneging_rate_nn'], label='Srv2 Reneging NN', color='red', linestyle='--')

    # Jockeying rates comparison
    plt.plot(metrics['episode'], metrics['srv1_jockeying_rate_markov'], label='Srv1 Jockeying Markov', color='green', linestyle='-')
    plt.plot(metrics['episode'], metrics['srv2_jockeying_rate_markov'], label='Srv2 Jockeying Markov', color='green', linestyle='--')
    plt.plot(metrics['episode'], metrics['srv1_jockeying_rate_nn'], label='Srv1 Jockeying NN', color='orange', linestyle='-')
    plt.plot(metrics['episode'], metrics['srv2_jockeying_rate_nn'], label='Srv2 Jockeying NN', color='orange', linestyle='--')

    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.title('Comparison of Jockeying and Reneging Rates: NN vs Markov')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

    # Calculate mean rates per source per episode
    #metrics['mean_reneging_markov'] = metrics[['srv1_reneging_rate_markov', 'srv2_reneging_rate_markov']].mean(axis=1)
    #metrics['mean_jockeying_markov'] = metrics[['srv1_jockeying_rate_markov', 'srv2_jockeying_rate_markov']].mean(axis=1)
    #metrics['mean_reneging_nn'] = metrics[['srv1_reneging_rate_nn', 'srv2_reneging_rate_nn']].mean(axis=1)
    #metrics['mean_jockeying_nn'] = metrics[['srv1_jockeying_rate_nn', 'srv2_jockeying_rate_nn']].mean(axis=1)

    # Plot the mean rates per episode
    #plt.figure(figsize=(10,6))
    #plt.plot(metrics['episode'], metrics['mean_reneging_markov'], label='Mean Reneging Markov', color='blue')
    #plt.plot(metrics['episode'], metrics['mean_reneging_nn'], label='Mean Reneging NN', color='red')
    #plt.plot(metrics['episode'], metrics['mean_jockeying_markov'], label='Mean Jockeying Markov', color='green')
    #plt.plot(metrics['episode'], metrics['mean_jockeying_nn'], label='Mean Jockeying NN', color='orange')
    #plt.xlabel('Episode')
    #plt.ylabel('Mean Rate')
    #plt.title('Mean Reneging and Jockeying Rates per Source (Aggregated Across Both Servers)')
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()


def visualize_comparison(adjusted_file="adjusted_metrics.csv", non_adjusted_file="non_adjusted_metrics.csv"):
    """
    Visualize and compare metrics between adjusted and non-adjusted service rates.

    Args:
        adjusted_file (str): Path to the metrics file for adjusted service rates.
        non_adjusted_file (str): Path to the metrics file for non-adjusted service rates.

    Returns:
        None
    """
    # Load metrics
    adjusted_metrics = pd.read_csv(adjusted_file)
    non_adjusted_metrics = pd.read_csv(non_adjusted_file)

    # Plot Jockeying Rates
    plt.figure(figsize=(12, 6))
    plt.plot(adjusted_metrics['episode'], adjusted_metrics['average_jockeying_rate'], label='Adjusted Service Rate', marker='o')
    plt.plot(non_adjusted_metrics['episode'], non_adjusted_metrics['average_jockeying_rate'], label='Non-Adjusted Service Rate', marker='x')
    #plt.title("Average Jockeying Rates per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Jockeying Rate")
    plt.legend()
    plt.grid()
    plt.show()

    

def main():       
	
    utility_basic = 1.0
    discount_coef = 0.1
    state_dim = 11  # Example state dimension
    action_dim = 2  # Example action dimension

    # Initialize Actor-Critic model and A2C agent
    actor_critic = ActorCritic(state_dim, action_dim)
    agent = A2CAgent(state_dim, action_dim)    
    
    # Initialize RequestQueue with all required arguments
    request_queue = RequestQueue(state_dim, action_dim, utility_basic, discount_coef, actor_critic, agent)

    # Initialize ImpatientTenantEnv with the RequestQueue instance
    env = ImpatientTenantEnv()
    env.requestObj = request_queue  # Set the RequestQueue instance in the environment

    print("Environment and RequestQueue initialized successfully!")
    
    # These are the number of iterations also labelled as step in the simulation
    # Ideally each step corresponds to an arrival and processing iteration
    duration = 8     
    
    # Start the scheduler
    #scheduler_thread = threading.Thread(target=request_queue.run(duration, env, adjust_service_rate=False, save_to_file="non_adjusted_metrics.csv")) # requestObj.run_scheduler) #
    scheduler_thread = threading.Thread(
        target=request_queue.run,
        args=(duration, env),
        kwargs={
            'adjust_service_rate': False,
            'num_episodes': 5, #   <-- set to 120 episodes, or any number > 100
            'save_to_file': "non_adjusted_metrics.csv"
            # "arrival_rates": env.arrival_rates
        }
    ) 
    scheduler_thread.start()
    scheduler_thread.join()  # <-- Wait until simulation is done then plot below functions
    
    # Let us visualize some results
    visualize_results(metrics_file="simu_results.csv") # has the rewards metrics, and figure 2 in the paper
    request_queue.plot_rates() # change me not to use dispatch_data but use self.subscribers variables    
    request_queue.compare_rates_by_information_source() # _with_graph
    # request_queue.plot_asymptotic_behavior_of_jockeying_and_reneging(max_queue_length=1000)
    request_queue.plot_reneging_and_jockeying_rates_vs_queue_size_by_type() # plot_reneging_and_jockeying_rates_vs_queue_size()
    
    # request_queue.plot_decision_probability_vs_queue_length(bin_width=1)
    # request_queue.plot_diagnostics_error_and_misclassification() #max_queue_length=500)    
    # request_queue.plot_successful_jockey_probability_vs_queue_length() # T=None, bin_width=1)
    
    request_queue.compute_success_table_with_max_local_utility()
    
    #request_queue.plot_successful_jockey_probability_vs_queue_length(T=None, max_queue_length=500)
    
    # request_queue.compare_behavior_rates_by_information_how_often()
    # request_queue.plot_jockeying_reneging_rates_vs_waiting_time_difference()
    
    # request_queue.plot_reneging_time_vs_queue_length()
    # request_queue.plot_jockeying_time_vs_queue_length()
    # request_queue.plot_waiting_time_vs_queue_length_with_fit()
    
    # visualize_comparison("adjusted_metrics.csv", "non-adjusted_metrics.csv")
    
    # Run equal service rate experiment
    #arrival_rates_even = [2,4,6,8,10,12,14,16]
    # equal_rate = 5.0  # <-- Set your desired equal service rate
    #request_queue.run_equal_service_rate_experiment(arrival_rates_even, duration, env, num_episodes=20)        
          

if __name__ == "__main__":
    main()
