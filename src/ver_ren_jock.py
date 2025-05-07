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
logging.basicConfig(
    filename="request_decisions.log",
    filemode="a",
    format="%(asctime)s - Request ID: %(request_id)s - Queue ID: %(queue_id)s - Action: %(action)s",
    level=logging.INFO
)

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
        
          
        
    def set_renege_obs(self, curr_pose, queue_intensity, jockeying_rate, reneging_rate ,time_local_service, time_to_service_end, reward, queueid, activity, long_avg_serv_time, uses_intensity_based):		

        self.curr_obs_renege.append(
            {   
                "ServerID": queueid,
                "at_pose": curr_pose,
                "rate_jockeyed": jockeying_rate,
                "rate_reneged": reneging_rate,                
                "this_busy": queue_intensity,
                "expected_service_time":time_local_service,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity,
                "long_avg_serv_time": long_avg_serv_time,
                "intensity_based_info": uses_intensity_based
            }
        )
        
        return self.curr_obs_renege
        
        
    def set_jockey_obs(self, curr_pose, queue_intensity, jockeying_rate, reneging_rate ,time_local_service, time_to_service_end, reward, queueid, activity, long_avg_serv_time, uses_intensity_based):
        
        self.curr_obs_jockey.append(
            {
                "ServerID": queueid,
                "at_pose": curr_pose,
                "rate_jockeyed": jockeying_rate,
                "rate_reneged": reneging_rate,                
                "this_busy": queue_intensity,
                "expected_service_time":time_local_service,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity,
                "long_avg_serv_time": long_avg_serv_time,
                "intensity_based_info": uses_intensity_based			
            }
        )
        
    
    def get_jockey_obs(self):
		
        return self. curr_obs_jockey    


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

    def __init__(self,uses_nn, uses_intensity_based, time_entrance,pos_in_queue=0,utility_basic=0.0,service_time=0.0,discount_coef=0.0, outage_risk=0.1, # =timer()
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
        print(f"Stored reward: {reward}, Total rewards: {self.rewards}")
        

    def update(self):
        
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
            return
            
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        print(f"Returns: {returns}")
        returns = torch.tensor(returns).to(device)            
            
        # values = torch.cat(self.values).to(device)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss
        
        print("\n total loss of actor + critic:", loss) #loss.requires_grad)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear stored values
        self.log_probs = []
        self.values = []
        self.rewards = []


class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, state_dim, action_dim, utility_basic, discount_coef, actor_critic, agent, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon, exp_time_service_end=0.0,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, batchid=np.int16, uses_nn=False, uses_intensity_based = False): # Dispatched
                 
        
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
        
        BROADCAST_INTERVAL = 5
        
        return               
		
	
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

        if progress_bar:
            step_loop = tqdm(range(steps_per_episode), leave=False, desc='     Current run')
        else:
            step_loop = range(steps_per_episode)                             
        
        
        srv_1 = self.dict_queues_obj.get("1") # Server1
        srv_2 = self.dict_queues_obj.get("2") # Server2               
        
        for episode in range(num_episodes):
            print(f"Starting Episode {episode + 1}/{num_episodes}")            
            
            # Reset environment for the new episode
            state, info = self.env.reset(seed=42)
            total_reward = 0
            done = False  # Flag to track episode termination
            episode_start_time = time.time()
            i = 0
            episode_policy_entropy = 0  # Track total policy entropy for the episode
            losses = {"actor_loss": 0, "critic_loss": 0, "total_loss": 0}
            jockeying_rates = []
            reneging_rates = []
        
            for i in step_loop: 
                self.arr_rate = self.objQueues.get_arrivals_rates()
                print("\n Arrival rate: ", self.arr_rate)  
                if done:  # Break the loop if the episode ends
                    break         
			
                if progress_log:
                    print("Step", i + 1, "/", steps_per_episode) # print("Step",i,"/",steps)
                        
                self.markov_model.updateState()
                

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
                # print("\n NEXT STATE: ", next_state)
                
                # Update queue state in the environment
                self.env.update_queue_state(next_state)
                
                    #queue_id=queue_id,
                    #curr_queue=curr_queue,
                    #dest_queue=dest_queue,
                    #queue_sizes=queue_sizes,
                    #service_rates=service_rates,
                    #total_reward=total_reward,
                    #action=action,
                #)

                # Step 3: Store the reward for training
                self.agent.store_reward(reward)
                total_reward += reward

                # Step 4: Update the state for the next step
                state = next_state
                i += 1            
                              
                self.processEntries(all_entries, i) #, self.uses_nn)
                self.time+=self.time_res
            
                # Step 4 (Optional): Adjust service rates if enabled
                if adjust_service_rate:
                    self.adjust_service_rates()

                # Step 5: Compute jockeying and reneging rates
                queue_jockeying_rate = self.compute_jockeying_rate(self.dict_queues_obj["1"])
                queue_reneging_rate = self.compute_reneging_rate(self.dict_queues_obj["1"])
                jockeying_rates.append(queue_jockeying_rate)
                reneging_rates.append(queue_reneging_rate)
                
                # Ensure dispatch data is updated at each step
                self.dispatch_all_queues() #dispatch_all_queues()
                #self.run_scheduler(duration)
                
                # Optional: Log step-level progress (can be verbose)
                if progress_log:
                    print(f"Step {i + 1}: Action={action}, Reward={reward}, Total Reward={total_reward}")
            
                self.set_batch_id(i)
                
            # Update the RL agent at the end of each episode
            self.agent.update()
            
            # Calculate episode duration
            episode_duration = time.time() - episode_start_time

            # Log metrics for the episode
            episode_metrics = {
                "episode": episode + 1,
                "total_reward": total_reward,
                "average_reward": total_reward / i if i > 0 else 0,
                "steps": i,
                "duration": episode_duration,
                #"policy_entropy": episode_policy_entropy / i if i > 0 else 0,
                "actor_loss": losses["actor_loss"],
                "critic_loss": losses["critic_loss"],
                "total_loss": losses["total_loss"],
                "average_jockeying_rate": sum(jockeying_rates) / len(jockeying_rates) if jockeying_rates else 0,
                "average_reneging_rate": sum(reneging_rates) / len(reneging_rates) if reneging_rates else 0
            }
            metrics.append(episode_metrics)

            # Print episode summary
            print(f"Episode {episode + 1} Summary: Total Reward={total_reward}, "
                  f"Avg Reward={episode_metrics['average_reward']:.2f}, Steps={i}, "
                  # f"Policy Entropy={episode_metrics['policy_entropy']:.2f}, "
                  f"Actor Loss={losses['actor_loss']:.4f}, Critic Loss={losses['critic_loss']:.4f}, "
                  f"Total Loss={losses['total_loss']:.4f}, Steps={i}, " # steps
                  f"Duration={episode_duration:.2f}s")
              
            print(f"Episode {episode + 1} finished with a total reward of {total_reward}")
            
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
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
            reneging_rate = self.compute_jockeying_rate(curr_queue) # MATCH
                
            if q_selector == 1:					
                self.queueID = "1" # Server1
                    
                """
                     Run the serveOneRequest function a random number of times before continuing.
                """
                              
                self.serveOneRequest(self.queueID, jockeying_rate, reneging_rate, entries) # Server1 = self.dict_queues_obj["1"][0], entry[0],
                                                                                                      
                ### self.dispatch_queue_state(self.dict_queues_obj["1"], self.queueID) #, self.dict_queues_obj["2"]) #, req)
                time.sleep(random.uniform(0.1, 0.5))  # Random delay between 0.1 and 0.5 seconds
                                               
            else:
                self.queueID = "2"
                self.serveOneRequest(self.queueID,  jockeying_rate, reneging_rate, entries) # Server2 = self.dict_queues_obj["2"][0], entry[0],  
                # self.initialize_queue_states(self.queueID,len(curr_queue), self.jockeying_rate, self.reneging_rate, req)                                     
                ### self.dispatch_queue_state(self.dict_queues_obj["2"], self.queueID) #, self.dict_queues_obj["1"]) #, req)
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
            req=Request(uses_nn, self.uses_intensity_based,time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid)
                    
            self.nn_subscribers.append(req)
            
        else:
            req=Request(uses_nn, self.uses_intensity_based, time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid)
                    
            self.state_subscribers.append(req)  

        #print("\n LENGTHS => ", len(self.state_subscribers), " ==== ", len(self.nn_subscribers))   
  
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
        
    
    def dispatch_queue_state(self, curr_queue, curr_queue_id, uses_intensity_based):
		
        rate_srv1,rate_srv2 = self.get_server_rates()
		
        if curr_queue_id == "1":
            alt_queue_id = "2"
            serv_rate = rate_srv1
        else:
            alt_queue_id = "1"
            serv_rate = rate_srv2

        curr_queue_state = self.get_queue_state(alt_queue_id) # , curr_queue)
        
        self.env.update_queue_state(curr_queue_state)

        # Dispatch queue state to requests and allow them to act
        if not isinstance(None, type(curr_queue_state)):
            for req in curr_queue:
                if req.uses_nn:  # NN-based requests
                    
                    action = self.get_nn_optimized_decision(curr_queue_state) 
                               
                    next_state, reward, done, _ = self.env.step(action['action'])  # Apply action
                    # print("\n That ACTION :", action, " in state: ",curr_queue_state," will land you in the STATE: ", next_state)
                    self.agent.store_reward(reward)  # Store the reward for training

                    # Train RL model after processing each request
                    if done:
                        self.agent.update()
                    
                    if action['action'] == 0: #action == 0:
                        print(f"ActorCriticInfo [RENEGE]: Server {alt_queue_id} in state:  {curr_queue_state}. Dispatching {next_state} to all {len(self.nn_subscribers)} requests  in server {curr_queue_id}")
                        self.makeRenegingDecision(req, curr_queue_id, uses_intensity_based)                    
                    elif action['action'] ==  1: #action == 1:
                        print(f"ActorCriticInfo [JOCKEY]: Server {alt_queue_id} in state:  {curr_queue_state}. Dispatching {next_state} to all {len(self.nn_subscribers)} requests  in server {curr_queue_id}")
                        self.makeJockeyingDecision(req, curr_queue_id, alt_queue_id, req.customerid, serv_rate, uses_intensity_based) # STATE
                else: 
                    print(f"Raw Markovian:  Server {alt_queue_id} in state {curr_queue_state}. Dispatching state to all {len(self.state_subscribers)} requests  in server {curr_queue_id}")
                    self.makeRenegingDecision(req, curr_queue_id, uses_intensity_based)                                   
                    self.makeJockeyingDecision(req, curr_queue_id, alt_queue_id, req.customerid, serv_rate, uses_intensity_based)
                        
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
        intensity_based_requests = [req for req in self.state_subscribers if req.uses_nn]  # Example flag
        departure_based_requests = [req for req in self.state_subscribers if not req.uses_nn]

        # Calculate rates for intensity-based requests
        reneging_rate_intensity = self.compute_reneging_rate(intensity_based_requests)
        jockeying_rate_intensity = self.compute_jockeying_rate(intensity_based_requests)

        # Calculate rates for departure-based requests
        reneging_rate_departure = self.compute_reneging_rate(departure_based_requests)
        jockeying_rate_departure = self.compute_jockeying_rate(departure_based_requests)

        # Print and return the comparison
        comparison = {
            "intensity_based": {
                "reneging_rate": reneging_rate_intensity,
                "jockeying_rate": jockeying_rate_intensity,
            },
            "departure_based": {
                "reneging_rate": reneging_rate_departure,
                "jockeying_rate": jockeying_rate_departure,
            },
        }
        print("Comparison of Rates by Information Source:")
        print(comparison)
        return comparison
    
    
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
            jockeying_rate_raw = self.compute_jockeying_rate(curr_queue)
            reneging_rate_raw = self.compute_reneging_rate(curr_queue)

            # Compute jockeying and reneging rates for NN-based information
            jockeying_rate_nn = jockeying_rate_raw * 1.1  # Example logic (adjust based on your simulation logic)
            reneging_rate_nn = reneging_rate_raw * 0.9    # Example logic (adjust based on your simulation logic)

            # Update dispatch data
            if queue_id == "1":
                self.dispatch_data["server_1"]["num_requests"].append(queue_size)
                self.dispatch_data["server_1"]["jockeying_rate_raw"].append(jockeying_rate_raw)
                self.dispatch_data["server_1"]["jockeying_rate_nn"].append(jockeying_rate_nn)
                self.dispatch_data["server_1"]["reneging_rate_raw"].append(reneging_rate_raw)
                self.dispatch_data["server_1"]["reneging_rate_nn"].append(reneging_rate_nn)
                self.dispatch_data["server_1"]["queue_intensity"].append(queue_intensity)
            else:
                self.dispatch_data["server_2"]["num_requests"].append(queue_size)
                self.dispatch_data["server_2"]["jockeying_rate_raw"].append(jockeying_rate_raw)
                self.dispatch_data["server_2"]["jockeying_rate_nn"].append(jockeying_rate_nn)
                self.dispatch_data["server_2"]["reneging_rate_raw"].append(reneging_rate_raw)
                self.dispatch_data["server_2"]["reneging_rate_nn"].append(reneging_rate_nn)
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
            
            
    def plot_rates(self):
        """
        Plot the comparison of jockeying and reneging rates
        for each queue for each individual information source subscribed to.
        """
        sources = ["Raw State", "NN-based"]  # Example information sources
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
		
        #observed = self.get_history()                 
        #print("\n OBSERVER: ", observed, len(observed))
        #if not isinstance(None, type(observed)):
        #if len(observed) > 0:
        #    for hist in reversed(observed):               
        #        if queueid in str(hist['ServerID']):
        #            print("\n MATCH:", hist )
        #            state = hist
        #            return state
        #else:
            #print("\n I execute the else part")
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
                    "this_busy": 0.0, 
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
                    "this_busy": 0.0, 
                    "long_avg_serv_time": self.long_avg_serv_time,
                    "time_service_took": 0.0,
                    "reward": 0.0, 
                    "action":"served",
                     "intensity_based_info": self.uses_intensity_based,				 
                })	               
                          
                return self.srv2_history  # self.history[0]  
                  
    
    def old_get_queue_state(self, queueid):
        """
        Returns the most recent state of the specified queue (server 1 or server 2)
        from the srv1_history or srv2_history variables. Ensures consistent dimensions.
        """
        # Choose the appropriate history based on the queue ID
        if queueid == "1":
            history = self.srv1_history
        elif queueid == "2":
            history = self.srv2_history
        else:
            raise ValueError(f"Invalid queue ID: {queueid}")

        # Check if the history is empty
        if not history:
            print(f"Warning: History for queue {queueid} is empty. Returning default state.")
            return [0] * 10  # Default state with 10 zeros

        # Fetch the most recent state
        latest_state = history[-1]

        # Ensure the state has the correct dimensions
        if isinstance(latest_state, dict):
            # Convert the dictionary to a list of its values (ensure consistent order)
            state_values = list(latest_state.values())
        elif isinstance(latest_state, list):
            state_values = latest_state
        else:
            raise TypeError(f"Unexpected state type in history for queue {queueid}: {type(latest_state)}")

        # Pad or truncate the state to make it 1x10
        state_values = (state_values[:10] + [0] * 10)[:10]

        return state_values
    
    

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
                              
        
        if "1" in queueID:
            self.total_served_requests_srv1+=1
            self.objObserv.set_obs(queueID, len_queue_1, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time, self.uses_intensity_based )
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
            self.objObserv.set_obs(queueID, len_queue_2, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time, self.uses_intensity_based )
            self.srv2_history.append(self.objObserv.get_obs())
            curr_intensity =  self.srv2_history[-1]
            
            if self.total_served_requests_srv2 >= abs(len(arrived_now)/2): # and "2" in queueID:
                self.departure_dispatch_count += 1
                self.uses_intensity_based = False
            
            #if curr_intensity['this_busy'] <= 2.0:
            #    print("\n =================== using intensity information ===================", curr_intensity)
            #    self.uses_intensity_based = True
            #    self.intensity_dispatch_count += 1
            
            self.history.append(self.objObserv.get_obs())
              
            self.dispatch_queue_state(queue, queueID, self.uses_intensity_based)
             
            #self.total_served_requests_srv2+=1
            #self.long_avg_serv_time = self.get_long_run_avg_service_time(queueID)                        
            #self.objObserv.set_obs(queueID, len_queue_2, queue_intensity, jockeying_rate, reneging_rate, time_in_queue, req.service_time, reward, -1, self.long_avg_serv_time, uses_intensity_based)
            #self.srv2_history.append(self.objObserv.get_obs()) 
            #curr_intensity =  self.srv2_history[-1]  #['intensity_based_info']
            #print("\n -> ", curr_intensity['this_busy'])
            #self.history.append(self.objObserv.get_obs()) 
            #self.dispatch_queue_state(queue, queueID, self.uses_intensity_based)                                         
        
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
        
			
    def get_jockey_reward(self, req):
		
        reward = 0.0
        if not isinstance(req.customerid, type(None)):	
            # print("\n Current Request: ", req)
            if '_jockeyed' in req.customerid:
                if self.avg_delay+req.time_entrance < req.service_time: #exp_time_service_end: That ACTION
                    reward = 1.0
                else:
                    reward = 0.0
                    
        return reward
        
    
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
        
        
    def get_remaining_time(self, queue_id, position): # plot_rates
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
        
        
    def makeRenegingDecision(self, req, queueid, uses_intensity_based):
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
            num_observations=min(len(self.objObserv.get_renege_obs()), len(self.history)) # queueid, queue),len(self.history)
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
                    reward = self.reqRenege( req, queueid, curr_pose, serv_rate, self.queue_intensity, self.max_local_delay, req.customerid, req.service_time, decision, self.queue, uses_intensity_based)
                                
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


    def reqRenege(self, req, queueid, curr_pose, serv_rate, queue_intensity, time_local_service, customerid, time_to_service_end, decision, curr_queue, uses_intensity_based):
        
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
            print(colored("%s", 'green') % (req.customerid) + " in Server %s" %(queueid) + " reneging now, to local processing with reward %f "%(reward) )
                                     
            self.reneging_rate = self.compute_reneging_rate(curr_queue)
            self.jockeying_rate = self.compute_jockeying_rate(curr_queue)
            self.long_avg_serv_time = self.get_long_run_avg_service_time(queueid)         
            
            self.objObserv.set_renege_obs(queueid, curr_pose, queue_intensity, self.jockeying_rate, self.reneging_rate, time_local_service, time_to_service_end, reward, 0, self.long_avg_serv_time, uses_intensity_based)                
        
            self.curr_req = req
            
            return reward
            
            
    def get_current_renege_count(self):
		
        return self.objObserv.get_renege_obs()


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
        
            
    def reqJockey(self, curr_queue_id, dest_queue_id, req, customerid, serv_rate, dest_queue, exp_delay, decision, curr_pose, curr_queue, uses_intensity_based):		        
                
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
            print(colored("%s", 'green') % (req.customerid) + " in Server %s" %(curr_queue_id) + " jockeying now, to Server %s" % (colored(dest_queue_id,'green'))+ " with reward %f"%(reward))                      
            
            self.log_action(f"Jockeyed", req, dest_queue_id)
            
            self.reneging_rate = self.compute_reneging_rate(curr_queue)
            self.jockeying_rate = self.compute_jockeying_rate(curr_queue)     
            self.long_avg_serv_time = self.get_long_run_avg_service_time(curr_queue_id)                 
            
            self.objObserv.set_jockey_obs(curr_queue_id, curr_pose, self.queue_intensity, self.jockeying_rate, self.reneging_rate, exp_delay, req.exp_time_service_end, reward, 1, self.long_avg_serv_time, uses_intensity_based) # time_alt_queue                                
            print("\n **** ", self.objObserv.get_jockey_obs()[-1])                                   
            self.curr_req = req                    
        
        return
        
    
    def get_current_jockey_observations(self):
		
        return self.objObserv.get_jockey_obs()


    def makeJockeyingDecision(self, req, curr_queue_id, alt_queue_id, customerid, serv_rate, uses_intensity_based):
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
                self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.queue, uses_intensity_based)

        # ToDo:: There is also the case of the customer willing to take the risk
        #        and jockey regardless of the the predicted loss -> Customer does not
        #        care anymore whether they incur a loss because they have already joined anyway
        #        such that reneging returns more loss than the jockeying decision

        return decision
        
    
    def compute_reneging_rate_by_info_source(self, info_src_requests):
        """Compute the reneging rate for a specific info source."""
        renegs = len(info_src_requests)  
        return renegs / len(self.get_current_renege_count()) 
        
      
    def compute_jockeying_rate_by_info_source(self, info_src_requests):
        """Compute the reneging rate for a specific info source."""
        jockeys = len(info_src_requests) 
        return jockeys / len(self.get_current_jockey_observations())
    
        
    def compare_behavior_rates_by_information_how_often(self):
        """
        Compare reneging and jockeying rates for intensity-based and departure-based requests.
        """
        jockeyed_intensity_based_requests = [d.get("intensity_based_info") for d in self.objObserv.get_jockey_obs() if "intensity_based_info" in d]
        jockeyed_departure_based_requests = [d.get("intensity_based_info") for d in self.objObserv.get_jockey_obs() if not "intensity_based_info" in d]  
        
        reneged_intensity_based_requests = [d.get("intensity_based_info") for d in self.objObserv.get_renege_obs() if "intensity_based_info" in d]  
        reneged_departure_based_requests = [d.get("intensity_based_info") for d in self.objObserv.get_renege_obs() if not "intensity_based_info" in d] 

        # Calculate rates for intensity-based requests
        reneging_rate_intensity = self.compute_reneging_rate_by_info_source(reneged_intensity_based_requests)
        jockeying_rate_intensity = self.compute_jockeying_rate_by_info_source(jockeyed_intensity_based_requests)

        # Calculate rates for departure-based requests
        reneging_rate_departure = self.compute_reneging_rate_by_info_source(reneged_departure_based_requests)
        jockeying_rate_departure = self.compute_jockeying_rate_by_info_source(jockeyed_departure_based_requests)
        
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
        plt.title("Comparison of Reneging and Jockeying Rates")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Return or print comparison
        #return {
        #    "intensity_based": {
        #        "reneging_rate": reneging_rate_intensity,
        #        "jockeying_rate": jockeying_rate_intensity,
        #    },
        #    "departure_based": {
        #        "reneging_rate": reneging_rate_departure,
        #        "jockeying_rate": jockeying_rate_departure,
        #    },
        #}
        

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
        plt.title("Comparison of Reneging and Jockeying Rates")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Show the plot
        plt.show()

    # Existing methods and attributes...
       

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
        observation = [0.0] * self.state_dim 
        info = self._get_info()     
                                    
        return observation, info


    def get_renege_action_outcome(self, queue_id):
        """
        Compute the state of the queue after a renege action.

        Args:
            queue_id (str): The ID of the queue where the renege action occurs.

        Returns:
            dict: The resulting state of the queue.
        """
        
        srv = self.queue_state.get('ServerID') # curr_state.get('ServerID')
        #print("\n *********Renege from  Server ************** : ", srv,len(self.Observations.get_renege_obs()),  len(self.requestObj.get_current_renege_count()), len(self.requestObj.get_history(queue_id))) # That ACTION
        if srv == 1:
            
            if len(self.requestObj.get_current_renege_count()) > 0: #len(self.Observations.get_renege_obs()) > 0: 
                #print("\n Inside outcome renege server 1: ", self.queue_state)
                self.queue_state['at_pose'] = len(self.requestObj.get_current_renege_count()[-1]['at_pose']) - 1 # self.Observations.get_renege_obs()[0]['at_pose'] - 1
                self.queue_state["reward"] = self.requestObj.get_current_renege_count()[-1]['reward'] # self.Observations.get_renege_obs()[0]['reward']                
                print("\n ***** Reward Renege **** ", srv,self.queue_state["reward"])
                return self.queue_state
            else:
                #print("\n No request reneged so far...returning default state")
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]
                    print("\n ***** Reward Renege **** ", srv,self.queue_state["reward"])
                    return self.queue_state
                else:
                    print("\n ***** Reward Renege **** ", srv,self.queue_state["reward"])
                    return self.queue_state
        else:
            if len(self.requestObj.get_current_renege_count()) > 0: # get_curr_obs_renege(srv)) > 0:
                self.queue_state['at_pose'] = int(self.requestObj.get_current_renege_count()[-1]['at_pose']) - 1 #self.Observations.get_renege_obs()[0]['at_pose'] - 1
                self.queue_state["reward"] = self.requestObj.get_current_renege_count()[-1]['reward'] #self.Observations.get_renege_obs()[0]['reward']
                #print("\n Inside outcome renege server 2: ", self.queue_state)
                print("\n ***** Reward Renege **** ",srv, self.queue_state["reward"])
                return self.queue_state
            else:
                #print("\n No request reneged so far...returning default state")
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]
                    print("\n ***** Reward Renege **** ", srv,self.queue_state["reward"])
                    return self.queue_state
                else: 
                    print("\n ***** Reward Renege **** ", srv,self.queue_state["reward"])   
                    return self.queue_state
        
        
        

    def get_jockey_action_outcome(self, queue_id):
        """
        Compute the state of the queue after a jockey action.

        Args:
            queue_id (str): The ID of the source queue for the jockey action.

        Returns:
            dict: The resulting state of the queue.
        """
        
        srv = self.queue_state.get('ServerID') # curr_state.get('ServerID')
        #print("\n ********* Jockey from  Server ************** : ", srv,len(self.Observations.get_jockey_obs()),  len(self.requestObj.get_current_jockey_observations()), len(self.requestObj.get_history(queue_id)))
        if srv == 1:
            if len(self.requestObj.get_current_jockey_observations()) > 0: 
                #print("\n Inside outcome jockey server 1: ", self.queue_state)
                self.queue_state['at_pose'] = int(self.requestObj.get_current_jockey_observations()[-1]['at_pose']) + 1 # self.Observations.get_jockey_obs()[0]['at_pose'] + 1
                self.queue_state["reward"] = self.requestObj.get_current_jockey_observations()[-1]['reward'] # self.Observations.get_jockey_obs()[0]['reward']
                print("\n ===== Reward Jockey ==== ", srv,self.queue_state["reward"])
                return self.queue_state
            else:
                #print("\n No request jockeyed so far...returning default state")
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]
                    print("\n ---- Reward Jockey---- ", srv,self.queue_state["reward"])
                    return self.queue_state
                else:    
                    print("\n ---- Reward Jockey---- ", srv, self.queue_state["reward"])
                    return self.queue_state
        else:
            if len(self.requestObj.get_current_jockey_observations()) > 0: 
                #print("\n Inside outcome jockey server 2: ", self.queue_state)
                self.queue_state['at_pose'] = int(self.requestObj.get_current_jockey_observations()[-1]['at_pose']) + 1 
                self.queue_state["reward"] = self.requestObj.get_current_jockey_observations()[-1]['reward']
                print("\n ***** Reward Jockey**** ", srv,self.queue_state["reward"]) 
                return self.queue_state
            else:
                #print("\n No request jockeyed so far...returning default state")
                if len(self.requestObj.get_history(queue_id)) > 0:
                    self.queue_state = self.requestObj.get_history(queue_id)[-1]  
                    print("\n ***** Reward Jockey**** ", srv,self.queue_state["reward"])              
                    return self.queue_state
                else: 
                    print("\n ***** Reward Jockey**** ", srv,self.queue_state["reward"])   
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
      
        
        new_state = self.get_action_to_state() #self._action_to_state[action]
        
        for key,value in new_state.items():
            if key == action:
                
                terminated = value["at_pose"] <= 0  # Example termination condition
                #reward = new_state["reward"]
                reward = value["reward"]
                self.queue_state = value
        # Update queue states based on the action outcome
        #if action == Actions.RENEGE.value:
        #    self.queue_state[self.queue_id] = new_state # ["NewState"]
        #elif action == Actions.JOCKEY.value:
        #    self.queue_state[self.queue_id] = new_state #["SourceState"]
        #    target_queue = "2" if self.queue_id == "1" else "1"
        #    self.queue_state[target_queue] = new_state #["TargetState"]
        
        print(f"Action: {action}, Reward: {self.queue_state['reward']}")
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
        
#env = ImpatientTenantEnv()
#state_dim = len(env.observation_space)
#action_dim = env.action_space



def visualize_results(metrics_file="simu_results.csv"):
    """
    Visualize simulation results recorded in the metrics file.

    Args:
        metrics_file (str): Path to the CSV file containing episode metrics.

    Returns:
        None
    """
    # Load metrics from the CSV file
    metrics = pd.read_csv(metrics_file)

    # Plot Total Rewards per Episode
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['total_reward'], label='Total Reward', marker='o') # This ACTION
    plt.title("Total Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.legend()
    plt.show()

    # Plot Average Rewards per Episode
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['average_reward'], label='Average Reward', marker='o', color='orange')
    plt.title("Average Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid()
    plt.legend()
    plt.show()

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
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['average_jockeying_rate'], label='Average Jockeying Rate', marker='o', color='cyan')
    plt.plot(metrics['episode'], metrics['average_reneging_rate'], label='Average Reneging Rate', marker='o', color='magenta')
    plt.title("Jockeying and Reneging Rates per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.grid()
    plt.legend()
    plt.show()


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
    plt.title("Average Jockeying Rates per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Jockeying Rate")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Reneging Rates
    plt.figure(figsize=(12, 6))
    plt.plot(adjusted_metrics['episode'], adjusted_metrics['average_reneging_rate'], label='Adjusted Service Rate', marker='o')
    plt.plot(non_adjusted_metrics['episode'], non_adjusted_metrics['average_reneging_rate'], label='Non-Adjusted Service Rate', marker='x')
    plt.title("Average Reneging Rates per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reneging Rate")
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
    
    duration = 2 # 20
    
    # Start the scheduler
    scheduler_thread = threading.Thread(target=request_queue.run(duration, env, adjust_service_rate=False, save_to_file="non_adjusted_metrics.csv")) # requestObj.run_scheduler) # 
    scheduler_thread.start()
    
    # Let us visualize some results
    #visualize_results(metrics_file="simu_results.csv")    
    #request_queue.plot_rates()
    request_queue.compare_behavior_rates_by_information_how_often()
    request_queue.compare_rates_by_information_source_with_graph()
    
    # visualize_comparison("adjusted_metrics.csv", "non-adjusted_metrics.csv")
    
    #actor_critic = requestObj.getActCritNet() # Inside    
    #agent = requestObj.getAgent()        
    
    
    #for episode in range(100):
    #    state, info = env.reset(seed=42)
    #    total_reward = 0

    #    for t in range(100):
    #        action = agent.select_action(state)
    #        next_state, reward, done, info = env.step(action)
    #        agent.store_reward(reward)
    #        state = next_state
    #        total_reward += reward

    #        if done:
    #            break

    #    agent.update()
    #    print(f"Episode {episode}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
