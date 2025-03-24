###############################################################################
# Author: anthony.kiggundu@dfki.de
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
from a2c import ActorCritic, A2CAgent
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
###############################################################################

#from RenegeJockey import RequestQueue, Queues, Observations


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
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
        

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.rewards = []

    def select_action(self, state):
		
        if isinstance(state, dict):
            state = np.concatenate([state[key].flatten() for key in state.keys()])
            
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, state_value = self.model(state)
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

        returns = torch.tensor(returns).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.cat(self.values).to(self.device)
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
        duration = 3
        self.requestObj.run(duration)
        self.QueueObj = Queues()
        self.queue_id = self.requestObj.get_curr_queue_id()
        srv1, srv2 = self.requestObj.get_queue_sizes()

        if srv1 < srv2:
            low = srv1
            high = srv2
        else:
            low = srv2
            high = srv1            

        self.action_space = 2  # Number of discrete actions
        self.history = self.requestObj.get_history()
        serv_rate_one, serv_rate_two = self.requestObj.get_server_rates()

        self.observation_space = {
            "ServerID": (1, 2),
            "Renege": (0.0, 1.0),
            "ServRate": (1.0, np.inf),
            "Intensity": (0.0, np.inf),
            "Jockey": (0.0, 1.0),
            "Waited": (-1.0, np.inf),
            "EndUtility": (1.0, np.inf),
            "Reward": (0.0, 1.0),
            "QueueSize": (1.0, np.inf),
        }

        self._action_to_state = {
            Actions.RENEGE.value: self.get_renege_action_outcome(), 
            Actions.JOCKEY.value: self.get_jockey_action_outcome()
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        obs = {key: np.zeros(1) for key in self.observation_space.keys()}
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
        
        for item in self.requestObj.get_curr_obs_renege():
            hist_id = item.get("ServerID")
            size = item.get("QueueSize")           
            if "_reneged" in self.requestObj.customerid:
                if size == self.queuesize and hist_id == _id_:
                    reneged_matched_dict.append(item)
                    
        for item in self.requestObj.get_curr_obs_jockey():
            hist_id = item.get("ServerID")
            size = item.get("QueueSize")
            if "_jockeyed" in self.requestObj.customerid:      
                if size == self.queuesize and hist_id == _id_:            
                    jockeyed_matched_dict.update(item)                
                    
        return reneged_matched_dict, jockeyed_matched_dict

    def reset(self, seed=None, options=None):
        random.seed(seed)
        np.random.seed(seed)
        observation = self._get_obs()         
        info = self._get_info()                                 
        return observation, info

    def get_renege_action_outcome(self):
        curr_state = self.requestObj.get_queue_curr_state()
        srv = curr_state.get('ServerID')
        
        if srv == 1:
            if len(self.Observations.get_curr_obs_renege(srv)) > 0:
                curr_state['QueueSize'] = self.queuesize - 1
                curr_state["Reward"] = self.Observations.get_curr_obs_renege(srv)[0]['reward']
        else:
            if len(self.Observations.get_curr_obs_renege(srv)) > 0:
                curr_state['QueueSize'] = self.queuesize - 1
                curr_state["Reward"] = self.Observations.get_curr_obs_renege[0]['reward']
        return curr_state
        

    def get_jockey_action_outcome(self):
        curr_state = self.requestObj.get_queue_curr_state()
        srv = curr_state.get('ServerID')
        
        if srv == 1:
            if len(self.Observations.get_curr_obs_jockey(srv)) > 0:
                curr_state['QueueSize'] = self.queuesize + 1
                curr_state["Reward"] = self.Observations.get_curr_obs_jockey(srv)[0]['reward']
        else:
            if len(self.Observations.get_curr_obs_jockey(srv)) > 0:
                curr_state['QueueSize'] = self.queuesize + 1
                curr_state["Reward"] = self.Observations.get_curr_obs_jockey(srv)[0]['reward']
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
            "action_to_state": self._action_to_state
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

    def __init__(self,time_entrance,pos_in_queue=0,utility_basic=0.0,service_time=0.0,discount_coef=0.0, outage_risk=0.1, # =timer()
                 customerid="", learning_mode='online',min_amount_observations=1,time_res=1.0,markov_model=msm.StateMachine(orig=None),
                 exp_time_service_end=0.0, serv_rate=1.0, dist_local_delay=stats.expon,para_local_delay=[1.0,2.0,10.0], batchid=0 ):  #markov_model=a2c.A2C, 
        
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
        #self.certainty=float(outage_risk)


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
        # print("   User making reneging decision...")
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


class Observations:
    def __init__(self, reneged=False, serv_rate=0.0, queue_intensity=0.0, jockeyed=False, time_waited=0.0,end_utility=0.0, reward=0.0, queue_size=0): # reward=0.0, 
        self.reneged=reneged
        self.serv_rate = serv_rate
        self.queue_intensity = queue_intensity
        self.jockeyed=jockeyed
        self.time_waited=float(time_waited)
        self.end_utility=float(end_utility)
        self.reward= reward # id_queue
        self.queue_size=int(queue_size)
        self.obs = {} # OrderedDict() #{} # self.get_obs()  
        self.curr_obs_jockey = []
        self.curr_obs_renege = [] 

        return


    def set_obs (self, queue_id,  serv_rate, intensity, time_in_serv, activity, rewarded, curr_pose): # reneged, jockeyed,
        		
        if queue_id == "1": # Server1
            _id_ = 1
        else:
            _id_ = 2
			
        self.obs = {
			        "ServerID": _id_, #queue_id,
                    #"EndUtility":utility,
                    "Intensity":intensity,
                    #"Jockey":jockeyed,
                    "QueueSize": curr_pose,
                    #"Renege":reneged,
                    "Reward":rewarded,
                    "ServRate":serv_rate,
                    "Waited":time_in_serv,
                    "Action":activity,
                }
              

    def get_obs (self):
        
        return dict(self.obs)
        
        
    def set_renege_obs(self, curr_pose, queue_intensity, reneged,time_local_service, time_to_service_end, reward, queueid, activity):		

        self.curr_obs_renege.append(
            {   
                "queue": queueid,
                "at_pose": curr_pose,
                "reneged": reneged,
                "this_busy": queue_intensity,
                "expected_local_service":time_local_service,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity
            }
        )
        
        
    def get_renege_obs(self, queueid, queue): # , intensity, pose): # get_curr_obs_renege
		
        renegs = sum(1 for req in queue if '_reneged' in req.customerid)
        			    
        # print("\n Num Reneged in ", queueid, "is =", renegs)
	    
        return renegs #self.curr_obs_renege 
  
        
    def set_jockey_obs(self, curr_pose, queue_intensity, jockeyed, time_alt_queue, time_to_service_end, reward, queueid, activity):
        
        self.curr_obs_jockey.append(
            {
                "queue": queueid,
                "at_pose": curr_pose,
                "this_busy": queue_intensity,
                "jockeyed": jockeyed,
                "expected_local_service":time_alt_queue,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity   			
            }
        )
        
    
    def get_curr_obs_jockey(self, queueid): #, intensity, pose):
		
        return self.curr_obs_jockey
        
    
    def get_curr_obs_renege(self, queueid): #, intensity, pose):
		
        return self.curr_obs_renege
        

class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, utility_basic, discount_coef, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon, exp_time_service_end=0.0,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, batchid=np.int16):
                 
        
        self.dispatch_data = {
            "server_1": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": []},
            "server_2": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": []}
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

        self.arr_prev_times = np.array([])

        self.objQueues = Queues()
        self.objRequest = Request(time)
        self.objObserv = Observations()

        self.dict_queues_obj = self.objQueues.get_dict_queues()
        self.dict_servers_info = self.objQueues.get_dict_servers()
        self.jockey_threshold = 1
        self.reward = 0.0
        self.curr_state = {} # ["Busy","Empty"]

        self.arr_rate = 0.0 #self.objQueues.get_arrivals_rates()

        # self.objObserve = Observations()
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
        self.nn_subscribers = []  # Requests that use NN knowledge
        self.state_subscribers = []  # Requests that use raw queue state
        
        BROADCAST_INTERVAL = 5
        
        return
    
    def enqueue(self, request, use_nn=False):
        if len(self.queue) < self.capacity:
            self.queue.append(request)
            if use_nn:
                self.nn_subscribers.append(request)
            else:
                self.state_subscribers.append(request)
            return True
        return False
        
        
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
            #print("\n ID => ", hist.get('ServerID'), " GIVEN ID: ",queueid) integratedEffectiveFeature
            if str(hist.get('ServerID')) == str(queueid):
                lst_srv1.append(hist)
                #print("\n +++++ ", list(lst_srv1))
                return lst_srv1
            else:
                lst_srv2.append(hist)
                #print("\n **** ", list(lst_srv2)) position
                return lst_srv2
		
		   
    def get_queue_curr_state(self):
		
        if self.queueID == "1":
			# if len(self.get_matching_entries(self.queueID) > 0):
            self.curr_state = self.get_matching_entries(self.queueID)[-1]
        else:
            self.curr_state = self.get_matching_entries(self.queueID)[-1]
						
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


    def run(self,duration,progress_bar=True,progress_log=False):
        steps=int(duration/self.time_res)
    
        if progress_bar!=None:
            loop=tqdm(range(steps),leave=False,desc='     Current run')
        else:
            loop=range(steps)                 
        
        for i in loop:
            self.arr_rate = self.objQueues.get_arrivals_rates()
			
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
            
            print("\n Arrival rate: ", self.arr_rate)      
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
            self.processEntries(all_entries, i)
            self.time+=self.time_res
            
            
            # Ensure dispatch data is updated at each step
            self.dispatch_all_queues()
            
            self.set_batch_id(i)
            
        return
    
    
    def set_batch_id(self, id):
		
        self.batchid = id
		
		
    def get_batch_id(self):
		
        return self.batchid
	
		
    def get_all_service_times(self):
        
        return self.all_serv_times        
		

    def processEntries(self,entries=np.array([]), batchid=np.int16): # =np.int16
        
        num_iterations = random.randint(1, 5)  # Random number of iterations between 1 and 5
        
        for entry in entries:
            # print("Processing a new request entry...")
            #self.time=entry[0]            
            if entry[1]==True:
                # print("  Adding a new request into task queue...")                

                req = self.addNewRequest(entry[0], batchid)
                self.arr_prev_times = np.append(self.arr_prev_times, entry[0])
                
            else:                
                q_selector = random.randint(1, 2)
                # observer = {}
                
                if q_selector == 1:					
                    self.queueID = "1" # Server1
                    
                    """
                         Run the serveOneRequest function a random number of times before continuing.
                    """
                    
                    #for _ in range(num_iterations):
                        #self.serveOneRequest(to_delete, time_entrance, queueID)
                        # Introduce a short delay to simulate processing time            
                    self.serveOneRequest(self.queueID) # Server1 = self.dict_queues_obj["1"][0], entry[0],                                                                  
                    ## self.dispatch_queue_state(self.dict_queues_obj["1"], self.queueID, self.dict_queues_obj["2"]) #, req)
                    time.sleep(random.uniform(0.1, 0.5))  # Random delay between 0.1 and 0.5 seconds
                        # time.sleep(1)
                    
                    #if self.capacity is not None and len(self.dict_queues_obj["1"]) >= self.capacity:
                    #    raise Exception("Queue has reached its capacity limit")
                        # sys.exit(1)
                        #return 
                        
                else:
					#req = self.dict_queues_obj["2"][0]
                    self.queueID = "2"
                    #for _ in range(num_iterations):
                    self.serveOneRequest(self.queueID) # Server2 = self.dict_queues_obj["2"][0], entry[0],                                       
                    ## self.dispatch_queue_state(self.dict_queues_obj["2"], self.queueID, self.dict_queues_obj["1"]) #, req)
                    time.sleep(random.uniform(0.1, 0.5))
                    
                    #if self.capacity is not None and len(self.dict_queues_obj["2"]) >= self.capacity:
                    #    raise Exception("Queue has reached its capacity limit")
                        # sys.exit(1)
                        #return 
                        
                # print("  Wait to Broadcasting the updated queue information...")                
                #self.broadcastQueueInfo()
                    
                    
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
               

    def addNewRequest(self, expected_time_to_service_end, batchid): #, time_entered):
        # Join the shorter of either queues
               
        lengthQueOne = len(self.dict_queues_obj["1"]) # Server1
        lengthQueTwo = len(self.dict_queues_obj["2"]) # Server1 
        rate_srv1,rate_srv2 = self.get_server_rates()
        
        # self.set_customer_id()       

        if lengthQueOne < lengthQueTwo:
            time_entered = self.time   #self.estimateMarkovWaitingTime(lengthQueOne) ID
            pose = lengthQueOne+1
            server_id = "1" # Server1
            self.customerid = self.get_customer_id()
            self.customerid = "Batch"+str(self.get_batch_id())+"_"+self.customerid
            queue_intensity = self.arr_rate/rate_srv1
            #expected_time_to_service_end = self.estimateMarkovWaitingTime(float(pose)) # , queue_intensity, time_entered)
            #time_local_service = self.generateLocalCompUtility(req)

        else:
            pose = lengthQueTwo+1
            server_id = "2" # Server2
            self.customerid = self.get_customer_id()
            self.customerid = "Batch"+str(self.get_batch_id())+"_"+self.customerid
            time_entered = self.time #self.estimateMarkovWaitingTime(lengthQueTwo)
            queue_intensity = self.arr_rate/rate_srv2
            #expected_time_to_service_end = self.estimateMarkovWaitingTime(float(pose)) #, queue_intensity, time_entered)
            #time_local_service = self.generateLocalCompUtility(req)
            
            
        req=Request(time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid) # =self.batchid
                    
        # #markov_model=self.markov_model,  
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
        
    
    def get_queue_observation(self, queue_id):
        """Returns queue observations formatted for Actor-Critic training."""
        
        state = self.get_queue_state(queue_id)
        observation = {
            "total_customers": state["total_customers"],
            "intensity": state["intensity"],
            "capacity": state["capacity"],
            "long_avg_serv_time": state["long_avg_serv_time"]
        }
        
        return observation

  
    def dispatch_queue_state(self, curr_queue, curr_queue_id, alt_queue):
		
        rate_srv1,rate_srv2 = self.get_server_rates()
		
        if curr_queue_id == "1":
            alt_queue_id = "2"
            serv_rate = rate_srv1
        else:
            alt_queue_id = "1"
            serv_rate = rate_srv2

        curr_queue_state = self.get_queue_state(alt_queue_id)

        # Compute reneging rate and jockeying rate
        reneging_rate = self.compute_reneging_rate(curr_queue)
        jockeying_rate = self.compute_jockeying_rate(curr_queue)

        # Append these rates to the state
        curr_queue_state['reneging_rate'] = reneging_rate
        curr_queue_state['jockeying_rate'] = jockeying_rate

        for client in range(len(curr_queue)):
            req = self.dict_queues_obj[curr_queue_id][client]
            # print(f"Dispatching state of server {alt_queue_id} to client {req.customerid} : {curr_queue_state}.")
            
            if curr_queue_id == "1":
                self.makeJockeyingDecision(req, curr_queue_id, "2", req.customerid, serv_rate)
                self.makeRenegingDecision(req, curr_queue_id)
            else:
                self.makeJockeyingDecision(req, curr_queue_id, "1", req.customerid, serv_rate)
                self.makeRenegingDecision(req, curr_queue_id)

        return reneging_rate, jockeying_rate
        
    
    def get_nn_optimized_decision(self, req):
        """Uses Actor-Critic to get an action for NN-subscribed requests."""
        state = self.get_queue_observation(req.queue_id)
        state_tensor = torch.tensor(list(state.values()), dtype=torch.float32).to(device)
    
        action, _, _, _ = self.select_action(state_tensor)
        
        return {"action": action.cpu().numpy(), "nn_based": True}
    
    
    def dispatch(self, use_nn):
        """
        Dispatch state-action information or raw server status based on the use_nn flag.
        """
        if use_nn:
            self.dispatch_nn_based_requests()
        else:
            self.dispatch_raw_server_status_requests()
            

    def dispatch_nn_based_requests(self):
        """
        Dispatch state-action information to NN-based subscribers.
        """
        for req in self.nn_subscribers:
            state_action_info = env.get_state_action_info()
            # Add logic to send state_action_info to the request
            print(f"Dispatching state-action info to request {req.customerid}: {state_action_info}")
            

    def dispatch_raw_server_status_requests(self):
        """
        Dispatch raw server status information to state-based subscribers.
        """
        for req in self.state_subscribers:
            raw_server_status = env.get_raw_server_status()
            # Add logic to send raw_server_status to the request
            print(f"Dispatching raw server status to request {req.customerid}: {raw_server_status}")
       
    
    def dispatch_all_queues(self):
        """
        Dispatch the status of all queues and collect jockeying and reneging rates.
        """
        for queue_id in ["1", "2"]:
            curr_queue = self.dict_queues_obj[queue_id]
            alt_queue_id = "2" if queue_id == "1" else "1"
            alt_queue = self.dict_queues_obj[alt_queue_id]
            reneging_rate, jockeying_rate = self.dispatch_queue_state(curr_queue, queue_id, alt_queue)
            serv_rate = self.dict_servers_info[queue_id]
            num_requests = len(curr_queue)

            self.dispatch_data[f"server_{queue_id}"]["num_requests"].append(num_requests)
            self.dispatch_data[f"server_{queue_id}"]["jockeying_rate"].append(jockeying_rate)
            self.dispatch_data[f"server_{queue_id}"]["reneging_rate"].append(reneging_rate)
            self.dispatch_data[f"server_{queue_id}"]["service_rate"].append(serv_rate)
            
            print(f"Server {queue_id} - Num requests: {num_requests}, Jockeying rate: {jockeying_rate}, Reneging rate: {reneging_rate}, Service rate: {serv_rate}")   
            
        self.dispatch(use_nn=True)  # Dispatch NN-based information
        self.dispatch(use_nn=False)  # Dispatch raw server status information             
            
        
    def setup_dispatch_intervals(self):
        """
        Set up the intervals for dispatching the queue status information.
        """
        # schedule.every(10).seconds.do(self.dispatch_all_queues)
        # schedule.every(30).seconds.do(self.dispatch_all_queues)
        schedule.every(60).seconds.do(self.dispatch_all_queues)       
            
            
    def plot_rates(self):
        """
        Plot the jockeying and reneging rates over time.
        """       
        #print("\n Data: ", self.dispatch_data)
        
        num_requests_1 = self.dispatch_data["server_1"]["num_requests"]
        num_requests_2 = self.dispatch_data["server_2"]["num_requests"]
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        axs[0].plot(num_requests_1, self.dispatch_data["server_1"]["jockeying_rate"], label='Server 1 Jockeying Rate')
        axs[0].plot(num_requests_1, self.dispatch_data["server_1"]["reneging_rate"], label='Server 1 Reneging Rate')
        axs[0].plot(num_requests_1, self.dispatch_data["server_1"]["service_rate"], label='Server 1 Service Rate')
        axs[0].set_title('Server 1 Rates')
        axs[0].set_ylabel('Rate')
        axs[0].legend()
        
        axs[1].plot(num_requests_2, self.dispatch_data["server_2"]["jockeying_rate"], label='Server 2 Jockeying Rate')
        axs[1].plot(num_requests_2, self.dispatch_data["server_2"]["reneging_rate"], label='Server 2 Reneging Rate')
        axs[1].plot(num_requests_2, self.dispatch_data["server_2"]["service_rate"], label='Server 2 Service Rate')
        axs[1].set_title('Server 2 Rates')
        axs[1].set_xlabel('Number of Requests')
        axs[1].set_ylabel('Rate')
        axs[1].legend()        
        
        plt.show()
        
    
    def setup_dispatch_intervals(self):
        """
        Set up the intervals for dispatching the queue status information.
        """
        # schedule.every(10).seconds.do(self.dispatch_all_queues)
        # schedule.every(30).seconds.do(self.dispatch_all_queues)
        schedule.every(60).seconds.do(self.dispatch_all_queues)       
            

    def run_scheduler(self):
        """
        Run the scheduler to dispatch queue status at different intervals.
        """
        self.setup_dispatch_intervals()
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    
    def get_long_run_avg_service_time(self, queue_id):
		
        total_service_time = 0        
    
        if queue_id == "1":
            for req in self.dict_queues_obj["1"]:                
                total_service_time += req.service_time # exp_time_service_end                
                return total_service_time / self.total_served_requests_srv1
        else:
            for req in self.dict_queues_obj["2"]:                
                total_service_time += req.service_time  # exp_time_service_end                
                return total_service_time / self.total_served_requests_srv2
    
        # print("\n WHAT IS HERE?? ", total_service_time, " === ", self.total_served_requests_srv1, " *** ", self.total_served_requests_srv2)
        if self.total_served_requests_srv1 == 0 or self.total_served_requests_srv2 == 0:
            return 0
    
                       
    def get_queue_state(self, queueid):
		
        rate_srv1,rate_srv2 = self.get_server_rates()        
		
        if queueid == "1":		
            queue_intensity = self.objQueues.get_arrivals_rates()/ rate_srv1    
            customers_in_queue = self.dict_queues_obj["1"]   
            #renege_rate = self.get_renege_rate( queueid)                
            
            state = {
                "total_customers": len(customers_in_queue),
                "intensity": queue_intensity,
                "capacity": self.capacity,
                #"renege_rate": renege_rate,
                #"jockey_rate": jockey_rate,
                "long_avg_serv_time": self.get_long_run_avg_service_time(queueid)
            }
        else:
			#serv_rate = self.get_server_rates()[1] #dict_servers_info["2"] 
            queue_intensity = self.objQueues.get_arrivals_rates()/ rate_srv2            
            customers_in_queue = self.dict_queues_obj["2"]
            #renege_rate = self.get_renege_rate( queueid)
			
            state = {
                "total_customers": len(customers_in_queue),
                "intensity": queue_intensity,
                "capacity": self.capacity,
                #"renege_rate": renege_rate,
                #"jockey_rate": jockey_rate,
                "long_avg_serv_time": self.get_long_run_avg_service_time(queueid)
                
            }
            
        return state


    def serveOneRequest(self, queueID): # to_delete, serv_time, 
        #randomly select which queue to process at a time t+1
        q_selector = random.randint(1, 2)                            
        
        # ToDo:: run the processing of queues for some specific interval of time 
        # before admitting more into the queue
        len_queue_1,len_queue_2 = self.get_queue_sizes()
        
        if "1" in queueID:   # Server1               
            req =  self.dict_queues_obj["1"][0] # Server1
            serv_rate = self.dict_servers_info["1"] # Server1
            queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
            queueID = "1" # Server1
    
            reward = self.get_jockey_reward(req)       
            # serve request in queue            
                    
            self.queueID = queueID
            self.dict_queues_obj["1"] = self.dict_queues_obj["1"][1:self.dict_queues_obj["1"].size]       # Server1 
            self.total_served_requests_srv2+=1                       
            
            # Set the exit time
            req.time_exit = self.time              
            
            # take note of the observation ... self.time  queue_id,  serv_rate, intensity, time_in_serv, activity, rewarded, curr_pose
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, req.time_exit-req.time_entrance, reward, len_queue_1, 2)   # req.exp_time_service_end,                                    
            self.history.append(self.objObserv.get_obs())
                
            #time_to_service_end = self.estimateMarkovWaitingTime(float(curr_pose), queue_intensity, reqObj.time_entrance)
            #time_local_service = self.generateLocalCompUtility(req)				                           
                                
            self.arr_prev_times = self.arr_prev_times[1:self.arr_prev_times.size]
            
            self.objQueues.update_queue_status(queueID)
            # req, curr_queue_id, alt_queue_id, customerid, serv_rate
            # Any Customers interested in jockeying or reneging when a request is processed get_curr_obs_jockey
            #print("\n Inside Server 1, calling the  decision procedures")
            
            '''
                Now after serving a request, dispatch the new state of the queues
            '''
            
            # self.makeJockeyingDecision(req, self.queueID, "2", req.customerid, serv_rate)
            # self.makeRenegingDecision(req, self.queueID)

        else:                        
            req = self.dict_queues_obj["2"][0] # Server2
            serv_rate = self.dict_servers_info["2"] # Server2
            queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
            queueid = "2"   # Server2  
            
            #req.service_time = serv_time   
            #print("\n SERV T IN SERV 2: ", req.service_time)    
                        
            self.dict_queues_obj["2"] = self.dict_queues_obj["2"][1:self.dict_queues_obj["2"].size] # Server2
            
            reward = self.get_jockey_reward(req)
         
            self.queueID = queueID 
            self.dict_queues_obj["S2"] = self.dict_queues_obj["2"][1:self.dict_queues_obj["2"].size]      # Server2 
            self.total_served_requests_srv1+=1
            
            #print("\n ==> ", self.total_served_requests_srv1)
            
            # Set the exit time
            req.time_exit = self.time                 
            
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, req.time_exit-req.time_entrance, reward, len_queue_2, 2)    # req.exp_time_service_end,                                  
            self.history.append(self.objObserv.get_obs())                   
               
            #if time_local_service < time_to_service_end:   
		    #   reqObj.customerid = reqObj.customerid+"_reneged"                 
            #    self.reqRenege(reqObj, queueID, curr_pose, serv_rate, queue_intensity, time_local_service, reqObj.customerid) #, time_to_service_end) #self.queue[0].id)
            #    self.queueID = queueID
            #    self.setCurrQueueState(self.queueID, serv_rate, reward, time_local_service)                   
                                        
            #else:               
            #    reward = 0.0          
            #    self.queueID = queueID          
            #    self.objObserv.set_obs(self.queueID, False, serv_rate, queue_intensity, False,self.time-req.time_entrance,time_to_service_end, reward, len_queue_2)                                      
            #    self.history.append(self.objObserv.get_obs())
            #    self.setCurrQueueState(self.queueID, serv_rate, reward, time_to_service_end)                                       
                                    
            self.arr_prev_times = self.arr_prev_times[1:self.arr_prev_times.size]  
            self.objQueues.update_queue_status(queueID)
            
            # Any Customers interested in jockeying or reneging when a request is processed
            #print("\n Inside Server 2, calling the  decision procedures")
            # self.makeJockeyingDecision(req, self.queueID, "1", req.customerid, serv_rate)# Server1
            # self.makeRenegingDecision(req, self.queueID)
        
        self.curr_req = req
                                                                  
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
                
                
            #print("\n ==>> LOCAL and CLOUD: ", self.max_local_delay," ==== ",self.max_cloud_delay)
            
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
                    self.reqRenege( req, queueid, curr_pose, serv_rate, queue_intensity, self.max_local_delay, req.customerid, req.service_time, decision, self.queue)
                                
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
        
        if "Server1" in queueid:
            self.queue = self.dict_queues_obj["1"]            
        else:
            self.queue = self.dict_queues_obj["2"] 
            
        if curr_pose >= len(curr_queue):
            return
            
        else:
            self.queue = np.delete(self.queue, curr_pose) # index)        
            self.queueID = queueid  
        
            req.customerid = req.customerid+"_reneged"
        
        # In the case of reneging, you only get a reward if the time.entrance plus
        # the current time minus the time _to_service_end is greater than the time_local_service
        
            reward = self.getRenegeRewardPenalty(req, time_local_service, time_to_service_end)
        # self.objObserv.set_obs(queueid, True, serv_rate, queue_intensity, False,self.time-req.time_entrance,self.generateLocalCompUtility(req), reward, len(self.queue))

        #for t in range(len(self.queue)):
        #    if self.queue[t].customerid == customerid:self.arr_rate
        #        curr_pose = t                                       
        
            self.objObserv.set_renege_obs(curr_pose, queue_intensity, decision,time_local_service, time_to_service_end, reward, queueid, "reneged")
        
            # self.curr_obs_renege.append(self.objObserv.get_renege_obs(queueid, self.queue)) #queueid, queue_intensity, curr_pose))
            self.curr_obs_renege.append(self.objObserv.get_curr_obs_renege(queueid))      
        
            self.curr_req = req
        
            self.objQueues.update_queue_status(queueid)


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
		
        from termcolor import colored
                
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
            
            self.objObserv.set_jockey_obs(curr_pose, queue_intensity, decision, exp_delay, req.exp_time_service_end, reward, 1.0, "jockeyed") # time_alt_queue        
            # self.curr_obs_jockey.append(self.objObserv.get_jockey_obs(curr_queue_id, queue_intensity, curr_pose))
            self.curr_obs_jockey.append(self.objObserv.get_curr_obs_jockey(curr_queue_id))                                      
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
        queue_intensity = self.arr_rate/self.dict_servers_info[alt_queue_id]
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
    env = ImpatientTenantEnv()
    duration = 10
    
    # Start the scheduler
    scheduler_thread = threading.Thread(target=requestObj.run_scheduler)
    scheduler_thread.start()
        
    requestObj.run(duration)
    
    state_dim = len(env.observation_space)
    action_dim = env.action_space

    agent = A2CAgent(state_dim, action_dim)

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
