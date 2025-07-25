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
import threading
from tqdm import tqdm
import MarkovStateMachine as msm
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from math import exp, factorial

# --- Add for predictive modeling ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

###############################################################################

# --- PredictiveModel: Learns relationships between queue/server state and user actions ---
class PredictiveModel:
    """
    Predictive model to estimate probabilities of user actions (wait, renege, jockey)
    based on observed queue/server state at dispatch.
    """
    ACTIONS = ["wait", "renege", "jockey"]
    ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
    
    def __init__(self):
        # self.model = LogisticRegression(multi_class='multinomial', max_iter=200)
        self.model = LogisticRegression(max_iter=200)
        self.scaler = StandardScaler()
        self.X = []
        self.y = []
        self.is_fitted = False
        self.fit_history = []  # <--- model fit history
        

    def record_observation(self, features, reaction):
        """features: 1D array-like, reaction: str from ACTIONS"""
        self.X.append(features)
        self.y.append(self.ACTION_IDX[reaction])
        

    def fit(self):
        #if len(self.X) > 15:
        # Only fit if enough samples and at least two classes present
        if len(self.X) > 15 and len(set(self.y)) >= 2:
            X_scaled = self.scaler.fit_transform(np.array(self.X))
            self.model.fit(X_scaled, np.array(self.y))
            self.is_fitted = True
            # Save diagnostics info for this fit
            self.fit_history.append({
                "n_samples": len(self.X),
                "coef_": self.model.coef_.copy(),
                "intercept_": self.model.intercept_.copy(),
            })
            

    def get_fit_history(self):
        return self.fit_history
        

    def predict_proba(self, features):
        """Return probability vector for each action given features."""
        if self.is_fitted:
            try:
                X_scaled = self.scaler.transform([features])
                proba_out = self.model.predict_proba(X_scaled)[0]
                proba = np.zeros(len(self.ACTIONS))
                for idx, cls in enumerate(self.model.classes_):
                    if isinstance(cls, str):
                        proba[self.ACTION_IDX[cls]] = proba_out[idx]
                    else:
                        proba[cls] = proba_out[idx]
                return proba
            except AttributeError:
                # Scaler/model not fit yet
                return np.ones(len(self.ACTIONS)) / len(self.ACTIONS)
        else:
            # Uniform probabilities if not enough data
            return np.ones(len(self.ACTIONS)) / len(self.ACTIONS)
            

    def most_likely_action(self, features):
        proba = self.predict_proba(features)
        idx = np.argmax(proba)
        return self.ACTIONS[idx], proba[idx]


# --- ServerPolicy: Adjusts service rates based on predictions and utility maximization ---
class ServerPolicy:
    """
    Server policy that adapts service rate to optimize a utility function
    based on predicted probabilities of user actions.
    """
    def __init__(self, predictive_model, min_rate=1.0, max_rate=15.0):
        self.model = predictive_model
        self.current_service_rate = min_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        # Utility function weights
        self.w_wait = 1.0      # Reward for waiting (good for server)
        self.w_renege = -2.0   # Penalty for reneging (bad for server)
        self.w_jockey = -1.0   # Penalty for jockeying (loss of customer)
        self.history = []

    def utility(self, proba):
        """Expected utility given predicted action probabilities."""
        return (self.w_wait * proba[0] +
                self.w_renege * proba[1] +
                self.w_jockey * proba[2])

    def update_policy(self, queue_state_features):
        """
        Update service rate to maximize expected utility based on model prediction.
        """
        proba = self.model.predict_proba(queue_state_features)
        # print("\n --> ", proba)
        util = self.utility(proba)
        prev_rate = self.current_service_rate

        # Simple rule: if utility is low (many reneges/jockeys), increase rate; else decrease
        if util < 0:
            self.current_service_rate = min(self.current_service_rate * 1.15, self.max_rate)
        elif util > 0.2:
            self.current_service_rate = max(self.current_service_rate * 0.95, self.min_rate)
        # else small/no adjustment

        self.history.append({
            "features": queue_state_features,
            "proba": proba,
            "utility": util,
            "prev_rate": prev_rate,
            "new_rate": self.current_service_rate
        })
        return self.current_service_rate
        
        
    def get_policy_history(self):
        return self.history



class Queues(object):
    def __init__(self):
        super().__init__()
        
        self.num_of_queues = 2
        self.dict_queues = {}
        self.dict_servers = {}
        self.arrival_rates = [3,4,5,6,7,8,9,10,11,12,13,14,15,17]
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
        deltaLambda=random.uniform(0.1, 0.9)
        
        serv_rate_one=self.sampled_arr_rate + deltaLambda 
        serv_rate_two=self.sampled_arr_rate - deltaLambda

        _serv_rate_one=serv_rate_one / 2
        _serv_rate_two=serv_rate_two / 2
                
        self.dict_servers["1"] = _serv_rate_one # Server1
        self.dict_servers["2"] = _serv_rate_two # Server2
        
        # print("\n Current Arrival Rate:", self.sampled_arr_rate, "Server1:", _serv_rate_one, "Server2:", _serv_rate_two, " Lambda: ",deltaLambda) 


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
                 exp_time_service_end=0.0, serv_rate=1.0, dist_local_delay=stats.expon,para_local_delay=[0.0,0.05,1.0], batchid=0 ):  #markov_model=a2c.A2C, 
        
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
        self.service_time = service_time # self.objQueues.get_dict_servers()
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
        self.loc_local_delay=np.random.uniform(low=float(para_local_delay[0]),high=(para_local_delay[1])) # 0 and 1
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
    def __init__(self, reneged=False, serv_rate=0.0, jockeyed=False, time_waited=0.0,end_utility=0.0, reward=0.0, queue_size=0): # reward=0.0, queue_intensity=0.0,
        self.reneged=reneged
        self.serv_rate = serv_rate
        #self.queue_intensity = queue_intensity
        self.jockeyed=jockeyed
        self.time_waited=float(time_waited)
        self.end_utility=float(end_utility)
        self.reward= reward # id_queue
        self.queue_size=int(queue_size)
        self.obs = {} # OrderedDict() #{} # self.get_obs()  
        self.curr_obs_jockey = []
        self.curr_obs_renege = [] 

        return


    def set_obs (self, queue_id,  serv_rate, intensity, time_in_serv, activity, rewarded, curr_pose, req, queue_length): # reneged, jockeyed,
        		
        if queue_id == "1": # Server1
            _id_ = 1
        else:
            _id_ = 2
			
        self.obs = {
			        "ServerID": _id_, #queue_id,
                    "customerid": req.customerid,
                    #"Intensity":intensity,
                    #"Jockey":jockeyed,
                    "QueueSize": curr_pose,
                    #"Renege":reneged,
                    "Reward":rewarded,
                    "ServRate":serv_rate,
                    "Waited":time_in_serv,
                    "Action":activity,
                    "queue_length": queue_length
                }
              

    def get_obs (self):
        
        return dict(self.obs)
        
        
    def set_renege_obs(self, curr_pose, reneged,time_local_service, time_to_service_end, reward, queueid, activity, queue_length):		

        self.curr_obs_renege.append(
            {   
                "queue": queueid,
                "at_pose": curr_pose,
                "reneged": reneged,
                #"this_busy": queue_intensity,
                "expected_local_service":time_local_service,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity,
                "queue_length": queue_length
            }
        )
        
        
    def get_renege_obs(self, queueid, queue): # , intensity, pose): # get_curr_obs_renege
		
        renegs = sum(1 for req in queue if '_reneged' in req.customerid)        			         
	    
        return renegs # self.curr_obs_renege 
  
        
    def set_jockey_obs(self, curr_pose, jockeyed, time_alt_queue, time_to_service_end, reward, queueid, activity, queue_length):
        
        self.curr_obs_jockey.append(
            {
                "queue": queueid,
                "at_pose": curr_pose,
                #"this_busy": queue_intensity,
                "jockeyed": jockeyed,
                "expected_local_service":time_alt_queue,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity,
                "queue_length": queue_length  			
            }
        )
        
    
    def get_jockey_obs(self, queueid, intensity, pose):
		
        return self. curr_obs_jockey
           
    
class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, utility_basic, discount_coef, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon, exp_time_service_end=0.0,
                 para_local_delay=[0.01,0.1,1.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, 
                 batchid=np.int16, policy_enabled=True, seed=None):
                 
        
        self.dispatch_data = {}
        #self.dispatch_data = {
        #    "server_1": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []},
        #    "server_2": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []}
        #}
        
        self.markov_model=msm.StateMachine(orig=markov_model)
        # self.arr_rate=float(arr_rate) arr_rate, queue=np.array([])
        ## self.customerid = self.set_customer_id()
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
        # self.curr_state = {} # ["Busy","Empty"]

        self.arr_rate = None #self.objQueues.get_arrivals_rates()

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
            raise ValueError('Invalid outage risk threshold! Please select between (0,1)')
        
        self.exp_time_service_end = exp_time_service_end
        #self.dist_local_delay=dist_local_delay
        self.loc_local_delay=np.random.uniform(low=float(para_local_delay[0]),high=(para_local_delay[1]))
        self.scale_local_delay=float(para_local_delay[2])
        self.max_local_delay=self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay)
        self.max_cloud_delay= np.inf 
                
        self.error_loss=1
        
        self.capacity = self.objQueues.get_queue_capacity()
        self.total_served_requests_srv1 = 0
        self.total_served_requests_srv2 = 0 
        self.srvrates_1 = None
        self.srvrates_2 = None                           
        
        BROADCAST_INTERVAL = 5        
        
        self.policy_enabled = policy_enabled
        self.predictive_model = PredictiveModel()
        #self.policy = ServerPolicy(self.predictive_model, min_rate=1.0, max_rate=15.0)
        self.policy1 = ServerPolicy(self.predictive_model, min_rate=1.0, max_rate=15.0)
        self.policy2 = ServerPolicy(self.predictive_model, min_rate=1.0, max_rate=15.0)
        
        self.interval_stats = {
            "reneging_rate": {"server_1": [], "server_2": []},
            "jockeying_rate": {"server_1": [], "server_2": []}
        }
        
        # Schedule the dispatch function to run every minute (or any preferred interval)
        # schedule.every(1).minutes.do(self.dispatch_queue_state, queue=queue_1, queue_name="Server1")
        # schedule.every(1).minutes.do(dispatch_queue_state, queue=queue_2, queue_name="Server2")
        
        # Start the scheduler     
        #scheduler_thread = threading.Thread(target=self.run_scheduler)
        #scheduler_thread.start()
        seed=None
        
        return
    
    
    def record_interval_rates(self):
        for server_id in ["1", "2"]:
            server_label = f"server_{server_id}"
            curr_queue = self.dict_queues_obj[server_id]
            ren_rate = self.compute_reneging_rate(curr_queue)
            jky_rate = self.compute_jockeying_rate(curr_queue)
            self.interval_stats["reneging_rate"][server_label].append(ren_rate)
            self.interval_stats["jockeying_rate"][server_label].append(jky_rate)
            
        
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
        

    def get_service_rates(self, queue_id):
		
        return self.srvrates_1 if "1" in queue_id else self.srvrates_2
        #srvrate1 = self.dict_servers_info.get("1")# Server1
        #srvrate2 = self.dict_servers_info.get("2") # Server2

        #return [srvrate1, srvrate2] # srvrate1, srvrate2]


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
        
    
    def dispatch_timer(self, interval):
        """
        Timer-based dispatch function to log queue state at fixed intervals.
        """
        while self.running:
            print(f"Dispatching queue state at interval: {interval} seconds")
            for queue_id in ["1", "2"]:
                curr_queue = self.dict_queues_obj[queue_id]

                alt_queue_id = "2" if "1" in queue_id  else "1"
                alt_queue = self.dict_queues_obj[alt_queue_id]         
            
                # Compute rates
                #jockeying_rate = self.compute_jockeying_rate(curr_queue)
                #reneging_rate = self.compute_reneging_rate(curr_queue)                                
                
                # curr_queue_state = self.get_queue_state(queue_id)
                
                #print("\n\n AFTER get_queue_state:", id(curr_queue_state), curr_queue_state)
                 
                #curr_queue_state["jockeying_rate"] = jockeying_rate
                #curr_queue_state["reneging_rate"] = reneging_rate  
                
                #curr_queue, queue_id, alt_queue, alt_queue_id, interval
                self.dispatch_queue_state( curr_queue, queue_id, alt_queue, alt_queue_id, interval) #, curr_queue_state)
                       
                # Record the statistics
                #if "1" in queue_id:
                #    serv_rate = self.srvrates_1
                #else:
                #    serv_rate = self.srvrates_2
					                
                #num_requests = len(curr_queue)
            
                # Append rates to interval-specific dispatch data
                #self.dispatch_data[interval][f"server_{queue_id}"]["num_requests"].append(num_requests)
                #self.dispatch_data[interval][f"server_{queue_id}"]["jockeying_rate"].append(jockeying_rate)
                #self.dispatch_data[interval][f"server_{queue_id}"]["reneging_rate"].append(reneging_rate)
                #self.dispatch_data[interval][f"server_{queue_id}"]["service_rate"].append(serv_rate)
            
                #self.dispatch_data[f"server_{queue_id}"]["num_requests"].append(num_requests)
                #self.dispatch_data[f"server_{queue_id}"]["jockeying_rate"].append(jockeying_rate)
                #self.dispatch_data[f"server_{queue_id}"]["reneging_rate"].append(reneging_rate)
                #self.dispatch_data[f"server_{queue_id}"]["service_rate"].append(serv_rate)
                #self.dispatch_data[f"server_{queue_id}"]["intervals"].append(interval)

                #print(f"Server {queue_id} - Num requests: {num_requests}, Jockeying rate: {jockeying_rate}, "
                #     f"Reneging rate: {reneging_rate}, Service rate: {serv_rate}, Long-run rate: {curr_queue_state['long_run_change_rate']}")
                 
            # Reset state at the end of the interval
            # self.reset_state()
            time.sleep(interval)  # Wait for the next interval


    # Example: feature extraction for predictive modeling
    def extract_features(self, queue_id):
        """
        Feature vector: [queue_length, arrival_rate, service_rate, queue_intensity,
                         avg_waiting_time, reneging_rate, jockeying_rate, sample_interchange_time]
        Can be extended with more features.
        """
        queue = self.dict_queues_obj[queue_id]
        queue_length = len(queue)
        arrival_rate = self.arr_rate
        service_rate = self.get_service_rates(queue_id) # self.srvrates_1 if queue_id == "1" else self.srvrates_2 get_server_rates
        queue_intensity = (arrival_rate / service_rate) if service_rate > 0 else 0
        waiting_times = getattr(self, f"waiting_times_srv{queue_id}", [])
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
        curr_queue_state = self.get_queue_state(queue_id)
        reneging_rate = curr_queue_state["reneging_rate"]
        jockeying_rate = curr_queue_state["jockeying_rate"]
        sample_interchange_time = curr_queue_state["sample_interchange_time"]
        steady_state_distribution = curr_queue_state["steady_state_distribution"]
        
        return [
            queue_length,
            arrival_rate,
            service_rate,
            queue_intensity,
            avg_waiting_time,
            reneging_rate,
            jockeying_rate,
            sample_interchange_time,
            steady_state_distribution,
        ]
        
        #["max"],
        #steady_state_distribution["min"],
        #steady_state_distribution["mean"],

    # After each dispatch or user action, record observation and update model/policy
    def record_user_reaction(self, queue_id, action_label):
        """
        Call this after observing a user action (wait, renege, jockey).
        """
        features = self.extract_features(queue_id)
        self.predictive_model.record_observation(features, action_label)
        self.predictive_model.fit()
        # Optionally: immediately update policy based on new prediction
        if queue_id == "1":
            self.policy1.update_policy(features)
        else:
            self.policy2.update_policy(features)


    def run(self,duration, interval, progress_bar=True,progress_log=False):
        steps=int(duration/self.time_res)

        if progress_bar!=None:
            loop=tqdm(range(steps),leave=False,desc='     Current run')
        else:
            loop=range(steps)                
        
        self.running = True  # Flag to control threads

        # Start dispatch timer in a separate thread
        dispatch_thread = threading.Thread(target=self.dispatch_timer, args=(interval,))
        dispatch_thread.start()
        
        try:  
            for i in loop:
            
                # Randomize arrival rate and service rates at each iteration
                self.arr_rate = self.objQueues.randomize_arrival_rate()  # Randomize arrival rate
                # self.objQueues.get_dict_servers
                deltaLambda=random.randint(1, 2)
                ##deltaLambda=random.uniform(0.1, 0.9)
        
                serv_rate_one = self.arr_rate + deltaLambda 
                serv_rate_two = self.arr_rate - deltaLambda

                self.srvrates_1 = serv_rate_one / 2
                #self.srvrates_1 = self.objQueues.get_dict_servers()["1"]
                self.srvrates_2 = serv_rate_two / 2
                #self.srvrates_2 = self.objQueues.get_dict_servers()["2"]
             
                srv_1 = self.dict_queues_obj.get("1") # Server1
                srv_2 = self.dict_queues_obj.get("2") 
                print("\n Arrival rate: ", self.arr_rate, "Rates 1: ----", self.srvrates_1,  "Rates 2: ----", self.srvrates_2)
                
                
                # --- Predictive modeling: adjust service rates using learned policy ---
                #features_srv1 = self.extract_features("1")
                #features_srv2 = self.extract_features("2")
                # Update server rates based on current policy/model
                #self.srvrates_1 = self.policy.update_policy(features_srv1)
                #self.srvrates_2 = self.policy.update_policy(features_srv2)
                
                features_srv1 = self.extract_features("1")
                features_srv2 = self.extract_features("2")
                
                if getattr(self, "policy_enabled", True):
                    self.srvrates_1 = self.policy1.update_policy(features_srv1)
                    self.srvrates_2 = self.policy2.update_policy(features_srv2)
                    #self.srvrates_1 = self.policy.update_policy(features_srv1)
                    #self.srvrates_2 = self.policy.update_policy(features_srv2)
                    
                
               
                if progress_log:
                    print("Step",i,"/",steps)                 

                if len(srv_1) < len(srv_2):
                    self.queue = srv_2
                    srv_rate = self.srvrates_1 # self.dict_servers_info.get("2") # Server2                            

                else:            
                    self.queue = srv_1
                    srv_rate = self.srvrates_2 # self.dict_servers_info.get("1") # Server1                                
                              
                # service_intervals=np.random.exponential(1/srv_rate,max(int(srv_rate*self.time_res*5),2)) # to ensure they exceed one sampling interval
                
                safe_srv_rate = srv_rate if abs(srv_rate) > 1e-8 else 1e-3
                service_intervals = np.random.exponential(1/safe_srv_rate, max(int(safe_srv_rate * self.time_res * 5), 2))
                service_intervals=service_intervals[np.where(np.add.accumulate(service_intervals)<=self.time_res)[0]]
                service_intervals=service_intervals[0:np.min([len(service_intervals),self.queue.size])]
                arrival_intervals=np.random.exponential(1/self.arr_rate, max(int(self.arr_rate*self.time_res*5),2))

                arrival_intervals=arrival_intervals[np.where(np.add.accumulate(arrival_intervals)<=self.time_res)[0]]
                service_entries=np.array([[self.time+i,False] for i in service_intervals]) # False for service
                service_entries=service_entries.reshape(int(service_entries.size/2),2)
                time.sleep(1)
                arrival_entries=np.array([[self.time+i,True] for i in arrival_intervals]) # True for request
                # print("\n Arrived: ",arrival_entries) ####
                time.sleep(1)
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
                self.processEntries(all_entries, i, interval)
                self.time+=self.time_res
                
                self.record_interval_rates()
            
                # Reset the dispatch data at the end of each interval
                #if (i + 1) % (interval / self.time_res) == 0:  # Check if the interval has ended
                #    self.reset_state()
                # Ensure dispatch data is updated at each step
                ## self.dispatch_all_queues()
            
                self.set_batch_id(i)
            
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        finally:
            self.running = False  # Stop the dispatch thread
            dispatch_thread.join()
            print("Simulation completed.")    
            
        return
    
    
    # In your request arrival/service/renege/jockey events in your simulation, log each request like this:
    # Example (you may need to adapt field names to your code):

    def log_request(self, arrival_time, outcome, exit_time): # , queue=None
        request_log.append({
            'arrival_time': arrival_time,
            'outcome': outcome,  # "served", "reneged", "jockeyed"
            'departure_time': exit_time if outcome == "served" else None,
            'reneged_time': exit_time if outcome == "reneged" else None,
            'jockeyed_time': exit_time if outcome == "jockeyed" else None #,
            #'queue': queue
        })
      
        
    def reset_state(self):
        """
        Reset the state dictionary to its initial state.
        """
        self.dispatch_data = {
            "server_1": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []},
            "server_2": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []}
        }
        print("State has been reset.")
    
    
    def set_batch_id(self, id):
		
        self.batchid = id
		
		
    def get_batch_id(self):
		
        return self.batchid
	
		
    def get_all_service_times(self):
        
        return self.all_serv_times 
        
        
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
		

    def processEntries(self,entries=np.array([]), batchid=np.int16, interval=None): 
        
        #num_iterations = random.randint(1, 5)  # Random number of iterations between 1 and 5
        
        for entry in entries:
            # print("Processing a new request entry...")
            #self.time=entry[0]            
            if entry[1]==True:
                # print("  Adding a new request into task queue...")                

                req = self.addNewRequest(entry[0], batchid)
                self.arr_prev_times = np.append(self.arr_prev_times, entry[0])
                
            else:
				# Process the queue for a random number of times before new antrants
                #num_iterations = random.randint(1, self.arr_rate)  # Random number between 1 and 5
                #for _ in range(num_iterations):                
                q_selector = random.randint(1, 2)
                if q_selector == 1:					
                    self.queueID = "1" # Server1                    
                    curr_queue_len = len(self.dict_queues_obj[self.queueID])
                    if  curr_queue_len > 0:       
                        self.serveOneRequest(self.queueID, interval) # Server1 = self.dict_queues_obj["1"][0], entry[0],                                                                                      
                        time.sleep(random.uniform(0.1, 0.5))  # Random delay between 0.1 and 0.5 seconds                        
                        
                else:					
                    self.queueID = "2" 
                    curr_queue_len = len(self.dict_queues_obj[self.queueID])
                    if  curr_queue_len > 0:                    
                        self.serveOneRequest(self.queueID, interval) 
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
               

    def addNewRequest(self, expected_time_to_service_end, batchid): #, time_entered):
        # Join the shorter of either queues
               
        lengthQueOne = len(self.dict_queues_obj["1"]) # Server1
        lengthQueTwo = len(self.dict_queues_obj["2"]) # Server1 
        #rate_srv1,rate_srv2 = self.get_server_rates()
        
        # self.set_customer_id()       

        if lengthQueOne < lengthQueTwo:
            time_entered = self.time   #self.estimateMarkovWaitingTime(lengthQueOne) ID
            pose = lengthQueOne+1
            server_id = "1" # Server1
            self.customerid = self.get_customer_id()
            self.customerid = "Batch"+str(self.get_batch_id())+"_"+self.customerid
            #queue_intensity = self.arr_rate/rate_srv1
            #expected_time_to_service_end = self.estimateMarkovWaitingTime(float(pose)) # , queue_intensity, time_entered)
            #time_local_service = self.generateLocalCompUtility(req)

        else:
            pose = lengthQueTwo+1
            server_id = "2" # Server2
            self.customerid = self.get_customer_id()
            self.customerid = "Batch"+str(self.get_batch_id())+"_"+self.customerid
            time_entered = self.time #self.estimateMarkovWaitingTime(lengthQueTwo)
            #queue_intensity = self.arr_rate/rate_srv2
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

  
    def dispatch_queue_state(self, curr_queue, queue_id, alt_queue, alt_queue_id, interval): #, curr_queue_state): # curr_queue_id
    
        if interval not in self.dispatch_data:
            self.dispatch_data[interval] = {
                "server_1": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []},
                "server_2": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []}
            }
		
        #rate_srv1, rate_srv2 = self.get_service_rates() # get_server_rates
		
        if "1" in queue_id:
            #curr_queue_id = "1"
            serv_rate = self.get_service_rates(queue_id) # rate_srv1
        else:
            #curr_queue_id = "2"
            serv_rate = self.get_service_rates(queue_id) # rate_srv2
        
        curr_queue_state = self.get_queue_state(alt_queue_id)
         
        # Compute reneging rate and jockeying rate
        reneging_rate = self.compute_reneging_rate(curr_queue)
        jockeying_rate = self.compute_jockeying_rate(curr_queue)
        num_requests = curr_queue_state['total_customers'] # len(curr_queue)        

        for client in range(len(curr_queue)):
            req = curr_queue[client]  # self.dict_queues_obj[curr_queue_id][client]
            print(f"Dispatching state of server {alt_queue_id} to client {req.customerid} : {curr_queue_state}.")
            
            if "1" in alt_queue_id: # == "1":
                self.makeJockeyingDecision(req, alt_queue_id, queue_id, req.customerid, serv_rate)
                self.makeRenegingDecision(req, alt_queue_id, req.customerid)
                alt_queue_id = str(alt_queue_id) # "Server_"+
            else:
                self.makeJockeyingDecision(req, alt_queue_id, queue_id, req.customerid, serv_rate)
                self.makeRenegingDecision(req, alt_queue_id, req.customerid)
                alt_queue_id = str(alt_queue_id) # "Server_"+
        
        # Append rates to interval-specific dispatch data
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["intervals"].append(interval)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["num_requests"].append(num_requests)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["jockeying_rate"].append(jockeying_rate)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["reneging_rate"].append(reneging_rate)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["service_rate"].append(serv_rate)  
              
        # self.dispatch_data["num_requests"][curr_queue_id].append(len(curr_queue))
        #self.dispatch_data[f"server_{alt_queue_id}"]["num_requests"].append(len(curr_queue))
        #self.dispatch_data[f"server_{alt_queue_id}"]["jockeying_rate"].append(curr_queue_state['jockeying_rate']) # jockeying_rate)
        #self.dispatch_data[f"server_{alt_queue_id}"]["reneging_rate"].append(curr_queue_state['reneging_rate']) #reneging_rate)

        # return reneging_rate, jockeying_rate
        
    
    def dispatch_all_queues(self, interval): #  , interval=None
        """
        Dispatch the status of all queues and collect jockeying and reneging rates.
        """

        for queue_id in ["1", "2"]:
            curr_queue = self.dict_queues_obj[queue_id]
            alt_queue_id = "2" if queue_id == "1" else "1"
            alt_queue = self.dict_queues_obj[alt_queue_id]         
            
            # Compute rates
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
            reneging_rate = self.compute_reneging_rate(curr_queue)

            #curr_queue_state = self.get_queue_state(queue_id)         
            reneging_rate, jockeying_rate = self.dispatch_queue_state( curr_queue, queue_id, alt_queue, alt_queue_id, interval)       
            # Record the statistics
            serv_rate = self.get_service_rates(queue_id) # self.dict_servers_info[queue_id]
            num_requests = len(curr_queue)
            
            self.dispatch_data[f"server_{queue_id}"]["num_requests"].append(num_requests)
            self.dispatch_data[f"server_{queue_id}"]["jockeying_rate"].append(jockeying_rate)
            self.dispatch_data[f"server_{queue_id}"]["reneging_rate"].append(reneging_rate)
            self.dispatch_data[f"server_{queue_id}"]["service_rate"].append(serv_rate)
            self.dispatch_data[f"server_{queue_id}"]["intervals"].append(interval)

            #print(f"Server {queue_id} - Num requests: {num_requests}, Jockeying rate: {jockeying_rate}, "
            #      f"Reneging rate: {reneging_rate}, Service rate: {serv_rate}, Long-run rate: {curr_queue_state['long_run_change_rate']}")                
              
    
    
    def plot_rates(self):
        """
        Plot the jockeying and reneging rates over time.
        """       
        # Ensure the number of requests is sorted and consistent
        
        num_requests_srv1 = sorted(self.dispatch_data["server_1"]["num_requests"])
        num_requests_srv2 = sorted(self.dispatch_data["server_2"]["num_requests"])
        num_requests = num_requests_srv1 + num_requests_srv2
        
        print("Into the plotting area now")
        
        # Ensure all data lists are of the same length
        server_1_jockeying_rate = self.dispatch_data["server_1"]["jockeying_rate"]
        server_1_reneging_rate = self.dispatch_data["server_1"]["reneging_rate"]
        server_1_service_rate = self.dispatch_data["server_1"]["service_rate"]
        server_2_jockeying_rate = self.dispatch_data["server_2"]["jockeying_rate"]
        server_2_reneging_rate = self.dispatch_data["server_2"]["reneging_rate"]
        server_2_service_rate = self.dispatch_data["server_2"]["service_rate"]

        min_len = min(len(num_requests_srv1), len(num_requests_srv2),
                  len(server_1_jockeying_rate), len(server_1_reneging_rate), len(server_1_service_rate),
                  len(server_2_jockeying_rate), len(server_2_reneging_rate), len(server_2_service_rate))
                                   

        num_requests_srv1 = num_requests_srv1[:min_len]
        num_requests_srv2 = num_requests_srv2[:min_len]
        server_1_jockeying_rate = server_1_jockeying_rate[:min_len]
        server_1_reneging_rate = server_1_reneging_rate[:min_len]
        server_1_service_rate = server_1_service_rate[:min_len]
        server_2_jockeying_rate = server_2_jockeying_rate[:min_len]
        server_2_reneging_rate = server_2_reneging_rate[:min_len]
        server_2_service_rate = server_2_service_rate[:min_len]

        # Define intervals (example: every 10 requests)
        interval_markers = range(0, len(num_requests_srv1), 10)

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot for Server 1
        axs[0].plot(num_requests_srv1, server_1_jockeying_rate, label='Jockeying Rate', color='blue')
        axs[0].plot(num_requests_srv1, server_1_reneging_rate, label='Reneging Rate', color='red')
        axs[0].plot(num_requests_srv1, server_1_service_rate, label='Service Rate', color='green')
        axs[0].scatter([num_requests_srv1[i] for i in interval_markers],
                   [server_1_jockeying_rate[i] for i in interval_markers],
                   color='blue', marker='o', label='Interval Marker (Jockeying)')
        axs[0].scatter([num_requests_srv1[i] for i in interval_markers],
                   [server_1_reneging_rate[i] for i in interval_markers],
                   color='red', marker='x', label='Interval Marker (Reneging)')
        axs[0].set_title('Server 1 Rates')
        axs[0].set_ylabel('Rate')
        axs[0].legend()

        # Plot for Server 2
        axs[1].plot(num_requests_srv2, server_2_jockeying_rate, label='Jockeying Rate', color='blue')
        axs[1].plot(num_requests_srv2, server_2_reneging_rate, label='Reneging Rate', color='red')
        axs[1].plot(num_requests_srv2, server_2_service_rate, label='Service Rate', color='green')
        axs[1].scatter([num_requests_srv2[i] for i in interval_markers],
                   [server_2_jockeying_rate[i] for i in interval_markers],
                   color='blue', marker='o', label='Interval Marker (Jockeying)')
        axs[1].scatter([num_requests_srv2[i] for i in interval_markers],
                   [server_2_reneging_rate[i] for i in interval_markers],
                   color='red', marker='x', label='Interval Marker (Reneging)')
        axs[1].set_title('Server 2 Rates')
        axs[1].set_xlabel('Number of Requests')
        axs[1].set_ylabel('Rate')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        
        
    def plot_rates_by_intervals_old(self):
        """
        For each interval, plot jockeying and reneging rates vs. service rates in SEPARATE subplots
        for each server (Server 1 and Server 2).
        """
   

        intervals = [3, 6, 9]  # Or: sorted(self.dispatch_data.keys())
        servers = ["server_1", "server_2"]
        interval_labels = {3: "3 seconds", 6: "6 seconds", 9: "9 seconds"}

        for interval in intervals:
            if interval not in self.dispatch_data:
                print(f"No data available for the {interval_labels.get(interval, interval)} interval.")
                continue

            fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
            fig.suptitle(f"Rates vs Service Rate | Interval: {interval_labels.get(interval, interval)}")
            for server_idx, server in enumerate(servers):
                service_rate = np.array(self.dispatch_data[interval][server].get("service_rate", []))
                jockeying_rate = np.array(self.dispatch_data[interval][server].get("jockeying_rate", []))
                reneging_rate = np.array(self.dispatch_data[interval][server].get("reneging_rate", []))

                min_len = min(len(service_rate), len(jockeying_rate), len(reneging_rate))
                if min_len == 0:
                    continue

                # Sort by service_rate for smooth plotting
                sort_idx = np.argsort(service_rate[:min_len])
                x = service_rate[:min_len][sort_idx]
                y_jockey = jockeying_rate[:min_len][sort_idx]
                y_renege = reneging_rate[:min_len][sort_idx]

                # Jockeying Rate vs Service Rate
                ax_jockey = axs[server_idx, 0]
                ax_jockey.plot(x, y_jockey, 'b-o', label="Jockeying Rate")
                ax_jockey.set_title(f"{server.replace('_', ' ').title()} - Jockeying Rate")
                ax_jockey.set_xlabel("Service Rate")
                ax_jockey.set_ylabel("Jockeying Rate")
                ax_jockey.grid(True)
                ax_jockey.legend()

                # Reneging Rate vs Service Rate
                ax_renege = axs[server_idx, 1]
                ax_renege.plot(x, y_renege, 'r-x', label="Reneging Rate")
                ax_renege.set_title(f"{server.replace('_', ' ').title()} - Reneging Rate")
                ax_renege.set_xlabel("Service Rate")
                ax_renege.set_ylabel("Reneging Rate")
                ax_renege.grid(True)
                ax_renege.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
            
    def plot_rates_by_intervals(self, window_length=7, polyorder=2):
        """
        Improved plotting of jockeying and reneging rates vs service rate for each server and interval.
        - Smooths the rates using Savitzky-Golay filter if enough data points exist.
        - Aggregates duplicate service rates by averaging their rates.
        - Plots both scatter (raw) and line (smoothed/aggregated) representations.
        Args:
            window_length: int, window for smoothing (must be odd and <= data length)
            polyorder: int, order for Savitzky-Golay filter
        """
        intervals = list(self.dispatch_data.keys())
        
        for interval in intervals:
            server_names = list(self.dispatch_data[interval].keys())
            fig, axs = plt.subplots(2, len(server_names), figsize=(6*len(server_names), 8), sharex=False)
            if len(server_names) == 1:
                axs = np.array([[axs[0]], [axs[1]]])
            for idx, server in enumerate(server_names):
                # Get data
                service_rates = np.array(self.dispatch_data[interval][server]["service_rate"])
                jockeying_rates = np.array(self.dispatch_data[interval][server]["jockeying_rate"])
                reneging_rates = np.array(self.dispatch_data[interval][server]["reneging_rate"])
                # Aggregate: average rates for duplicate service rates
                uniq_sr, inv = np.unique(service_rates, return_inverse=True)
                jockeying_mean = np.array([jockeying_rates[inv==i].mean() for i in range(len(uniq_sr))])
                reneging_mean = np.array([reneging_rates[inv==i].mean() for i in range(len(uniq_sr))])
                # Smoothing (optional, if enough points)
                if len(uniq_sr) >= window_length:
                    jockeying_smooth = savgol_filter(jockeying_mean, window_length, polyorder)
                    reneging_smooth = savgol_filter(reneging_mean, window_length, polyorder)
                else:
                    jockeying_smooth = jockeying_mean
                    reneging_smooth = reneging_mean
                # Plot Jockeying Rate
                axs[0, idx].scatter(service_rates, jockeying_rates, alpha=0.3, color='blue', label="Raw data")
                axs[0, idx].plot(uniq_sr, jockeying_mean, color="black", linewidth=1.5, label="Mean (per SR)")
                axs[0, idx].plot(uniq_sr, jockeying_smooth, color="red", linewidth=2, label="Smoothed")
                axs[0, idx].set_title(f"{server.title()} - Jockeying Rate")
                axs[0, idx].set_xlabel("Service Rate")
                axs[0, idx].set_ylabel("Jockeying Rate")
                axs[0, idx].legend()
                axs[0, idx].grid(True, linestyle='--', alpha=0.5)
                # Plot Reneging Rate
                axs[1, idx].scatter(service_rates, reneging_rates, alpha=0.3, color='red', label="Raw data")
                axs[1, idx].plot(uniq_sr, reneging_mean, color="black", linewidth=1.5, label="Mean (per SR)")
                axs[1, idx].plot(uniq_sr, reneging_smooth, color="blue", linewidth=2, label="Smoothed")
                axs[1, idx].set_title(f"{server.title()} - Reneging Rate")
                axs[1, idx].set_xlabel("Service Rate")
                axs[1, idx].set_ylabel("Reneging Rate")
                axs[1, idx].legend()
                axs[1, idx].grid(True, linestyle='--', alpha=0.5)
            plt.suptitle(f"Rates vs Service Rate | Interval: {interval} seconds")
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.show()          
        
        
    def plot_waiting_time_vs_rates_by_interval(self, smoothing_window=5):
        """
        For each interval and server, plot waiting time vs smoothed jockeying and reneging rates.
        Ensures all arrays are of the same length before plotting.
        """        
        
        def moving_average(data, window_size):
            """Compute the moving average of a 1D array."""
            if window_size < 2 or len(data) < window_size:
                return data
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')

        intervals = sorted(self.dispatch_data.keys())
        servers = ["server_1", "server_2"]

        fig, axs = plt.subplots(len(intervals), len(servers), figsize=(6 * len(servers), 4 * len(intervals)), sharex=False)
        if len(intervals) == 1 and len(servers) == 1:
            axs = [[axs]]
        elif len(intervals) == 1 or len(servers) == 1:
            axs = [axs] if len(intervals) == 1 else [[a] for a in axs]

        for i, interval in enumerate(intervals):
            for j, server in enumerate(servers):
                waiting_times = self.dispatch_data[interval][server].get("waiting_times", [])
                jockeying_rates = self.dispatch_data[interval][server].get("jockeying_rate", [])
                reneging_rates = self.dispatch_data[interval][server].get("reneging_rate", [])

                min_len = min(len(waiting_times), len(jockeying_rates), len(reneging_rates))
                if min_len == 0:
                    continue  # Skip empty panels

                w = np.array(waiting_times[:min_len])
                j_rate = np.array(jockeying_rates[:min_len])
                r_rate = np.array(reneging_rates[:min_len])

                # Sort by waiting time for better visualization
                sort_idx = np.argsort(w)
                w, j_rate, r_rate = w[sort_idx], j_rate[sort_idx], r_rate[sort_idx]
                
                # Optional: remove outliers
                j_rate = np.clip(j_rate, np.percentile(j_rate, 5), np.percentile(j_rate, 95))
                r_rate = np.clip(r_rate, np.percentile(r_rate, 5), np.percentile(r_rate, 95))
                
                # Apply smoothing
                #w_smooth = w
                #j_smooth = moving_average(j_rate, smoothing_window)
                #r_smooth = moving_average(r_rate, smoothing_window)
                # Truncate w to match smoothing result length
                #min_smooth_len = min(len(w_smooth), len(j_smooth), len(r_smooth))
                #w_smooth = w_smooth[:min_smooth_len]
                #j_smooth = j_smooth[:min_smooth_len]
                #r_smooth = r_smooth[:min_smooth_len]
                
                # Apply Savitzky-Golay smoothing
                min_len = len(w)
                if min_len >= 5:
                    window_length = min(21, min_len)
                    if window_length % 2 == 0:
                        window_length -= 1
                    j_smooth = savgol_filter(j_rate, window_length, polyorder=2)
                    r_smooth = savgol_filter(r_rate, window_length, polyorder=2)
                else:
                    j_smooth = j_rate
                    r_smooth = r_rate

                ax = axs[i][j]
                ax.plot(w, j_smooth, label="Jockeying Rate (smoothed)", color="blue", marker='o')
                ax.plot(w, r_smooth, label="Reneging Rate (smoothed)", color="red", marker='x')
                ax.scatter(w, j_rate, color="blue", s=10, alpha=0.3)
                ax.scatter(w, r_rate, color="red", s=10, alpha=0.3)
                ax.set_xlabel("Waiting Time")
                ax.set_ylabel("Rate")
                ax.set_title(f"{server.replace('_', ' ').title()} | Interval: {interval}s")
                ax.legend()
                ax.grid(True)

        plt.tight_layout()
        plt.show()

        
        
    def plot_reneged_waiting_times_by_interval(self):
        """
        For each dispatch interval, plot the waiting times of requests that were reneged.
        """
        if not hasattr(self, 'reneged_waiting_times_by_interval'):
            print("No reneged waiting times data collected.")
            return

        intervals = sorted(self.reneged_waiting_times_by_interval.keys())
        servers = ["server_1", "server_2"]
        fig, axs = plt.subplots(len(intervals), len(servers), figsize=(6 * len(servers), 4 * len(intervals)), sharex=True)
        if len(intervals) == 1 and len(servers) == 1:
            axs = [[axs]]
        elif len(intervals) == 1 or len(servers) == 1:
            axs = [axs] if len(intervals) == 1 else [[a] for a in axs]

        for i, interval in enumerate(intervals):
            for j, server in enumerate(servers):
                waiting_times = self.reneged_waiting_times_by_interval[interval][server]
                ax = axs[i][j]
                if waiting_times:
                    ax.hist(waiting_times, bins=10, color='red', alpha=0.7)
                    ax.set_title(f"{server.replace('_', ' ').title()} | Interval: {interval}s\n(Reneged)")
                    ax.set_xlabel("Waiting Time")
                    ax.set_ylabel("Count")
                else:
                    ax.set_title(f"{server.replace('_', ' ').title()} | Interval: {interval}s (No reneged)")
                    ax.set_xlabel("Waiting Time")
                    ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()
        
        
    def plot_jockeyed_waiting_times_by_interval(self):
        """
        For each dispatch interval, plot the waiting times of requests that were jockeyed.
        """
        if not hasattr(self, 'jockeyed_waiting_times_by_interval'):
            print("No jockeyed waiting times data collected.")
            return

        intervals = sorted(self.jockeyed_waiting_times_by_interval.keys())
        servers = ["server_1", "server_2"]
        fig, axs = plt.subplots(len(intervals), len(servers), figsize=(6 * len(servers), 4 * len(intervals)), sharex=True)
        if len(intervals) == 1 and len(servers) == 1:
            axs = [[axs]]
        elif len(intervals) == 1 or len(servers) == 1:
            axs = [axs] if len(intervals) == 1 else [[a] for a in axs]

        for i, interval in enumerate(intervals):
            for j, server in enumerate(servers):
                waiting_times = self.jockeyed_waiting_times_by_interval[interval][server]
                ax = axs[i][j]
                if waiting_times:
                    ax.hist(waiting_times, bins=10, color='blue', alpha=0.7)
                    ax.set_title(f"{server.replace('_', ' ').title()} | Interval: {interval}s\n(Jockeyed)")
                    ax.set_xlabel("Waiting Time")
                    ax.set_ylabel("Count")
                else:
                    ax.set_title(f"{server.replace('_', ' ').title()} | Interval: {interval}s (No jockeyed)")
                    ax.set_xlabel("Waiting Time")
                    ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()


    def plot_waiting_time_cdf(waiting_times, title="CDF of Waiting Times"):
        """
        Plots the cumulative distribution function (CDF) of waiting times.
        waiting_times: List or numpy array of waiting times.
        """
        waiting_times = np.sort(waiting_times)
        cdf = np.arange(1, len(waiting_times)+1) / len(waiting_times)
        plt.figure(figsize=(8,4))
        plt.plot(waiting_times, cdf, marker='.')
        plt.xlabel("Waiting Time")
        plt.ylabel("CDF")
        plt.title(title)
        plt.grid(True)
        plt.show()
    
     
    
    def setup_dispatch_intervals(self, intervals):
        """
        Set up a timer-based interval for dispatching queue status.
        Queue operations like arrivals and departures continue during the interval.
        """
        
        print(f"Starting dispatch scheduler with interval: {interval} seconds")
        # Start the dispatch timer thread
        dispatch_thread = threading.Thread(target=self.dispatch_timer, args=(interval,))
        dispatch_thread.start()

        # Start the background queue operations
        background_thread = threading.Thread(target=self.background_operations)
        background_thread.start()

        # Join threads to ensure proper termination
        dispatch_thread.join()
        background_thread.join()
        
        #schedule.every(10).seconds.do(self.dispatch_all_queues)
        #schedule.every(30).seconds.do(self.dispatch_all_queues)
        # schedule.every(60).seconds.do(self.dispatch_all_queues)      
            

    def run_scheduler(self): # , duration=None
        """
        Run the scheduler to dispatch queue status at different intervals.
        """
        self.setup_dispatch_intervals()
        start_time = time.time()
        while True:
            schedule.run_pending()
            time.sleep(1)
            
            # Check for termination condition
            #if duration is not None and (time.time() - start_time) >= duration:
            #    print("Scheduler terminated after running for {:.2f} seconds.".format(duration))
            #    break
    
    
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
    
                       
    def get_queue_state(self, queueid):
		        
        srvrate1 = self.get_service_rates("1")        
        srvrate2 = self.get_service_rates("2")
        
        # Ensure service rates are set to a default if None
        if isinstance( None, type(self.srvrates_1)):
            self.srvrates_1 = 1.0
        if isinstance( None, type(self.srvrates_2)):
            self.srvrates_2 = 1.0
        if isinstance( None, type(self.arr_rate)):
            self.arr_rate = self.srvrates_1 + self.srvrates_2
            
        
        b = 0.4
        a = 0.2       
	    
        trans_matrix = np.array([
            [0.0, a],   # row for state 0 (fill Q[0,0] next)
            [b,  0.0]   # row for state 1 (fill Q[1,1] next)
        ])
        
        for i in range(trans_matrix.shape[0]):
            trans_matrix[i,i] = -np.sum(trans_matrix[i, :]) + trans_matrix[i,i]
            
        #print("\n Server",queueid ," ******> ", self.get_service_rates(queueid),self.srvrates_1 ,self.srvrates_2 )
        
        if "1" in queueid:		
            # srvrates = self.get_service_rates(queueid) 
            customers_in_queue = self.dict_queues_obj["1"] 
            curr_queue = self.dict_queues_obj[queueid]  
            reneging_rate = self.compute_reneging_rate(curr_queue)
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
            # Instantiate MarkovQueueModel
            
            markov_model = MarkovQueueModel(self.arr_rate, srvrate1, max_states=len(customers_in_queue)) # 1000)
            servmarkov_model = MarkovModulatedServiceModel([srvrate1,srvrate2], trans_matrix)
            #long_run_change_rate = markov_model.long_run_change_rate()
            sample_interchange_time = markov_model.compute_expected_time_between_changes(self.arr_rate, self.srvrates_1, N=100) #len(customers_in_queue)) # 1000)
            #long_run_change_rate = markov_model.long_run, self.srvrates_1, len(customers_in_queue)) # sample_interchange_time() # _steady_state_distribution()
            arr_rate1, arr_rate2 = servmarkov_model.arrival_rates_divisor(self.arr_rate, self.srvrates_1, self.srvrates_2) #_steady_state_distribution()
            steady_state_distribution = servmarkov_model.best_queue_delay(arr_rate1, self.srvrates_1, arr_rate2, self.srvrates_2)
            #steady_state_distribution_list = steady_state_distribution.tolist()  # Convert to list                    
      
        else:
			 
            # srvrates = self.get_service_rates(queueid)            
            customers_in_queue = self.dict_queues_obj["2"]
            curr_queue = self.dict_queues_obj[queueid]
            reneging_rate = self.compute_reneging_rate(curr_queue)
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
            # Instantiate MarkovQueueModel
            # print("\n **********> ", self.arr_rate, srvrate2)
            markov_model = MarkovQueueModel(self.arr_rate, srvrate2, max_states=len(customers_in_queue)) # 1000)
            servmarkov_model = MarkovModulatedServiceModel([srvrate1,srvrate2], trans_matrix)
            #long_run_change_rate = markov_model.long_run_change_rate()
            sample_interchange_time = markov_model.compute_expected_time_between_changes(self.arr_rate, self.srvrates_2, N=100) # len(customers_in_queue)) # 1000)
            arr_rate1, arr_rate2 = servmarkov_model.arrival_rates_divisor(self.arr_rate, self.srvrates_1, self.srvrates_2) #_steady_state_distribution()
            steady_state_distribution = servmarkov_model.best_queue_delay(arr_rate1, self.srvrates_1, arr_rate2, self.srvrates_2)
            #long_run_change_rate = markov_model.long_run, self.srvrates_2, len(customers_in_queue))
            #steady_state_distribution = markov_model._steady_state_distribution()
            #steady_state_distribution_list = steady_state_distribution.tolist()  # Convert to list
            #sample_interchange_time = markov_model.sample_interchange_time()          
		       
        curr_queue_state = {
            "total_customers": len(customers_in_queue),
            #"intensity": queue_intensity,
            "capacity": self.capacity,              
            "long_avg_serv_time": self.get_long_run_avg_service_time(queueid),
            "sample_interchange_time": sample_interchange_time,
            #"long_run_change_rate": long_run_change_rate,
            "steady_state_distribution": steady_state_distribution, #{
                #"max": max(steady_state_distribution_list),
                #"min": min(steady_state_distribution_list),
                #"mean": sum(steady_state_distribution_list) / len(steady_state_distribution_list),
            #},
            "reneging_rate": reneging_rate,
            "jockeying_rate" : jockeying_rate             
        }
        
        return curr_queue_state
  

    def serveOneRequest(self, queueID, interval): # to_delete, serv_time, 
        #randomly select which queue to process at a time t+1
        q_selector = random.randint(1, 2)                            
        
        # ToDo:: run the processing of queues for some specific interval of time 
        # before admitting more into the queue
        len_queue_1,len_queue_2 = self.get_queue_sizes()
        
        if not hasattr(self, 'waiting_times_srv1'):
             self.waiting_times_srv1 = []
        if not hasattr(self, 'waiting_times_srv2'):
             self.waiting_times_srv2 = []
        # New: Initialize per-interval storage for jockeyed and reneged waiting times
        if not hasattr(self, 'jockeyed_waiting_times_by_interval'):
            self.jockeyed_waiting_times_by_interval = {}
        if not hasattr(self, 'reneged_waiting_times_by_interval'):
            self.reneged_waiting_times_by_interval = {}
        if interval not in self.jockeyed_waiting_times_by_interval:
            self.jockeyed_waiting_times_by_interval[interval] = {"server_1": [], "server_2": []}
        if interval not in self.reneged_waiting_times_by_interval:
            self.reneged_waiting_times_by_interval[interval] = {"server_1": [], "server_2": []}
        
        if "1" in queueID:   # Server1               
            req =  self.dict_queues_obj["1"][0] # Server1
            serv_rate = self.get_service_rates(queueID) # self.dict_servers_info["1"] # Server1
            queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
            queueID = "1" # Server1
    
            reward = self.get_jockey_reward(req)
                   
            # serve request in queue                                
            self.queueID = queueID
            self.dict_queues_obj["1"] = self.dict_queues_obj["1"][1:self.dict_queues_obj["1"].size]       # Server1 
            
            # When a request is served:
            self.log_request(req.time_entrance, "served", req.time_res) #, self.dict_queues_obj["1"]) # req.queue) arrival_time= outcome= exit_time= queue=

            self.total_served_requests_srv2+=1                       
            
            # Set the exit time
            req.time_exit = self.time 
            
            waiting_time = req.time_exit - req.time_entrance
            self.waiting_times_srv1.append(waiting_time)   
            
            self.record_waiting_time(interval, "server_1", waiting_time) 
            
            # Track jockeyed and reneged waiting times for this interval and server
            if "_jockeyed" in req.customerid:
                self.jockeyed_waiting_times_by_interval[interval]["server_1"].append(waiting_time)
            if "_reneged" in req.customerid:
                self.reneged_waiting_times_by_interval[interval]["server_1"].append(waiting_time)         
            
            # take note of the observation ... self.time  queue_id,  serv_rate, intensity, time_in_serv, activity, rewarded, curr_pose
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, req.time_exit-req.time_entrance, reward, len_queue_1, 2, req, len(self.dict_queues_obj["1"]))   # req.exp_time_service_end,                                    
            self.history.append(self.objObserv.get_obs())
      
            self.arr_prev_times = self.arr_prev_times[1:self.arr_prev_times.size]
            
            self.objQueues.update_queue_status(queueID)      
            
            '''
                Now after serving a request, dispatch the new state of the queues
            '''
            
            # self.makeJockeyingDecision(req, self.queueID, "2", req.customerid, serv_rate)
            # self.makeRenegingDecision(req, self.queueID)

        else:                        
            req = self.dict_queues_obj["2"][0] # Server2
            serv_rate = self.get_service_rates(queueID) #self.dict_servers_info["2"] # Server2
            queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
            queueid = "2"   # Server2      
                        
            self.dict_queues_obj["2"] = self.dict_queues_obj["2"][1:self.dict_queues_obj["2"].size] # Server2
            
            reward = self.get_jockey_reward(req)
         
            self.queueID = queueID 
            self.dict_queues_obj["S2"] = self.dict_queues_obj["2"][1:self.dict_queues_obj["2"].size]      
            self.log_request(req.time_entrance, "served", req.time_res) # , self.dict_queues_obj["2"]) # req.queue)  arrival_time= outcome=  exit_time= queue=
            self.total_served_requests_srv1+=1                        
            
            # Set the exit time
            req.time_exit = self.time   
            
            waiting_time = req.time_exit - req.time_entrance
            self.waiting_times_srv2.append(waiting_time) 
            self.record_waiting_time(interval, "server_2", waiting_time)             
            
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, req.time_exit-req.time_entrance, reward, len_queue_2, 2, req, len(self.dict_queues_obj["2"]))    # req.exp_time_service_end,                                  
            self.history.append(self.objObserv.get_obs())
            
            # Track jockeyed and reneged waiting times for this interval and server
            if "_jockeyed" in req.customerid:
                self.jockeyed_waiting_times_by_interval[interval]["server_2"].append(waiting_time)
            if "_reneged" in req.customerid:
                self.reneged_waiting_times_by_interval[interval]["server_2"].append(waiting_time)                   
               
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
    
    
    def record_waiting_time(self, interval, server_id, waiting_time):
		
        if interval not in self.dispatch_data:
            self.dispatch_data[interval] = {
                "server_1": {"waiting_times": [], "num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": [] },              
                "server_2": {"waiting_times": [], "num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": [] }
            }
        # Ensure waiting_times exists
        if "waiting_times" not in self.dispatch_data[interval][server_id]:
            self.dispatch_data[interval][server_id]["waiting_times"] = []
            
        self.dispatch_data[interval][server_id]["waiting_times"].append(waiting_time)
    

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
            serv_rate = self.get_service_rates(queue_id) #self.dict_servers_info["1"]  # Server1 get_service_rate
            queue = self.dict_queues_obj["1"]  # Queue1
        else:
            serv_rate = self.get_service_rates(queue_id) # self.dict_servers_info["2"]  # Server2
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
        
    
    
    def total_wasted_waiting_time(request_log, queue_id=None):
        """
        Calculate total wasted waiting time for requests that renege or jockey (i.e., leave unsatisfied).
        Also returns a breakdown by outcome (reneged, jockeyed, served, etc.)
        Args:
            request_log (list): List of dict/objects with at least
                'arrival_time', 'outcome', and an appropriate exit time field:
                    'departure_time', 'reneged_time', or 'jockeyed_time'.
                 Optionally, 'queue' or 'queue_id' field for per-queue analysis.
            queue_id (optional): If specified, only consider requests from this queue.

        Returns:
            total (float): Total wasted waiting time (sum for reneged/jockeyed).
            per_outcome (dict): Dict mapping outcome -> sum of waiting times for that outcome.
        """
        
        total = 0.0
        per_outcome = {}
        # print("\n ----> ", request_log, type(request_log))
        for req in request_log:
            # Optionally filter by queue
            if queue_id is not None and req.get('queue', req.get('queue_id', None)) != queue_id:
                continue
            outcome = req['outcome']
            # Determine exit time by outcome
            if outcome == 'reneged':
                exit_time = req.get('reneged_time', req.get('exit_time', req.get('departure_time')))
            elif outcome == 'jockeyed':
                exit_time = req.get('jockeyed_time', req.get('exit_time', req.get('departure_time')))
            elif outcome == 'served':
                exit_time = req.get('departure_time', req.get('exit_time'))
            else:
                # For any other outcome, attempt to use a generic exit/departure time
                exit_time = req.get('exit_time', req.get('departure_time'))
            # Only count wasted time for reneged and jockeyed in total
            if outcome in ['reneged', 'jockeyed']:
                total += exit_time - req['arrival_time']
            # Always count per-outcome wasted/waiting time
            per_outcome.setdefault(outcome, 0.0)
            per_outcome[outcome] += exit_time - req['arrival_time']
        return total, per_outcome
    
    
    
    def makeRenegingDecision_alternative(self, req, queueid):
		# self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay) # 
        decision=False 
        curr_queue_state = self.get_queue_state(queueid)                    
        queue_interchange_time = curr_queue_state["sample_interchange_time"]
        #T_local = self.dist_local_delay.rvs(loc=self.loc_local_delay, scale=self.scale_local_delay)
                
        if queueid == "1":
            serv_rate = self.dict_servers_info["1"]
            queue =  self.dict_queues_obj["1"]   
            alt_queue_id = "2"   
            curr_arriv_rate = self.objQueues.get_arrivals_rates()
            queue_intensity = curr_arriv_rate/ serv_rate 
            alt_queue_state = self.get_queue_state(alt_queue_id)
            alt_interchange_time = alt_queue_state["sample_interchange_time"] 
            T_local = stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=queue_interchange_time) #=0.75/req.pos_in_queue)
            #self. max_cloud_delay = self.calculate_max_cloud_delay(req.pos_in_queue, queue_intensity, req)  
            #self.max_cloud_delay=stats.erlang.ppf(self.certainty, loc=0, scale=curr_arriv_rate, a=req.pos_in_queue)
            
            # use Little's law to compute expected wait in alternative queue
            # instead of Little's Law, use the expected waiting time in steady state using (1 - queue_intensity e^{-(serv_rate - curr_arriv_rate)})
            expected_wait_in_alt_queue = 1 - queue_intensity * math.exp(-(serv_rate - curr_arriv_rate)) # float(len(self.dict_queues_obj["2"])/curr_arriv_rate)
            T_queue = expected_wait_in_alt_queue # queue_interchange_time + expected_wait_in_alt_queue
        else:
            serv_rate = self.dict_servers_info["2"] 
            queue =  self.dict_queues_obj["2"]
            alt_queue_id = "1"
            curr_arriv_rate = self.objQueues.get_arrivals_rates()
            queue_intensity = curr_arriv_rate/ serv_rate    
            alt_queue_state = self.get_queue_state(alt_queue_id)  
            alt_interchange_time = alt_queue_state["sample_interchange_time"]  
            T_local = stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale= queue_interchange_time) #=0.75/req.pos_in_queue)   # maybe scale= queue_interchange_time
            #self. max_cloud_delay = self.calculate_max_cloud_delay(len(queue), queue_intensity, req)
            #self.max_cloud_delay=stats.erlang.ppf(self.certainty, loc=0, scale=curr_arriv_rate, a=req.pos_in_queue)
            
            # use Little's law to compute expected wait in alternative queue
            # instead of Little's Law, use the expected waiting time in steady state using (1 - queue_intensity e^{-(serv_rate - curr_arriv_rate)})
            expected_wait_in_alt_queue = 1 - queue_intensity * math.exp(-(serv_rate - curr_arriv_rate)) # float(len(self.dict_queues_obj["1"])/curr_arriv_rate)
            T_queue = expected_wait_in_alt_queue # queue_interchange_time + expected_wait_in_alt_queue
        
        if self.learning_mode=='transparent':
            self.max_cloud_delay=stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=1/serv_rate) 
        else:                           
                                  
            # print("\n Times => ", T_local, " = ", T_queue)
            # Choose the best option
            if T_local < T_queue:
            # local_processing_time = self.dist_local_delay.rvs(loc=self.loc_local_delay, scale=self.scale_local_delay) # => too strict rule    
            # use a quantile below        
            #local_processing_time = self.dist_local_delay.ppf(0.9, loc=self.loc_local_delay, scale=self.scale_local_delay)
                
            # Decision: renege to local if local is (much) faster than expected queue interchange time
            # local delay is constant, no waiting is expected
            # if self.loc_local_delay < queue_interchange_time:
                self.loc_local_delay = T_local
                decision = True
                curr_pose = self.get_request_position(queueid, req.customerid)
        
                if curr_pose is None: #  if curr_pose >= len(self.queue): # 
                    print(f"Request ID {req.customerid} not found in queue {queueid}. Continuing with processing...")
                    return 
                else:               
                    self.reqRenege( req, queueid, curr_pose, serv_rate, queue_intensity, T_local, req.customerid, req.service_time, decision, self.queue)
                    
                
            else:
                decision = False
            
        return decision
    
    
    def makeRenegingDecision(self, req, queueid, customer_id, t_max=10.0, num_points=1000):	 #  T_local,	

        def exp_cdf(mu, t):
            """
            CDF of Exp(mu) at time t: P(W <= t) = 1 - exp(-mu * t).
            """
            return 1 - np.exp(-mu * t)
            
        def erlang_C(c, rho):
            """
            Compute the Erlang‐C probability (P_wait) for an M/M/c queue:
              P_wait = [ (rho^c / c!)*(c/(c - rho)) ] / [ sum_{k=0..c-1} (rho^k / k!) + (rho^c / c!)*(c/(c-rho)) ].
            """
            # Sum_{k=0 to c-1} (rho^k / k!)
            sum_terms = sum((rho**k) / factorial(k) for k in range(c))
            num = (rho**c / factorial(c)) * (c / (c - rho))
            return num / (sum_terms + num)

        def mmc_wait_cdf_and_tail(lambda_i, mu_i, c, t):
            """
            For M/M/c with arrival rate lambda_i, service rate mu_i, compute:
              - P_wait  = Erlang‐C(c, rho_i)
              - delta   = c*mu_i - lambda_i   (rate parameter for the exponential tail)
              - CDF(t)  = 1 - P_wait * exp(-delta * t)
              - tail(t) = P_wait * exp(-delta * t)
            Returns (CDF(t), tail(t)).
            If rho_i >= c, we assume the system is unstable, so tail=1 for any finite t.
            """
            
            rho_i = lambda_i / mu_i
            if rho_i >= c:
                return 0.0, 1.0  # CDF = 0, tail = 1 (infinite wait)
            P_wait = erlang_C(c, rho_i)
            delta = c*mu_i - lambda_i
            tail = P_wait * exp(-delta * t)
            cdf = 1 - tail
            
            return cdf, tail
                      

        def compute_steady_state_probs(rho, N):
            """Compute steady-state probabilities for M/M/1 with truncation at N."""
            return np.array([(1 - rho) * rho**n for n in range(N + 1)])

        def compute_rate_of_change(lambda_, mu, N):
            """
            Compute average rate at which queue length changes (birth + death)
            in an M/M/1 queue, truncated at state N.
            """
            rho = lambda_ / mu
            pi = compute_steady_state_probs(rho, N)
            # For n = 0: rate = lambda
            # For n > 0: rate = lambda + mu
            R_change = sum(pi[n] * (lambda_ if n == 0 else (lambda_ + mu)) for n in range(N + 1))
            return R_change
            
        def compute_expected_time_between_changes(lambda_, mu, N):
            """
            Compute expected time between changes in queue length.
            Handles degenerate/unstable cases robustly.
            """
            R_change = compute_rate_of_change(lambda_, mu, N)
            
            if R_change is None or not np.isfinite(R_change) or R_change <= 0:
                return lambda_ / mu # 1e4  # fallback: large finite value
                
            return 1 / R_change
            
        
        def arrival_rates_divisor(arrival_rate, mu1, mu2):
			# if the arrival rate is an odd number, divide it by two and 
			# add the reminder to the queue with the higher service rate
			# Else equal service rates
			
            """
               Divide n by 2. If n is odd, add its remainder (1) to rem_accumulator.
    
                Parameters:  n (int): The integer to divide.
                             rem_accumulator (int): The variable to which any odd remainder is added.
    
                Returns:
                    tuple:
                        half (int): Result of integer division n // 2.
                        new_accumulator (int): Updated rem_accumulator.
            """
            if mu1 < mu2:
                rem_accumulator = mu1
            else:
                rem_accumulator = mu2
				
            remainder = arrival_rate % 2
            half = arrival_rate // 2
            new_accumulator = rem_accumulator + remainder
            
            return half, new_accumulator
            

        def should_renege_using_tchange(lambda1, mu1, lambda2, mu2, T_local, N):
            """
            Decide whether to renege based on comparing T_local (local processing time)
            to the expected time_between_changes for two parallel M/M/1 queues.
    
            Reneging rule: If BOTH queues' expected time between changes >= T_local,
            then local processing is faster on average, so renege. Otherwise, stay.
    
            Parameters:
                lambda1, mu1 : floats
                    Arrival and service rates for queue 1.
                lambda2, mu2 : floats
                    Arrival and service rates for queue 2.
                T_local      : float
                    Local processing time.
                N            : int
                    Truncation cutoff for computing steady-state probabilities.
    
            Returns:
                renege        : bool
                    True if both T_change_1 >= T_local and T_change_2 >= T_local.
                t_change_1    : float
                    Expected time between changes for queue 1.
                t_change_2    : float
                    Expected time between changes for queue 2.
            """
            t_change_1 = compute_expected_time_between_changes(lambda1, mu1, N)
            t_change_2 = compute_expected_time_between_changes(lambda2, mu2, N)
    
            # Reneging if both queues change too slowly (i.e., T_change >= T_local)
            renege = (t_change_1 >= T_local) and (t_change_2 >= T_local)
            
            return renege, t_change_1, t_change_2


            # Parameters for the two queues
            #lambda1, mu1 = 2.0, 5.0
            #lambda2, mu2 = 3.0, 6.0
            #T_local = 0.2  # local processing time
                
    
            #print(f"Queue 1: expected time between changes = {t_change_1:.4f} sec")
            #print(f"Queue 2: expected time between changes = {t_change_2:.4f} sec")
            #print(f"Should renege to local? {renege}")



        #def decide_renege_with_local():
        """
        1) Compare Exp(mu1) vs Exp(mu2) in FSD sense to find the 'best' queue.
        2) Compare that best queue's CDF to the local deterministic CDF (step at T_local).
           If local FSD-dominates the best queue, return True (renege), else False.
    
        Parameters:
            mu1, mu2 : float
                The service rates for queue 1 and queue 2 (exponential waits).
            T_local  : float
                Deterministic local processing time.
            t_max    : float
                Max time up to which to sample the continuous distributions.
            num_points : int
                Number of points in [0, t_max] to sample.
    
        Returns:
            renege   : bool
                True  = both queues are stochastically worse than local → renege.
                False = at least one queue is not worse than local → stay.
            best_q   : int
                1 if queue 1 FSD‐dominates queue 2,
                2 if queue 2 FSD‐dominates queue 1,
                0 if neither strictly dominates (tie broken by mean wait).
            t_vals   : np.ndarray
                The time grid used for comparison.
            cdf1, cdf2 : np.ndarray
                Arrays of F1(t), F2(t) over t_vals.
        """
        
        c = 2
        
        if "1" in queueid :
            serv_rate = self.get_service_rates(queueid) # self.srvrates_1 # self.dict_servers_info["1"] get_service_rate
            queue =  self.dict_queues_obj["1"]   
            alt_queue_id = "2"   
            curr_arriv_rate = self.objQueues.get_arrivals_rates()
            #queue_intensity = curr_arriv_rate/ serv_rate 
            alt_queue_state = self.get_queue_state(alt_queue_id)
            alt_interchange_time = alt_queue_state["sample_interchange_time"] 
            # T_local = stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=0.75/req.pos_in_queue) # queue_interchange_time
            #self. max_cloud_delay = self.calculate_max_cloud_delay(req.pos_in_queue, queue_intensity, req)  
            #self.max_cloud_delay=stats.erlang.ppf(self.certainty, loc=0, scale=curr_arriv_rate, a=req.pos_in_queue)
            T_local = self.generateLocalCompUtility(req)
            
            # use Little's law to compute expected wait in alternative queue
            # instead of Little's Law, use the expected waiting time in steady state using (1 - queue_intensity e^{-(serv_rate - curr_arriv_rate)})
            #expected_wait_in_alt_queue = 1 - queue_intensity * math.exp(-(serv_rate - curr_arriv_rate)) # float(len(self.dict_queues_obj["2"])/curr_arriv_rate)
            #T_queue = expected_wait_in_alt_queue # queue_interchange_time + expected_wait_in_alt_queue
        else:
            serv_rate = self.get_service_rates(queueid) # self.srvrates_2 # self.dict_servers_info["2"] 
            queue =  self.dict_queues_obj["2"]
            alt_queue_id = "1"
            curr_arriv_rate = self.objQueues.get_arrivals_rates()
            #queue_intensity = curr_arriv_rate/ serv_rate    
            alt_queue_state = self.get_queue_state(alt_queue_id)  
            alt_interchange_time = alt_queue_state["sample_interchange_time"]  
            # T_local = stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=0.75/req.pos_in_queue)  
            T_local = self.generateLocalCompUtility(req)                                                         
        
        anchor = getattr(self, "jockey_anchor", "steady_state_distribution")
        mu1 = self.get_service_rates("1") # self.srvrates_1 # self.dict_servers_info["1"]
        mu2 = self.get_service_rates("2") # self.srvrates_2 # self.dict_servers_info["2"]
        
        if anchor == "steady_state_distribution":
            
            # Use steady-state distribution comparison (as currently implemented)
            # queue_states_compared = mmc_wait_cdf_and_tail(curr_arriv_rate, serv_rate, c, t): #self.compare_queues(alt_steady_state, curr_steady_state, K=1)
            #if queue_states_compared['first_order_dominance']:
            # If best_cdf[idx_T] <= eps, that means best_cdf(t)=0 (within tol) for all t<T_local
            # so local CDF(t)=0 >= 0 = best_cdf(t) for t<T_local, and at t>=T_local local CDF=1 >= best_cdf(t).
            # Therefore local FSD-dominates the best queue.                          
         
            # 1) Build time grid
            t_max = 10
            num_points = len(queue)
            t_vals = np.linspace(0, t_max, num_points)
    
            # 2) Compute CDFs for both queues on t_vals
            lambda1, lambda2 = arrival_rates_divisor(curr_arriv_rate, mu1, mu2)
            # 2. Compute CDFs for each queue at all t
            cdf1 = np.zeros_like(t_vals)
            cdf2 = np.zeros_like(t_vals)
            
            cdf1 = 1 - np.exp(-self.srvrates_1 * t_vals)  
            cdf2 = 1 - np.exp(-self.srvrates_2 * t_vals) 
            #for idx, t in enumerate(t_vals):
            #    cdf1,_ = 1 - np.exp(-mu1 * t_vals) # mmc_wait_cdf_and_tail(lambda1, mu1, c, t) #[0] # return only the cdf as a float from the turple
            #    cdf2,_ = 1 - np.exp(-mu2 * t_vals) # mmc_wait_cdf_and_tail(lambda2, mu2, c, t) #[0] # 1 - np.exp(-mu2 * t_vals)  # Exp(mu2)
    
            eps = 1e-6  # small tolerance
    
            # 3) Check FSD: Q1 ≥_FSD Q2 if CDF1(t) >= CDF2(t) for all t
            q1_dominates = np.all(cdf1 > cdf2) # + eps 
            q2_dominates = np.all(cdf2 > cdf1) #  + eps
            
            #print("\n DOMINANCE ", q1_dominates ,"  = = = " , q2_dominates)
    
            if q1_dominates and not q2_dominates:
                #print("\n **** SS 1 **** ", q1_dominates, self.srvrates_1)
                best_queue = "1"
                best_cdf = cdf1
            elif q2_dominates and not q1_dominates:
                #print("\n **** SS 2 **** ", q2_dominates, self.srvrates_2)
                best_queue = "2"
                best_cdf = cdf2
            else:
                best_queue = "0"
                best_cdf = T_local
                # Tie‐break by comparing mean waiting times: mean(Exp(mu)) = 1/mu
                #mean1 = 1.0 / mu1
                #mean2 = 1.0 / mu2
                #if mean1 < mean2:
                #    best_queue = "1"
                #    best_cdf = cdf1
                #else:
                #    best_queue = "2"
                #   best_cdf = cdf2
                    
            # print("\n BEST => ", best_queue)   
            # 4. Compare best queue's CDF to local's step‐CDF
            #    CDF_local(t) = 0 for t < T_local, 1 for t >= T_local.
            #    For local to FSD-dominate best queue, we need
            #      CDF_local(t) >= best_cdf(t) for all t in [0, t_max].
            #    That forces best_cdf(t) = 0 for all t < T_local, and best_cdf(t) <= 1 for t >= T_local.
            #    In practice, check best_cdf at t just below T_local.
            if "0" in best_queue:  
                              
                # Get the relevant queue
                queue = self.dict_queues_obj[queueid]
    
                # Find the request's position in the queue (0-based)
                found = False
                for pos, req_in_queue in enumerate(queue):
                    #print("\n **** SS 0 **** ", req_in_queue.customerid, " **** ",customer_id )
                    if customer_id in req_in_queue.customerid:
                    # Get the current service rate for the queue
                        if "1" in queueid:
                            serv_rate = self.get_service_rates(queueid) #self.srvrates_1  # self.dict_servers_info["1"] get_service_rate
                        else:
                            serv_rate = self.get_service_rates(queueid) # self.srvrates_2  # self.dict_servers_info["2"]

                        # Compute remaining waiting time (number of requests ahead * average service time)
                        # For M/M/1, expected remaining time = position * 1/service_rate
                        remaining_wait_time = pos * (1.0 / serv_rate) if serv_rate > 0 else 1e4  # avoid zero division

                        # If remaining wait exceeds T_local, renege
                        renege = (remaining_wait_time > T_local)
                        #print("\n I took the remaining time -> ", renege)
                        if renege:
                            decision = True
                            self.reqRenege(req_in_queue, queueid, pos, serv_rate, T_local, req_in_queue.customerid, req_in_queue.service_time, decision, queue)
                            found = True
                            break

                if not found:
                    print(f"Request ID {customer_id} not found in queue {queueid}. Continuing with processing...")
                    return False

                # If neither queue strictly dominates, we compare both to local.
                # We must have both best options < local everywhere → renege.
                # But since neither dominates, we just check both tails at T_local.
                #_, tail1 = mmc_wait_cdf_and_tail(lambda1, mu1, c, T_local)
                #_, tail2 = mmc_wait_cdf_and_tail(lambda2, mu2, c, T_local)
                # If both P(wait > T_local) > 0 => both CDFs at T_local < 1 => local FSD dominates both.
                #renege = (tail1 > T_local)  and (tail2 > T_local)  #(tail1 > eps) and (tail2 > eps)
                #print("\n ***** Renege to local? ", renege)
            # else:
                #idx_T = np.searchsorted(t_vals, T_local, side='right') - 1
            
                #if idx_T < 0:
                #    idx_T = 0
                
                #if isinstance(best_cdf, float):
                #    renege = (best_cdf <= eps)
                #else:
                #    renege = (best_cdf[idx_T] <= eps)
                
                #"""
                #Returns True if the remaining waiting time for the request exceeds T_local,
                #indicating the request should renege to local processing.
                #"""
                # Get the relevant queue
                #queue = self.dict_queues_obj[queueid]
    
                # Find the request's position in the queue (0-based)
                #for pos, req in enumerate(queue):
                #    if req.customerid == customer_id:
                #        break
                #    else:
                #        # Request not found
                #        return False

                # Get the current service rate for the queue
                #if "1" in queueid:
                #    serv_rate = self.dict_servers_info["1"]
                #else:
                #    serv_rate = self.dict_servers_info["2"]

                # Compute remaining waiting time (number of requests ahead * average service time)
                # For M/M/1, expected remaining time = position * 1/service_rate
                #remaining_wait_time = pos * (1.0 / serv_rate) if serv_rate > 0 else 1e4  # avoid zero division

                # If remaining wait exceeds T_local, renege
                #renege = (remaining_wait_time > T_local)
                #print("\n I took the remaining time -> ", renege)
                # return renege
                
               
                #if renege:
                #    decision = True
                #    self.reqRenege( req, queueid, curr_pose, serv_rate, queue_intensity, T_local, req.customerid, req.service_time, decision, queue)
            
            # For high throughput as objective, a low interchange_time shows a better state, if stability is the objective, a high value is better                       
        elif anchor == "inter_change_time":
			
            #print("\n ====  ICD ==== ")
            # Use interchange time
            # Example parameters
            #lambda_ = 2.0     # arrival rate
            #mu = 5.0          # service rate

            # Compute values
            #rate = compute_rate_of_change(lambda_, mu)
            #time_between_changes = compute_expected_time_between_changes(lambda_, mu)
            lambda1, lambda2 = arrival_rates_divisor(curr_arriv_rate, mu1, mu2)
            
            renege, t_change_1 , t_change_2 = should_renege_using_tchange(lambda1, mu1, lambda2, mu2, T_local, 100)

            #print(f"Rate of queue length change: {rate:.4f} events/sec")
            print(f"Expected time between changes in Queue 1: {t_change_1:.4f} sec and Expected time between changes in Queue 2: {t_change_2:.4f} sec")
            found = False
            for pos, req_in_queue in enumerate(queue):
                #print("\n **** SS 0 **** ", req_in_queue.customerid, " **** ",customer_id )
                if customer_id in req_in_queue.customerid:
                # Get the current service rate for the queue
                    if "1" in queueid:
                        serv_rate = self.get_service_rates(queueid) # self.srvrates_1  # self.dict_servers_info["1"] get_service_rate
                        remaining_wait_time = pos * t_change_1 
                    else:
                        serv_rate = self.get_service_rates(queueid) # self.srvrates_2  # self.dict_servers_info["2"]
                        remaining_wait_time = pos * t_change_2 
                    # Compute remaining waiting time (number of requests ahead * average service time)
                    # For M/M/1, expected remaining time = position * 1/service_rate
                    # remaining_wait_time = pos * (1.0 / serv_rate) if serv_rate > 0 else 1e4  # avoid zero division

                    # If remaining wait exceeds T_local, renege
                    renege = (remaining_wait_time > T_local)
            
                    if renege: #alt_queue_state["sample_interchange_time"] > curr_queue_state["sample_interchange_time"]:
                        decision = True 
                        self.reqRenege( req, queueid, pos, serv_rate, T_local, req.customerid, req.service_time, decision, queue)
                
            if not found:
                    print(f"Request ID {customer_id} not found in queue {queueid}. Continuing with processing...")
                    return False

      
    def makeRenegingDecision_original(self, req, queueid):
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
            
            if "1" in queueid:
                self.queue = self.dict_queues_obj["1"]            
            else:
                self.queue = self.dict_queues_obj["2"] 
        
            if self.max_local_delay <= self.max_cloud_delay: # will choose to renege
                decision=True
                curr_pose = self.get_request_position(queueid, req.customerid)
        
                if curr_pose is None: #  if curr_pose >= len(self.queue): # 
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


    def reqRenege(self, req, queueid, curr_pose, serv_rate, time_local_service, customerid, time_to_service_end, decision, curr_queue):
        
        if decision:
            self.record_user_reaction(queueid, "renege")
            
        if "Server1" in queueid:
            self.queue = self.dict_queues_obj["1"]            
        else:
            self.queue = self.dict_queues_obj["2"] 
            
        if curr_pose >= len(curr_queue):
            return
            
        else:
            # print("\n Error: ** ", len(self.queue), curr_pose)
            if len(self.queue) > curr_pose:
                self.queue = np.delete(self.queue, curr_pose) # index)  
                self.log_request(req.time_entrance, "reneged", req.time_res) # , self.queue) # req.queue)    arrival_time=    outcome= exit_time= queue=
                #print("\n ********* ", request_log[0])
                self.queueID = queueid  
        
                req.customerid = req.customerid+"_reneged"
        
                # In the case of reneging, you only get a reward if the time.entrance plus
                # the current time minus the time _to_service_end is greater than the time_local_service
        
                reward = self.getRenegeRewardPenalty(req, time_local_service, time_to_service_end)                                    
        
                self.objObserv.set_renege_obs(curr_pose, decision,time_local_service, time_to_service_end, reward, queueid, "reneged", len(curr_queue))
        
                self.curr_obs_renege.append(self.objObserv.get_renege_obs(queueid, self.queue)) #queueid, queue_intensity, curr_pose))        
                self.history.append(self.objObserv.get_renege_obs(queueid, self.queue))
                self.curr_req = req
        
                self.objQueues.update_queue_status(queueid)


    def get_request_position(self, queue_id, request_id): ######
        """
        Get the position of a given request in the queue.
        
        :param queue_id: The ID of the queue (1 or 2).
        :param request_id: The ID of the request.
        :return: The position of the request in the queue (0-indexed).
        """
        if "1" in queue_id:
            queue = self.dict_queues_obj["1"]  # Queue1
            #for t in queue:
            #print("\n -> ", request_id ,t.customerid)
        else:
            queue = self.dict_queues_obj["2"]  
            #for j in queue:
            #print("\n => ", request_id,j.customerid)
				
        for position, req in enumerate(queue):            
            if request_id in req.customerid:
                return position
            else:
                continue	

        #return None
    
        
    def compare_steady_state_distributions(self, dist_alt_queue, dist_curr_queue): # log_request
		        
        min1 = dist_alt_queue['min']
        max1 = dist_alt_queue['max']
        mean1 = dist_alt_queue['mean']
        
        min2 = dist_curr_queue['min']
        max2 = dist_curr_queue['max']
        mean2 = dist_curr_queue['mean']
        
        print("\n => ", min1, max1, mean1, "\n *** ", min2, max2, mean2)

        # 1) Pareto‐dominance
        le = (min1<=min2) and (max1<=max2) and (mean1<=mean2)
        lt = (min1< min2) or (max1< max2) or (mean1< mean2)
        if le and lt:
            return True # The alternative queue has a better steady state distribution "Queue1 strictly dominates Queue2"
        else:
            return False
     
            
    def reqJockey(self, curr_queue_id, dest_queue_id, req, customerid, serv_rate, dest_queue, exp_delay, decision, curr_pose, curr_queue):
		
        from termcolor import colored
        
        if decision:
            self.record_user_reaction(curr_queue_id, "jockey")
                
        if curr_pose >= len(curr_queue):
            return
            
        else:	
            np.delete(curr_queue, curr_pose) # np.where(id_queue==req_id)[0][0])
            self.log_request(req.time_entrance, "reneged", req.time_res ) #, curr_queue) # req.queue) arrival_time= outcome= exit_time= queue=
            
            reward = 1.0
            req.time_entrance = self.time # timer()
            dest_queue = np.append( dest_queue, req)
        
            self.queueID = curr_queue_id        
        
            req.customerid = req.customerid+"_jockeyed"
        
            if curr_queue_id == "1": # Server1
                queue_intensity = self.arr_rate/self.get_service_rates(curr_queue_id) # self.dict_servers_info["1"] # Server1
            
            else:
                queue_intensity = self.arr_rate/self.get_service_rates(curr_queue_id) # self.dict_servers_info["2"] # Server2
        
            reward = self.get_jockey_reward(req)                             
            print(colored("%s", 'green') % (req.customerid) + " in Server %s" %(curr_queue_id) + " jockeying now, to Server %s" % (colored(dest_queue_id,'green')))                      
            
            self.objObserv.set_jockey_obs(curr_pose,  decision, exp_delay, req.exp_time_service_end, reward, 1.0, "jockeyed", len(curr_queue)) # time_alt_queue        
            self.curr_obs_jockey.append(self.objObserv.get_jockey_obs(curr_queue_id, queue_intensity, curr_pose)) 
            self.history.append(self.objObserv.get_jockey_obs(curr_queue_id, queue_intensity, curr_pose))                                     
            self.curr_req = req        
            self.objQueues.update_queue_status(curr_queue_id) # long_avg_serv_time
        
        return
        
    # Add a method to record "wait" when the user stays in queue
    def record_wait_action(self, queueid):
        self.record_user_reaction(queueid, "wait")

    
    def compare_queues(self, pi1, pi2, K):
        # pi1, pi2: arrays of steady-state probabilities
        # 1) mean
        pi1 = np.array(list(pi1.values()))
        pi2 = np.array(list(pi2.values()))
        mean1, mean2 = np.dot(np.arange(len(pi1)), pi1), np.dot(np.arange(len(pi2)), pi2)
        # 2) tail P(Q > K)
        #tail1, tail2 = pi1[K+1:].sum(), pi2[K+1:].sum()
        # 3) stochastic dominance check (FSD)
        cdf1 = np.cumsum(pi1)
        cdf2 = np.cumsum(pi2)
        fsd = np.all(cdf1 >= cdf2) or np.any(cdf1 > cdf2) # and

        return {
            'mean': (mean1, mean2),
            #'P>{}'.format(K): (tail1, tail2),
            'first_order_dominance': fsd
        }
    
    
    def makeJockeyingDecision(self, req, curr_queue_id, alt_queue_id, customerid, serv_rate):
        # We make this decision if we have already joined the queue
        # First we analyse our current state -> which server, server intensity and expected remaining latency
        # Then we get information about the state of the alternative queue
        # Evaluate input from the actor-critic once we get in the alternative queue
        

        def erlang_C(c, rho):
            """
            Erlang‐C: probability that an arriving job must wait in an M/M/c queue.
            Requires rho < c for stability.
            """
            sum_terms = sum((rho**k) / factorial(k) for k in range(c))
            last_term = (rho**c / factorial(c)) * (c / (c - rho))
            return last_term / (sum_terms + last_term)

        def mm2_P_wait(lambda_i, mu_i):
            """
            Return P_wait for an M/M/2 queue with arrival lambda_i and service mu_i.
            If rho >= 2, returns 1.0 (unstable).
            """
            rho_i = lambda_i / mu_i
            if rho_i >= 2:
                return 1.0
            return erlang_C(2, rho_i)

        def compare_mmtwo_fsd(lambda1, mu1, lambda2, mu2, eps=1e-12):
            """
            Check FSD between two M/M/2 queues:
              Returns 1 if Q1 FSD‐dominates Q2,
                      2 if Q2 FSD‐dominates Q1,
                      0 otherwise.
            """
            P1 = mm2_P_wait(lambda1, mu1)
            P2 = mm2_P_wait(lambda2, mu2)
            alpha1 = 2*mu1 - lambda1
            alpha2 = 2*mu2 - lambda2
    
            cond1 = (P1 <= P2 + eps) and (alpha1 >= alpha2 - eps)
            cond2 = (P2 <= P1 + eps) and (alpha2 >= alpha1 - eps)
    
            if cond1 and not cond2:
                return 1
            elif cond2 and not cond1:
                return 2
            else:
                return 0

        def should_jockey_flag(current_queue, lambda1, mu1, lambda2, mu2):
            """
            Given the current queue index (1 or 2) and each queue's (lambda, mu),
            return True if a job in current_queue should jockey to the other queue,
            based on FSD comparison of waiting‐time distributions; else False.
    
            Parameters:
              current_queue : int (1 or 2)
              lambda1, mu1  : rates for queue 1
              lambda2, mu2  : rates for queue 2
    
            Returns:
              jockey_flag : bool
            """
            fsd_result = compare_mmtwo_fsd(lambda1, mu1, lambda2, mu2)
            # If current is 1 and Queue 2 dominates, jockey
            if  "1" in current_queue and fsd_result == 2:
                return True
            # If current is 2 and Queue 1 dominates, jockey
            if "2" in current_queue and fsd_result == 1:
                return True
                
            return False
            
        def arrival_rates_divisor(arrival_rate, mu1, mu2):
			# if the arrival rate is an odd number, divide it by two and 
			# add the reminder to the queue with the higher service rate
			# Else equal service rates
			
            """
            Divide n by 2. If n is odd, add its remainder (1) to rem_accumulator.
    
            Parameters:
                n (int): The integer to divide.
                rem_accumulator (int): The variable to which any odd remainder is added.
    
            Returns:
                tuple:
                    half (int): Result of integer division n // 2.
                    new_accumulator (int): Updated rem_accumulator.
            """
            if mu1 < mu2:
                rem_accumulator = mu1
            else:
                rem_accumulator = mu2
				
            remainder = arrival_rate % 2
            half = arrival_rate // 2
            new_accumulator = rem_accumulator + remainder
            
            return half, new_accumulator	
            
        def best_queue_delay(lambda1, mu1, lambda2, mu2): # 
            """
            Return the expected waiting-time-in-queue (delay) of the FSD-best queue.
            If neither strictly FSD-dominates, return the smaller mean wait of the two.
            """
            fsd_result = compare_mmtwo_fsd(lambda1, mu1, lambda2, mu2)
            w1 = mm2_mean_wait(lambda1, mu1)
            w2 = mm2_mean_wait(lambda2, mu2)
    
            if fsd_result == 1:
                return w1
            elif fsd_result == 2:
                return w2
            else:
                return min(w1, w2)	
                	
        
        if "1" in curr_queue_id:
            self.queue = self.dict_queues_obj["1"]  # Server1     
            serv_rate = self.get_service_rates(curr_queue_id) #self.dict_servers_info["1"]          get_service_rate    
            alt_queue_id = "2"   
            curr_arriv_rate = self.objQueues.get_arrivals_rates()      
        else:
            self.queue = self.dict_queues_obj["2"]
            serv_rate = self.get_service_rates(curr_queue_id) # self.dict_servers_info["2"]              
            alt_queue_id = "1"   
            curr_arriv_rate = self.objQueues.get_arrivals_rates() 
            
        if "1" in curr_queue_id: # curr_queue_id == "1":
            curr_queue_state = self.get_queue_state("1")
            alt_queue_state = self.get_queue_state("2")
        else:
            curr_queue_state = self.get_queue_state("2")
            alt_queue_state = self.get_queue_state("1")

        # Use steady_state_distribution and simulated_events
        curr_steady_state = curr_queue_state["steady_state_distribution"]
        alt_steady_state = alt_queue_state["steady_state_distribution"]

        #curr_simulated_events = curr_queue_state["simulated_events"]
        #alt_simulated_events = alt_queue_state["simulated_events"]
                
        decision=False                
        # queue_intensity = self.arr_rate/self.dict_servers_info[alt_queue_id]
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
            # disabling the waiting time based jockeying behavior and letting the server state decide
            #if time_to_get_served > self.avg_delay:
            #    decision = True
            #    self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.dict_queues_obj.get(curr_queue_id)) #self.queue)
            
            '''
               Observe the state of the current queue and compare that with the state of the
               other queue and jockey if the other queue is better than the current one.
               The better state is defined by first-order stochatsic dorminance and the jockeying rate (orprobability)
            ''' 
                        
            #queue_states_compared = self.compare_queues(alt_steady_state, curr_steady_state, K=1) 
            
            anchor = getattr(self, "jockey_anchor", "steady_state_distribution")
            
            if anchor == "steady_state_distribution":
                # Use steady-state distribution comparison (as currently implemented)
                #queue_states_compared = mmc_wait_cdf_and_tail(curr_arriv_rate, serv_rate, c, t): #self.compare_queues(alt_steady_state, curr_steady_state, K=1)
                lambda1, lambda2 = arrival_rates_divisor(curr_arriv_rate, self.get_service_rates("1"), self.get_service_rates("2")) #"1"], self.dict_servers_info["2"])
                # current_queue, lambda1, mu1, lambda2, mu2
                jockey_flag = should_jockey_flag(curr_queue_id, lambda1, self.get_service_rates("1"), lambda2, self.get_service_rates("2"))
                if jockey_flag: #queue_states_compared['first_order_dominance']:
                    decision = True
                    self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.dict_queues_obj.get(curr_queue_id))
            
            # For high throughput as objective, a low interchange_time shows a better state, if stability is the objective, a high value is better                       
            elif anchor == "inter_change_time":
                # Use interchange time
                if alt_queue_state["sample_interchange_time"] > curr_queue_state["sample_interchange_time"]:
                    decision = True 
                    self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.dict_queues_obj.get(curr_queue_id))
                    
            #if queue_states_compared['first_order_dominance']: #and (alt_queue_state['jockeying_rate'] <  curr_queue_state['jockeying_rate']):
            
            # For high throughput as objective, a low interchange_time shows a better state, if stability is the objective, a high value is better 
            ## if alt_queue_state['sample_interchange_time'] > curr_queue_state['sample_interchange_time']: 
                #alt_queue_state['long_avg_serv_time'] < time_to_get_served:
                # alt_queue_state['total_customers'] < curr_queue_state['total_customers']: # 'long_run_change_rate':       
            #    decision = True
            #    self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.dict_queues_obj.get(curr_queue_id))
                
            #elif len(alt_simulated_events) < len(curr_simulated_events):  # Fewer events might indicate less congestion
                #decision = True
                #self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.dict_queues_obj.get(curr_queue_id))

        return decision
        
    

class MarkovQueueModel:
    """
    Markovian M/M/1 queue model for queue-length change dynamics.
    """

    def __init__(self, arrival_rate, service_rate, max_states=1000):
        """
        Initialize the model.
        
        Parameters:
        - arrival_rate (λ): Poisson arrival rate
        - service_rate (μ): Exponential service rate
        - max_states: Number of states to approximate infinite sum
        """
        #print("\n ====> ", service_rate)
        #self.lambda_ = arrival_rate
        #if isinstance(None, type(service_rate)):
            # Option 1: Set to a safe small positive value, or raise an explicit error
        #    self.mu = 1e-2
            #raise ValueError("service_rate must not be None in MarkovQueueModel")
        #else:
        #    self.mu = service_rate
        
        #if abs(service_rate) < 1e-8:
        #    self.rho = 0.0  # fallback value; choose np.inf or another value if more appropriate for your model
        #else:
        #    self.rho = arrival_rate / service_rate
            
        #self.max_states = max_states
        #self.pi = self._steady_state_distribution()
               
    
    def _steady_state_distribution(self):
        """
        Compute steady-state distribution π_n for n = 0..max_states
        for an M/M/1 queue: π_n = (1-ρ) ρ^n        
        """
        
        # Validate rho to ensure it's within a valid range
        #if not (0 < self.rho < 1):
        #    raise ValueError("Invalid configuration: rho (arrival rate / service rate) must be between 0 and 1.")

        # Calculate the steady-state distribution
        pi = np.zeros(self.max_states + 1)
        one_minus_rho = abs(1 - self.rho)

        n = np.arange(self.max_states + 1)
        pi = one_minus_rho * (self.rho ** n)

        # Normalize to handle potential truncation errors
        total_sum = np.sum(pi)
        if total_sum > 0:
            pi /= total_sum
        else:
            # Fallback to uniform distribution if normalization fails
            pi = np.full(self.max_states + 1, 1.0 / (self.max_states + 1))

        return pi

    
    def state_change_rate(self, n):
        """
        Instantaneous rate of queue length changes in state n:
          γ_n = λ + μ_n
        where μ_n = μ when n>=1, else 0.
        """
        mu_n = self.mu if n >= 1 else 0.0
        return self.lambda_ + mu_n
    
    def long_run_change_rate(self):
        """
        Long-run average rate of queue-length changes:
          R_change = sum_n π_n (λ + μ_n)
        """
        if self.pi is None or len(self.pi) == 0:
            raise ValueError("Steady-state distribution is not initialized or invalid.")

        n = np.arange(self.max_states + 1)
        mu_n = np.where(n >= 1, self.mu, 0.0)
        gamma_n = self.lambda_ + mu_n

        long_run_rate = np.dot(self.pi, gamma_n)

        # Ensure the result is valid and non-zero
        if not np.isfinite(long_run_rate) or long_run_rate <= 0:
            # Fallback: Use a small positive constant to avoid zero
            long_run_rate = max(1e-6, self.lambda_)

        return long_run_rate
    
    def sample_interchange_time(self):
        """
        Sample time until next queue-length change in steady state:
          Exp(R_change)
        """
        try:
            rate = self.long_run_change_rate()
        except ValueError:
            # Fallback to a default positive rate if an exception occurs
            rate = max(1e-6, self.lambda_)

        # Ensure the sampling rate is positive
        if rate <= 0:
            rate = 1e-6

        return np.random.exponential(1.0 / rate)
                    

    def compute_steady_state_probs(self, rho, N=100):
        """Compute steady-state probabilities for M/M/1 with truncation at N."""
        return np.array([(1 - rho) * rho**n for n in range(N + 1)])

    def compute_rate_of_change(self, lambda_, mu, N=100):
        """Compute average rate at which queue length changes."""
        
        if isinstance(None, type(mu)):
            # Option 1: Set to a safe small positive value, or raise an explicit error
            mu = 1e-1
        
        if isinstance(None, type(lambda_)):
            lambda_ = 1e-2
            
        rho = lambda_ / mu
        pi = self.compute_steady_state_probs(rho, N)
        R_change = sum(pi[n] * (lambda_ + mu if n > 0 else lambda_) for n in range(N + 1))
        return R_change
    
        #rho = lambda_ / mu
        #pi = self.compute_steady_state_probs(rho, N)
    
        # Compute rate of change: λ for all n, and μ only if n > 0
        #R_change = sum(pi[n] * (lambda_ + mu if n > 0 else lambda_) for n in range(N + 1))
        #return R_change

    def compute_expected_time_between_changes(self, lambda_, mu, N=100):
		
        """Compute expected time between changes in queue length."""
        
        R_change = self.compute_rate_of_change(lambda_, mu, N)
        if R_change is None or not np.isfinite(R_change) or R_change <= 0:
            return 1/mu  # Or use another suitable large value
    
        T_change = 1 / R_change 
        
        return T_change
       


class MarkovModulatedServiceModel:
    """
    Models a time-varying service rate μ(t) as a continuous-time Markov chain (CTMC)
    over discrete states, each with an exponential service time distribution.
    """
    
    def __init__(self, mu_states, Q):
        """
        Parameters:
        - mu_states: array-like of shape (K,) of service rates μ_i for each CTMC state
        - Q: transition rate matrix of shape (K, K) for the CTMC (rows sum to zero)
        """
        self.mu_states = np.array(mu_states)
        self.Q = np.array(Q)
        self.num_states = len(mu_states)
        
        # Precompute cumulative exit rates and transition probabilities
        self.exit_rates = -np.diag(self.Q)  # rates λ_i = -q_{ii}
        self.trans_probs = np.zeros_like(self.Q)
        for i in range(self.num_states):
            if self.exit_rates[i] > 0:
                self.trans_probs[i] = self.Q[i] / self.exit_rates[i]
                self.trans_probs[i, i] = 0  # no self-transition
        
        # Initialize state
        self.current_state = 0
        self.current_time = 0.0
    
    def step(self):
        """
        Advance the CTMC to the next state and return the jump time.
        """
        i = self.current_state
        rate = self.exit_rates[i]
        if rate <= 0:
            # absorbing or no exit
            return np.inf
        
        # Sample time to next jump
        jump_time = np.random.exponential(1.0 / rate)
        # Choose next state
        probs = self.trans_probs[i]
        j = np.random.choice(self.num_states, p=probs)
        
        # Update state and time
        self.current_time += jump_time
        self.current_state = j
        return jump_time
    
    def sample_service_time(self):
        """
        Sample a service time from Exp(mu_current) distribution at current_state.
        """
        mu = self.mu_states[self.current_state]
        if mu <= 0:
            return np.inf
        return np.random.exponential(1.0 / mu)
    
    def erlang_C(self, c, rho):
        """
        Erlang‐C: probability that an arriving job must wait in an M/M/c queue.
        Requires rho < c for stability.
        """
        #sum_terms = sum((rho**k) / factorial(k) for k in range(c))
        #last_term = (rho**c / factorial(c)) * (c / (c - rho))
        #return last_term / (sum_terms + last_term)
        # Handle unstable or critically loaded queue (rho >= c)
        
        if rho >= c:
            return 1.0
        sum_terms = sum((rho**k) / factorial(k) for k in range(c))
        denom = sum_terms + (rho**c / factorial(c)) * (c / (c - rho))
        if denom == 0:
            return 1.0  # Or raise an error if this is truly unexpected
        last_term = (rho**c / factorial(c)) * (c / (c - rho))
        
        return last_term / denom

    def mm2_P_wait(self, lambda_i, mu_i):
        """
        Return P_wait for an M/M/2 queue with arrival lambda_i and service mu_i.
        If rho >= 2, returns 1.0 (unstable).
        """
        #rho_i = lambda_i / mu_i
        #if rho_i >= 2:
        #    return 1.0
        #return self.erlang_C(2, rho_i)
        
        rho_i = lambda_i / mu_i if mu_i != 0 else float('inf')
        if rho_i >= 2 or mu_i == 0:
            return 1.0
            
        return self.erlang_C(2, rho_i)
        
    def mm2_mean_wait(self, lambda_i, mu_i):
        """
        Expected waiting time in queue (W_q) for M/M/2:
            W_q = P_wait / (2*mu_i - lambda_i)
        If rho >= 2, return np.inf.
        """
        if mu_i == 0:
            return 1e-2
            
        rho_i = lambda_i / mu_i
        if rho_i >= 2:
            return np.inf
        P_wait = self.mm2_P_wait(lambda_i, mu_i)
        return P_wait / (2*mu_i - lambda_i)

    def compare_mmtwo_fsd(self, lambda1, mu1, lambda2, mu2, eps=1e-12):
        """
        Check FSD between two M/M/2 queues:
          Returns 1 if Q1 FSD‐dominates Q2,
                  2 if Q2 FSD‐dominates Q1,
                  0 otherwise.
        """
        P1 = self.mm2_P_wait(lambda1, mu1)
        P2 = self.mm2_P_wait(lambda2, mu2)
        alpha1 = 2*mu1 - lambda1
        alpha2 = 2*mu2 - lambda2
    
        cond1 = (P1 <= P2 + eps) and (alpha1 >= alpha2 - eps)
        cond2 = (P2 <= P1 + eps) and (alpha2 >= alpha1 - eps)
   
        if cond1 and not cond2:
            return 1
        elif cond2 and not cond1:
            return 2
        else:
            return 0

    def arrival_rates_divisor(self, arrival_rate, mu1, mu2):
        # if the arrival rate is an odd number, divide it by two and 
		# add the reminder to the queue with the higher service rate
		# Else equal service rates
			
        """
        Divide n by 2. If n is odd, add its remainder (1) to rem_accumulator.
    
        Parameters:
            n (int): The integer to divide.
            rem_accumulator (int): The variable to which any odd remainder is added.
    
        Returns:
            tuple:
                half (int): Result of integer division n // 2.
                new_accumulator (int): Updated rem_accumulator.
        """
        if mu1 < mu2:
            rem_accumulator = mu1
        else:
            rem_accumulator = mu2
				
        remainder = arrival_rate % 2
        half = arrival_rate // 2
        new_accumulator = rem_accumulator + remainder
            
        return half, new_accumulator	
            
    def best_queue_delay(self, lambda1, mu1, lambda2, mu2):
        """
        Return the expected waiting-time-in-queue (delay) of the FSD-best queue.
        If neither strictly FSD-dominates, return the smaller mean wait of the two.
        """
        fsd_result = self.compare_mmtwo_fsd(lambda1, mu1, lambda2, mu2)
        w1 = self.mm2_mean_wait(lambda1, mu1)
        w2 = self.mm2_mean_wait(lambda2, mu2)
        
        if fsd_result == 1:
            return w1
        elif fsd_result == 2:
            return w2
        else:
            return min(w1, w2)


def extract_waiting_times_and_outcomes(request_queue):
    """
    Extracts waiting times and outcomes from request_queue.history.
    Returns waiting_times, outcomes, time_stamps (if available).
    """
    waiting_times = []
    outcomes = []
    time_stamps = []

    for obs in request_queue.history:
        waited = obs.get('Waited', None)
        # Try to get customerid; fallback to Action if needed
        customerid = obs.get('customerid', None)
        action = obs.get('Action', None)

        # Infer outcome
        if customerid:
            if '_reneged' in customerid:
                outcome = 'reneged'
            elif '_jockeyed' in customerid:
                outcome = 'jockeyed'
            else:
                outcome = 'served'
        elif action:
            outcome = action
        else:
            outcome = 'served'

        # Time stamp: if obs has 'time_exit', use it, else None
        time_exit = obs.get('time_exit', None)
        if waited is not None:
            waiting_times.append(waited)
            outcomes.append(outcome)
            time_stamps.append(time_exit)

    return waiting_times, outcomes, time_stamps
    

def plot_waiting_time_cdf(waiting_times, title="CDF of Waiting Times"):
    """
    Plots the cumulative distribution function (CDF) of waiting times.
    """
    waiting_times = np.sort(waiting_times)
    cdf = np.arange(1, len(waiting_times)+1) / len(waiting_times)
    plt.figure(figsize=(8,4))
    plt.plot(waiting_times, cdf, marker='.')
    plt.xlabel("Waiting Time")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.show()
    

def plot_boxplot_waiting_times_by_outcome(waiting_times, outcomes, title="Waiting Times by Outcome"):
    """
    Plots a boxplot of waiting times for each outcome.
    """
    df = pd.DataFrame({'Waiting Time': waiting_times, 'Outcome': outcomes})
    plt.figure(figsize=(8,4))
    df.boxplot(column='Waiting Time', by='Outcome')
    plt.title(title)
    plt.suptitle('')
    plt.xlabel("Outcome")
    plt.ylabel("Waiting Time")
    plt.grid(True)
    plt.show()
    

    
    
def Per_Outcome_Wasted_Waiting_Time_by_Interval():
	
    outcomes = sorted({o for d in per_outcome_by_interval for o in d.keys()})
    data = np.zeros((len(outcomes), len(intervals)))
    for j, per_outcome in enumerate(per_outcome_by_interval):
        for i, outcome in enumerate(outcomes):
            data[i, j] = per_outcome.get(outcome, 0.0)

    fig, ax = plt.subplots(figsize=(10,6))
    bottom = np.zeros(len(intervals))
    for i, outcome in enumerate(outcomes):
        ax.bar(intervals, data[i], label=outcome, bottom=bottom)
        bottom += data[i]
    ax.set_title("Per-Outcome Wasted Waiting Time by Interval")
    ax.set_xlabel("Interval (seconds)")
    ax.set_ylabel("Wasted Waiting Time")
    ax.legend()
    plt.show()


def plot_policy_history(policy_history):
    """
    Plot the evolution of service rate and expected utility over time.
    """
    if not policy_history:
        print("No policy history to plot.")
        return
    rates = [h["new_rate"] for h in policy_history]
    utils = [h["utility"] for h in policy_history]
    steps = list(range(len(policy_history)))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(steps, rates, marker='o')
    plt.ylabel("Service Rate")
    plt.title("Service Rate (Policy) Evolution")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(steps, utils, marker='x', color='purple')
    plt.xlabel("Time Step")
    plt.ylabel("Expected Utility")
    plt.title("Expected Utility Evolution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictive_model_history(model_history):
    """
    Plot diagnostics for the predictive model fit over time.
    """
    if not model_history:
        print("No predictive model fit history to plot.")
        return
    n_samples = [h["n_samples"] for h in model_history]
    steps = list(range(len(model_history)))
    plt.figure(figsize=(8, 4))
    plt.plot(steps, n_samples, marker="o")
    plt.title("Predictive Model Fit Sample Size Over Time")
    plt.xlabel("Model Fit Step")
    plt.ylabel("Number of Samples Used")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def in_old_function_main():
	
    waiting_times, outcomes, time_stamps = extract_waiting_times_and_outcomes(requestObj)
    plot_waiting_time_cdf(waiting_times)
    plot_boxplot_waiting_times_by_outcome(waiting_times, outcomes)
    plot_avg_waiting_time_over_time(waiting_times, time_stamps, window=10)
    
    # After all interval runs, plot the aggregated results
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot reneging rates
    for i, interval in enumerate(intervals):
        axs[0].plot(range(len(all_reneging_rates[i]["server_1"])), all_reneging_rates[i]["server_1"], label=f'Server 1 - Interval {interval}s')
        axs[0].plot(range(len(all_reneging_rates[i]["server_2"])), all_reneging_rates[i]["server_2"], label=f'Server 2 - Interval {interval}s', linestyle='--')

    axs[0].set_title('Reneging Rates Across Intervals')
    axs[0].set_ylabel('Reneging Rate')
    axs[0].legend()

    # Plot jockeying rates
    for i, interval in enumerate(intervals):
        axs[1].plot(range(len(all_jockeying_rates[i]["server_1"])), all_jockeying_rates[i]["server_1"], label=f'Server 1 - Interval {interval}s')
        axs[1].plot(range(len(all_jockeying_rates[i]["server_2"])), all_jockeying_rates[i]["server_2"], label=f'Server 2 - Interval {interval}s', linestyle='--')

    axs[1].set_title('Jockeying Rates Across Intervals')
    axs[1].set_xlabel('Number of Requests')
    axs[1].set_ylabel('Jockeying Rate')
    axs[1].legend()


def plot_six_panels(results, intervals, jockey_anchors): # results
    #fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=False)
    fig, axs = plt.subplots(2, len(intervals), figsize=(6*len(intervals), 8), sharex=False)
    colors = {
        "steady_state_distribution": "blue",
        "inter_change_time": "green"
    }
    linestyles = {
        "steady_state_distribution": "-",
        "inter_change_time": "--"
    }

    for col, interval in enumerate(intervals):
        # Reneging rate plots
        ax_ren = axs[0, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                y = np.array(results[interval]["reneging_rates"][anchor][server])
                x = np.arange(len(y))
                #y = np.array([results[interval]["reneging_rates"][anchor][server]])
                #x = np.arange(len(y))
                ax_ren.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
        ax_ren.set_title(f"Reneging Rates | Interval {interval}s")
        ax_ren.set_xlabel("Steps")
        ax_ren.set_ylabel("Reneging Rate")
        ax_ren.legend()

        # Jockeying rate plots
        ax_jky = axs[1, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                #y = np.array(results[interval]["jockeying_rates"][anchor][server])
                #x = np.arange(len(y))
                y = np.array([results[interval]["jockeying_rates"][anchor][server]])
                x = np.arange(len(y))
                ax_jky.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
        ax_jky.set_title(f"Jockeying Rates | Interval {interval}s")
        ax_jky.set_xlabel("Steps")
        ax_jky.set_ylabel("Jockeying Rate")
        ax_jky.legend()

    plt.tight_layout()
    plt.show()
 

'''
    Simulation step (or time, or customer index):
       This allows you to plot moving averages or windowed averages for each outcome (served, jockeyed, reneged) as the simulation progresses.
    
'''
def plot_all_avg_waiting_time_by_anchor_interval(
    histories_policy,
    histories_nopolicy,
    window=20,
    title="Average Waiting Time per Class (Policy vs No Policy)"
):
    import numpy as np
    import matplotlib.pyplot as plt

    anchors = sorted({h['anchor'] for h in histories_policy + histories_nopolicy})
    intervals = sorted({h['interval'] for h in histories_policy + histories_nopolicy})

    color_map = {
        'served': 'tab:green',
        'jockeyed': 'tab:blue',
        'reneged': 'tab:red'
    }
    style_map = {
        'Policy': '-',
        'No Policy': '--'
    }

    def get_class_waiting(history):
        class_waits = {'served': [], 'jockeyed': [], 'reneged': []}
        class_indices = {'served': [], 'jockeyed': [], 'reneged': []}
        for idx, obs in enumerate(history):
            waited = obs.get('Waited', None)
            cid = obs.get('customerid', '')
            if waited is not None:
                if '_reneged' in cid:
                    class_waits['reneged'].append(waited)
                    class_indices['reneged'].append(idx)
                elif '_jockeyed' in cid:
                    class_waits['jockeyed'].append(waited)
                    class_indices['jockeyed'].append(idx)
                else:
                    class_waits['served'].append(waited)
                    class_indices['served'].append(idx)
        return class_indices, class_waits

    for anchor in anchors:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
        fig.suptitle(f"Information model: {anchor}", fontsize=12)
        axs = axs.flatten()
        for j, interval in enumerate(intervals):
            if j >= 4:
                break  # Only plot first 4 intervals in a 2x2 grid
            ax = axs[j]
            policy_hist = next((h['history'] for h in histories_policy if h['anchor'] == anchor and h['interval'] == interval), [])
            nopolicy_hist = next((h['history'] for h in histories_nopolicy if h['anchor'] == anchor and h['interval'] == interval), [])

            indices_p, waits_p = get_class_waiting(policy_hist)
            indices_np, waits_np = get_class_waiting(nopolicy_hist)

            for outcome in ['served', 'jockeyed', 'reneged']:
                # Policy
                xs_p = np.array(indices_p[outcome])
                ys_p = np.array(waits_p[outcome])
                if len(xs_p) >= window:
                    ys_p_smooth = np.convolve(ys_p, np.ones(window)/window, mode='valid')
                    xs_p_smooth = xs_p[window-1:]
                elif len(xs_p) > 0:
                    ys_p_smooth = ys_p
                    xs_p_smooth = xs_p
                else:
                    continue # No data for this class in policy mode

                ax.plot(xs_p_smooth, ys_p_smooth,
                        label=f"Policy: {outcome.capitalize()}",
                        color=color_map[outcome],
                        linestyle=style_map['Policy'])

                # No Policy
                xs_np = np.array(indices_np[outcome])
                ys_np = np.array(waits_np[outcome])
                if len(xs_np) >= window:
                    ys_np_smooth = np.convolve(ys_np, np.ones(window)/window, mode='valid')
                    xs_np_smooth = xs_np[window-1:]
                elif len(xs_np) > 0:
                    ys_np_smooth = ys_np
                    xs_np_smooth = xs_np
                else:
                    continue # No data for this class in no-policy mode

                ax.plot(xs_np_smooth, ys_np_smooth,
                        label=f"No Policy: {outcome.capitalize()}",
                        color=color_map[outcome],
                        linestyle=style_map['No Policy'])

            ax.set_xlabel('Customer Index (Simulation Step)')
            if j % 2 == 0:
                ax.set_ylabel('Avg Waiting Time (Moving Avg)')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"Interval: {interval}")
            ax.legend(fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        

def plot_avg_wait_by_queue_length(
    histories_policy, histories_nopolicy, window=1, title="Average Waiting Time vs Queue Length"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    def extract_queue_length_and_wait(histories, outcome):
        xs, ys = [], []
        for h in histories:
            for obs in h['history']:
                if isinstance(obs, dict):
                    waited = obs.get('Waited', None)
                    cid = obs.get('customerid', '')
                    queue_length = obs.get('queue_length', None)
                    if waited is not None and queue_length is not None:
                        if (outcome == 'served' and not ('_reneged' in cid or '_jockeyed' in cid)) or \
                           (outcome == 'reneged' and '_reneged' in cid) or \
                           (outcome == 'jockeyed' and '_jockeyed' in cid):
                            xs.append(queue_length)
                            ys.append(waited)
                elif isinstance(obs, list):
                    for o in obs:
                        if not isinstance(o, dict):
                            continue
                        waited = o.get('Waited', None)
                        cid = o.get('customerid', '')
                        queue_length = o.get('queue_length', None)
                        if waited is not None and queue_length is not None:
                            if (outcome == 'served' and not ('_reneged' in cid or '_jockeyed' in cid)) or \
                               (outcome == 'reneged' and '_reneged' in cid) or \
                               (outcome == 'jockeyed' and '_jockeyed' in cid):
                                xs.append(queue_length)
                                ys.append(waited)
        return np.array(xs), np.array(ys)

    def binned_average(xs, ys, window=1):
        # Bin by queue length (integer), average within each bin
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[int(x)].append(y)
        bin_xs = sorted(bins.keys())
        bin_ys = [np.mean(bins[x]) for x in bin_xs]
        # Optionally smooth with moving average across queue lengths
        if window > 1 and len(bin_ys) >= window:
            bin_ys = np.convolve(bin_ys, np.ones(window)/window, mode='same')
        return np.array(bin_xs), np.array(bin_ys)

    color_map = {
        'served': 'tab:green',
        'jockeyed': 'tab:blue',
        'reneged': 'tab:red'
    }
    style_map = {
        'Policy': '-',
        'No Policy': '--'
    }

    for outcome in ['served', 'jockeyed', 'reneged']:
        plt.figure(figsize=(8,5))
        # Policy
        xs_policy, ys_policy = extract_queue_length_and_wait(histories_policy, outcome)
        bin_xs_p, bin_ys_p = binned_average(xs_policy, ys_policy, window)
        plt.plot(bin_xs_p, bin_ys_p, style_map['Policy'], label='Policy', color=color_map[outcome])
        plt.scatter(xs_policy, ys_policy, color=color_map[outcome], alpha=0.3, marker='o')

        # No Policy
        xs_np, ys_np = extract_queue_length_and_wait(histories_nopolicy, outcome)
        bin_xs_np, bin_ys_np = binned_average(xs_np, ys_np, window)
        plt.plot(bin_xs_np, bin_ys_np, style_map['No Policy'], label='No Policy', color=color_map[outcome])
        plt.scatter(xs_np, ys_np, color=color_map[outcome], alpha=0.3, marker='x')

        plt.xlabel('Queue Length')
        plt.ylabel('Average Waiting Time')
        plt.title(f"{title} ({outcome.title()})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()



def plot_avg_wait_by_queue_length_grouped(
    histories_policy, histories_nopolicy, window=1, title="Average Waiting Time vs Queue Length (Grouped)"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Only include 'jockeyed' and 'reneged'
    outcomes = ['jockeyed', 'reneged']

    def extract_by_interval(histories, outcome):
        by_interval = defaultdict(lambda: ([], []))
        for h in histories:
            interval = h.get('interval', None)
            if interval is None:
                continue
            for obs in h['history']:
                if isinstance(obs, dict):
                    waited = obs.get('Waited', None)
                    cid = obs.get('customerid', '')
                    queue_length = obs.get('queue_length', None)
                    if waited is not None and queue_length is not None:
                        if (outcome == 'reneged' and '_reneged' in cid) or \
                           (outcome == 'jockeyed' and '_jockeyed' in cid):
                            by_interval[interval][0].append(queue_length)
                            by_interval[interval][1].append(waited)
                elif isinstance(obs, list):
                    for o in obs:
                        if not isinstance(o, dict):
                            continue
                        waited = o.get('Waited', None)
                        cid = o.get('customerid', '')
                        queue_length = o.get('queue_length', None)
                        if waited is not None and queue_length is not None:
                            if (outcome == 'reneged' and '_reneged' in cid) or \
                               (outcome == 'jockeyed' and '_jockeyed' in cid):
                                by_interval[interval][0].append(queue_length)
                                by_interval[interval][1].append(waited)
        return by_interval

    def binned_average(xs, ys, window=1):
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[int(x)].append(y)
        bin_xs = sorted(bins.keys())
        bin_ys = [np.mean(bins[x]) for x in bin_xs]
        if window > 1 and len(bin_ys) >= window:
            bin_ys = np.convolve(bin_ys, np.ones(window)/window, mode='same')
        return np.array(bin_xs), np.array(bin_ys)

    color_map = {
        'Policy': 'tab:blue',
        'No Policy': 'tab:orange'
    }

    # Collect all intervals
    intervals_policy = set(h.get('interval', None) for h in histories_policy)
    intervals_nopolicy = set(h.get('interval', None) for h in histories_nopolicy)
    all_intervals = sorted([i for i in intervals_policy | intervals_nopolicy if i is not None])

    nrows = len(all_intervals)
    ncols = len(outcomes)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False, sharey='row')

    for row, interval in enumerate(all_intervals):
        for col, outcome in enumerate(outcomes):
            by_interval_policy = extract_by_interval(histories_policy, outcome)
            by_interval_nopolicy = extract_by_interval(histories_nopolicy, outcome)

            xs_policy, ys_policy = by_interval_policy.get(interval, ([], []))
            xs_np, ys_np = by_interval_nopolicy.get(interval, ([], []))
            bin_xs_p, bin_ys_p = binned_average(xs_policy, ys_policy, window)
            bin_xs_np, bin_ys_np = binned_average(xs_np, ys_np, window)

            ax = axs[row, col]
            # Policy curve
            if len(bin_xs_p) > 0:
                ax.plot(bin_xs_p, bin_ys_p, '-', label='Policy', color=color_map['Policy'])
            # No policy curve
            if len(bin_xs_np) > 0:
                ax.plot(bin_xs_np, bin_ys_np, '--', label='No Policy', color=color_map['No Policy'])

            if row == 0:
                ax.set_title(f"{outcome.title()}", fontsize=14)
            if col == 0:
                ax.set_ylabel(f"Interval: {interval}\nAverage Waiting Time", fontsize=12)
            ax.set_xlabel('Queue Length')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_all_avg_waiting_time_by_anchor_interval_original(
    histories_policy,
    histories_nopolicy,
    window=20,
    #title="Average Waiting Time per Class (Policy vs No Policy)"
):
    """
    For each anchor, for each interval, plot a comparison of average waiting times for
    served, jockeyed, and reneged requests (policy vs no-policy).

    Args:
        histories_policy: list of dicts with keys 'interval', 'anchor', 'history'
        histories_nopolicy: same as above
        window: moving average window
        title: plot title prefix
    """

    # Identify all anchors and intervals present
    anchors = sorted({h['anchor'] for h in histories_policy + histories_nopolicy})
    intervals = sorted({h['interval'] for h in histories_policy + histories_nopolicy})

    color_map = {
        'served': 'tab:green',
        'jockeyed': 'tab:blue',
        'reneged': 'tab:red'
    }
    style_map = {
        'Policy': '-',
        'No Policy': '--'
    }

    def get_class_waiting(history):
        class_waits = {'served': [], 'jockeyed': [], 'reneged': []}
        class_indices = {'served': [], 'jockeyed': [], 'reneged': []}
        for idx, obs in enumerate(history):
            waited = obs.get('Waited', None)
            cid = obs.get('customerid', '')
            if waited is not None:
                if '_reneged' in cid:
                    class_waits['reneged'].append(waited)
                    class_indices['reneged'].append(idx)
                elif '_jockeyed' in cid:
                    class_waits['jockeyed'].append(waited)
                    class_indices['jockeyed'].append(idx)
                else:
                    class_waits['served'].append(waited)
                    class_indices['served'].append(idx)
        return class_indices, class_waits

    for anchor in anchors:
        fig, axs = plt.subplots(1, len(intervals), figsize=(6*len(intervals), 5), sharey=True)
        if len(intervals) == 1:
            axs = [axs]
        #fig.suptitle(f"{title}\nAnchor: {anchor}", fontsize=16)

        for j, interval in enumerate(intervals):
            # Get the right history for this anchor/interval
            policy_hist = next((h['history'] for h in histories_policy if h['anchor'] == anchor and h['interval'] == interval), [])
            nopolicy_hist = next((h['history'] for h in histories_nopolicy if h['anchor'] == anchor and h['interval'] == interval), [])

            indices_p, waits_p = get_class_waiting(policy_hist)
            indices_np, waits_np = get_class_waiting(nopolicy_hist)

            ax = axs[j]
            for outcome in ['served', 'jockeyed', 'reneged']:
                # Policy
                xs_p = np.array(indices_p[outcome])
                ys_p = np.array(waits_p[outcome])
                if len(xs_p) >= window:
                    ys_p_smooth = np.convolve(ys_p, np.ones(window)/window, mode='valid')
                    xs_p_smooth = xs_p[window-1:]
                    ax.plot(xs_p_smooth, ys_p_smooth,
                            label=f"Policy: {outcome.capitalize()}",
                            color=color_map[outcome],
                            linestyle=style_map['Policy'])
                elif len(xs_p) > 0:
                    ax.plot(xs_p, ys_p,
                            label=f"Policy: {outcome.capitalize()}",
                            color=color_map[outcome],
                            linestyle=style_map['Policy'],
                            marker='o')

                # No Policy
                xs_np = np.array(indices_np[outcome])
                ys_np = np.array(waits_np[outcome])
                if len(xs_np) >= window:
                    ys_np_smooth = np.convolve(ys_np, np.ones(window)/window, mode='valid')
                    xs_np_smooth = xs_np[window-1:]
                    ax.plot(xs_np_smooth, ys_np_smooth,
                            label=f"No Policy: {outcome.capitalize()}",
                            color=color_map[outcome],
                            linestyle=style_map['No Policy'])
                elif len(xs_np) > 0:
                    ax.plot(xs_np, ys_np,
                            label=f"No Policy: {outcome.capitalize()}",
                            color=color_map[outcome],
                            linestyle=style_map['No Policy'],
                            marker='x')

            ax.set_xlabel('Customer Index (Simulation Step)')
            if j == 0:
                ax.set_ylabel('Avg Waiting Time (Moving Avg)')
            ax.set_title(f"Interval: {interval}")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
       


def plot_avg_wait_by_queue_length_grouped_by_anchor(
    histories_policy, histories_nopolicy, window=1, title="Avg Waiting Time vs Queue Length (Grouped by Anchor)"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import itertools

    outcomes = ['jockeyed', 'reneged']
    # Gather all anchors and intervals present
    anchors_policy = sorted(set(h.get('anchor', None) for h in histories_policy))
    anchors_nopolicy = sorted(set(h.get('anchor', None) for h in histories_nopolicy))
    all_anchors = [a for a in anchors_policy + anchors_nopolicy if a is not None]
    all_anchors = sorted(set(all_anchors))
    intervals_policy = set(h.get('interval', None) for h in histories_policy)
    intervals_nopolicy = set(h.get('interval', None) for h in histories_nopolicy)
    all_intervals = sorted([i for i in intervals_policy | intervals_nopolicy if i is not None])

    nrows = len(all_intervals)
    ncols = len(all_anchors) * len(outcomes)

    def extract_by_anchor_interval(histories, anchor, interval, outcome):
        xs, ys = [], []
        for h in histories:
            if h.get('anchor') != anchor or h.get('interval') != interval:
                continue
            for obs in h['history']:
                if isinstance(obs, dict):
                    waited = obs.get('Waited', None)
                    cid = obs.get('customerid', '')
                    queue_length = obs.get('queue_length', None)
                    if waited is not None and queue_length is not None:
                        if (outcome == 'reneged' and '_reneged' in cid) or \
                           (outcome == 'jockeyed' and '_jockeyed' in cid):
                            xs.append(queue_length)
                            ys.append(waited)
                elif isinstance(obs, list):
                    for o in obs:
                        if not isinstance(o, dict):
                            continue
                        waited = o.get('Waited', None)
                        cid = o.get('customerid', '')
                        queue_length = o.get('queue_length', None)
                        if waited is not None and queue_length is not None:
                            if (outcome == 'reneged' and '_reneged' in cid) or \
                               (outcome == 'jockeyed' and '_jockeyed' in cid):
                                xs.append(queue_length)
                                ys.append(waited)
        return np.array(xs), np.array(ys)

    def binned_average(xs, ys, window=1):
        from collections import defaultdict
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[int(x)].append(y)
        bin_xs = sorted(bins.keys())
        bin_ys = [np.mean(bins[x]) for x in bin_xs]
        if window > 1 and len(bin_ys) >= window:
            bin_ys = np.convolve(bin_ys, np.ones(window)/window, mode='same')
        return np.array(bin_xs), np.array(bin_ys)

    color_map = {
        'Policy': 'tab:blue',
        'No Policy': 'tab:orange'
    }

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows), squeeze=False, sharey='row')

    for row, interval in enumerate(all_intervals):
        for col, (anchor, outcome) in enumerate(itertools.product(all_anchors, outcomes)):
            ax = axs[row, col]
            # Policy
            xs_policy, ys_policy = extract_by_anchor_interval(histories_policy, anchor, interval, outcome)
            bin_xs_p, bin_ys_p = binned_average(xs_policy, ys_policy, window)
            if len(bin_xs_p) > 0:
                ax.plot(bin_xs_p, bin_ys_p, '-', label='Policy', color=color_map['Policy'])
            # No Policy
            xs_np, ys_np = extract_by_anchor_interval(histories_nopolicy, anchor, interval, outcome)
            bin_xs_np, bin_ys_np = binned_average(xs_np, ys_np, window)
            if len(bin_xs_np) > 0:
                ax.plot(bin_xs_np, bin_ys_np, '--', label='No Policy', color=color_map['No Policy'])
            # Labels
            if row == 0:
                ax.set_title(f"{anchor}\n{outcome.title()}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Interval: {interval}\nAvg Waiting Time", fontsize=11)
            ax.set_xlabel('Queue Length')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=17)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def full_waiting_time_plot(histories_policy, histories_nopolicy, window=1, title="Avg Waiting Time vs Queue Length"):
    outcomes = ['jockeyed', 'reneged']

    def aggregate_by_anchor_and_interval(histories, outcome):
        agg = defaultdict(lambda: defaultdict(lambda: ([], [])))
        for h in histories:
            anchor = h['anchor']
            interval = h['interval']
            for obs in h['history']:
                waited = obs.get('Waited')
                cid = obs.get('customerid', '')
                queue_length = obs.get('queue_length')
                if waited is not None and queue_length is not None:
                    if (outcome == 'reneged' and '_reneged' in cid) or \
                       (outcome == 'jockeyed' and '_jockeyed' in cid):
                        agg[anchor][interval][0].append(queue_length)
                        agg[anchor][interval][1].append(waited)
        return agg

    def binned_average(xs, ys, window=1):
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[int(x)].append(y)
        bin_xs = sorted(bins.keys())
        bin_ys = [np.mean(bins[x]) for x in bin_xs]
        if window > 1 and len(bin_ys) >= window:
            bin_ys = np.convolve(bin_ys, np.ones(window)/window, mode='same')
        return np.array(bin_xs), np.array(bin_ys)

    anchors = sorted(set(h['anchor'] for h in histories_policy + histories_nopolicy))
    intervals = sorted(set(h['interval'] for h in histories_policy + histories_nopolicy))

    agg_policy = {o: aggregate_by_anchor_and_interval(histories_policy, o) for o in outcomes}
    agg_nopolicy = {o: aggregate_by_anchor_and_interval(histories_nopolicy, o) for o in outcomes}

    nrows = len(intervals)
    ncols = len(anchors)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    colors = {'jockeyed': 'tab:blue', 'reneged': 'tab:red'}
    styles = {'Policy': '-', 'No Policy': '--'}

    for i, interval in enumerate(intervals):
        for j, anchor in enumerate(anchors):
            ax = axs[i, j]
            for outcome in outcomes:
                # Policy
                xs_p, ys_p = agg_policy[outcome][anchor][interval]
                bin_xs_p, bin_ys_p = binned_average(xs_p, ys_p, window)
                if len(bin_xs_p) > 0:
                    ax.plot(bin_xs_p, bin_ys_p, styles['Policy'], label=f'Policy - {outcome}', color=colors[outcome])
                # No Policy
                xs_np, ys_np = agg_nopolicy[outcome][anchor][interval]
                bin_xs_np, bin_ys_np = binned_average(xs_np, ys_np, window)
                if len(bin_xs_np) > 0:
                    ax.plot(bin_xs_np, bin_ys_np, styles['No Policy'], label=f'No Policy - {outcome}', color=colors[outcome])
            if i == 0:
                ax.set_title(f"Anchor: {anchor}", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Interval: {interval}\nAvg Waiting Time", fontsize=12)
            ax.set_xlabel("Queue Length")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_waiting_time_by_anchor_and_interval(     histories_policy, histories_nopolicy, window=1, title="Avg Waiting Time vs Queue Length"):
    outcomes = ['jockeyed', 'reneged']

    def aggregate_by_anchor_and_interval(histories, outcome):
        agg = defaultdict(lambda: defaultdict(lambda: ([], [])))
        for h in histories:
            anchor = h['anchor']
            interval = h['interval']
            for obs in h['history']:
                waited = obs.get('Waited')
                cid = obs.get('customerid', '')
                queue_length = obs.get('queue_length')
                if waited is not None and queue_length is not None:
                    if (outcome == 'reneged' and '_reneged' in cid) or \
                       (outcome == 'jockeyed' and '_jockeyed' in cid):
                        agg[anchor][interval][0].append(queue_length)
                        agg[anchor][interval][1].append(waited)
        return agg

    def binned_average(xs, ys, window=1):
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[int(x)].append(y)
        bin_xs = sorted(bins.keys())
        bin_ys = [np.mean(bins[x]) for x in bin_xs]
        if window > 1 and len(bin_ys) >= window:
            bin_ys = np.convolve(bin_ys, np.ones(window)/window, mode='same')
        return np.array(bin_xs), np.array(bin_ys)

    anchors = sorted(set(h['anchor'] for h in histories_policy + histories_nopolicy))
    intervals = sorted(set(h['interval'] for h in histories_policy + histories_nopolicy))

    agg_policy = {o: aggregate_by_anchor_and_interval(histories_policy, o) for o in outcomes}
    agg_nopolicy = {o: aggregate_by_anchor_and_interval(histories_nopolicy, o) for o in outcomes}

    nrows = len(intervals)
    ncols = len(anchors)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    colors = {'jockeyed': 'tab:blue', 'reneged': 'tab:red'}
    styles = {'Policy': '-', 'No Policy': '--'}

    for i, interval in enumerate(intervals):
        for j, anchor in enumerate(anchors):
            ax = axs[i, j]
            for outcome in outcomes:
                # Policy
                xs_p, ys_p = agg_policy[outcome][anchor][interval]
                bin_xs_p, bin_ys_p = binned_average(xs_p, ys_p, window)
                if len(bin_xs_p) > 0:
                    ax.plot(bin_xs_p, bin_ys_p, styles['Policy'], label=f'Policy - {outcome}', color=colors[outcome])
                # No Policy
                xs_np, ys_np = agg_nopolicy[outcome][anchor][interval]
                bin_xs_np, bin_ys_np = binned_average(xs_np, ys_np, window)
                if len(bin_xs_np) > 0:
                    ax.plot(bin_xs_np, bin_ys_np, styles['No Policy'], label=f'No Policy - {outcome}', color=colors[outcome])
            if i == 0:
                ax.set_title(f"Anchor: {anchor}", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Interval: {interval}\nAvg Waiting Time", fontsize=12)
            ax.set_xlabel("Queue Length")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
       
######### Globals ########
request_log = []


def main():
	
    utility_basic = 1.0
    discount_coef = 0.1
    requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=True)
    duration = 5 # 0   
    
    # Set intervals for dispatching queue states
    intervals = [3, 5, 7, 9]
    wasted_by_interval = []
    per_outcome_by_interval = []
    
    jockey_anchors = ["steady_state_distribution", "inter_change_time"]

    all_results = {anchor: [] for anchor in jockey_anchors}       
    
    # Iterate through each interval in the intervals array
    all_reneging_rates = []
    all_jockeying_rates = []             
    
    ############################## last attempt ######################################
    
    def run_simulations_equal_samples(utility_basic, discount_coef, intervals, duration, jockey_anchors, policy_enabled):
        results = {
            interval: {
                "reneging_rates": {anchor: None for anchor in jockey_anchors},
                "jockeying_rates": {anchor: None for anchor in jockey_anchors}
            }
            for interval in intervals
        }                
        
        all_histories = []
        for interval in intervals:
            for anchor in jockey_anchors:
                # Optionally use a fixed seed for reproducibility:
                seed = hash((interval, anchor, policy_enabled)) % (2**32)
                requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=policy_enabled, seed=seed)
                requestObj.jockey_anchor = anchor
                requestObj.run(duration, interval)
                
                results[interval]["reneging_rates"][anchor]["server_1"] = requestObj.interval_stats["reneging_rate"]["server_1"]
                results[interval]["jockeying_rates"][anchor]["server_1"] = requestObj.interval_stats["jockeying_rate"]["server_1"]
                results[interval]["reneging_rates"][anchor]["server_2"] = requestObj.interval_stats["reneging_rate"]["server_2"]
                results[interval]["jockeying_rates"][anchor]["server_2"] = requestObj.interval_stats["jockeying_rate"]["server_2"]
                
                # Example: Calculate rates for this anchor/interval (adjust as needed for your code)
                reneged = sum(1 for obs in requestObj.history if '_reneged' in getattr(obs, 'customerid', ''))
                jockeyed = sum(1 for obs in requestObj.history if '_jockeyed' in getattr(obs, 'customerid', ''))
                total = len(requestObj.history)
                results[interval]["reneging_rates"][anchor] = reneged / total if total else 0
                results[interval]["jockeying_rates"][anchor] = jockeyed / total if total else 0
                # Store full history
                all_histories.append({
                    "policy_enabled": policy_enabled,
                    "interval": interval,
                    "anchor": anchor,
                    "history": list(requestObj.history)
                })
                
        return results, all_histories 
        
        
    def run_simulations_with_results(  utility_basic,    discount_coef,    intervals,    duration,    jockey_anchors,    policy_enabled,    RequestQueue ):
        """
        Returns:
            results: Nested dict [interval][stat_name][anchor] = rate
            all_histories: List of dicts (histories for plotting)
        """
        results = {
            interval: {
                "reneging_rates": {anchor: None for anchor in jockey_anchors},
                "jockeying_rates": {anchor: None for anchor in jockey_anchors}
            }
            for interval in intervals
        }
        all_histories = []
        for interval in intervals:
            for anchor in jockey_anchors:
                # Optional: Use a fixed seed for reproducibility
                seed = hash((interval, anchor, policy_enabled)) % (2**32)
                rq = RequestQueue(
                    utility_basic,
                    discount_coef,
                    policy_enabled=policy_enabled,
                    seed=42  # Only include if your class supports it
                )
                rq.jockey_anchor = anchor
                rq.run(duration, interval)
                
                results[interval]["reneging_rates"][anchor]["server_1"] = requestObj.interval_stats["reneging_rate"]["server_1"]
                results[interval]["jockeying_rates"][anchor]["server_1"] = requestObj.interval_stats["jockeying_rate"]["server_1"]
                results[interval]["reneging_rates"][anchor]["server_2"] = requestObj.interval_stats["reneging_rate"]["server_2"]
                results[interval]["jockeying_rates"][anchor]["server_2"] = requestObj.interval_stats["jockeying_rate"]["server_2"]
                
                # Convert history to dicts if not already
                event_list = []
                reneged = 0
                jockeyed = 0
                total = 0
                for obs in rq.history:
                    if isinstance(obs, dict):
                        event_list.append(obs)
                        cid = obs.get("customerid", "")
                    else:
                        d = {
                            "Waited": getattr(obs, "Waited", None),
                            "queue_length": getattr(obs, "queue_length", None),
                            "customerid": getattr(obs, "customerid", "")
                        }
                        event_list.append(d)
                        cid = d["customerid"]
                    total += 1
                    if "_reneged" in cid:
                        reneged += 1
                    if "_jockeyed" in cid:
                        jockeyed += 1
                results[interval]["reneging_rates"][anchor] = reneged / total if total else 0
                results[interval]["jockeying_rates"][anchor] = jockeyed / total if total else 0
                all_histories.append({
                    "policy_enabled": policy_enabled,
                    "interval": interval,
                    "anchor": anchor,
                    "history": event_list
                })
        return results, all_histories
        
        # Run for both policy modes
    results_no_policy, histories_nopolicy = run_simulations_with_results(    utility_basic,    discount_coef,    intervals,    duration,    jockey_anchors,    False,    RequestQueue)
    results_policy, histories_policy = run_simulations_with_results(    utility_basic,    discount_coef,    intervals,    duration,    jockey_anchors,    True,    RequestQueue)

    #histories_nopolicy = run_simulations(    utility_basic,    discount_coef,    intervals,    duration,    jockey_anchors,    policy_enabled=False,    RequestQueueClass=RequestQueue)

           
    # With policy-driven service rates
    #####results_policy , histories_policy = run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, jockey_anchors=jockey_anchors, policy_enabled=True)    

    # With static/non-policy-driven service rates
    #####results_no_policy, histories_nopolicy = run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, jockey_anchors=jockey_anchors, policy_enabled=False)   
   
    ##################################### End of last attempt ##################################               
    
    waiting_times, outcomes, time_stamps = extract_waiting_times_and_outcomes(requestObj)
    
   
    '''
     come back to the functions below
    '''
    # plot_boxplot_waiting_times_by_outcome(waiting_times, outcomes)
    #plot_avg_waiting_time_over_time(waiting_times, time_stamps, window=10)
    
    # service rates and jockeying/reneging rates not smooth
    requestObj.plot_rates_by_intervals()
    
    requestObj.plot_reneged_waiting_times_by_interval()
    requestObj.plot_jockeyed_waiting_times_by_interval()
    
    plot_six_panels(results_no_policy, intervals, jockey_anchors)
    
    plot_six_panels(results_policy, intervals, jockey_anchors)
    
    plot_waiting_time_by_anchor_and_interval(histories_policy, histories_nopolicy, window=2)
    
    ###### plot_avg_wait_by_queue_length_grouped_by_anchor(histories_policy, histories_nopolicy, window=2)
    ###### plot_avg_wait_by_queue_length(histories_policy, histories_nopolicy, window=2)
    
    ###### plot_avg_wait_by_queue_length_grouped(histories_policy, histories_nopolicy, window=2)
    
    # Choose a specific simulation (e.g., interval=3, anchor='steady_state_distribution')
    # histories_policy and histories_nopolicy are lists of dicts, each with 'interval', 'anchor', 'history'
    
    ###### plot_all_avg_waiting_time_by_anchor_interval(histories_policy, histories_nopolicy, window=20)        
    
    # --- New: Plot policy and model histories ---
    print("\n[Diagnostics] Plotting policy evolution...")
    plot_policy_history(requestObj.policy1.get_policy_history())
    plot_policy_history(requestObj.policy2.get_policy_history())

    print("\n[Diagnostics] Plotting predictive model fit history...")
    plot_predictive_model_history(requestObj.predictive_model.get_fit_history())
      
	 
if __name__ == "__main__":
    main()
    
# plot rates for each anchor, interval, against number of requests 
#plot_rates_vs_requests(results, intervals=[3,6,9], anchors=["steady_state_distribution", "inter_change_time"], metric="jockeying_rates")
#plot_rates_vs_requests(results, intervals=[3,6,9], anchors=["steady_state_distribution", "inter_change_time"], metric="reneging_rates")

######## block that works starts here  ########
    '''
    for anchor in jockey_anchors:
        for interval in intervals:
            print(f"Running simulation for anchor: {anchor}, interval: {interval}s")
            requestObj = RequestQueue(utility_basic, discount_coef)
            requestObj.jockey_anchor = anchor   # <-- Add this attribute

            requestObj.run(duration, interval)
            
            results[interval]["reneging_rates"][anchor] = {
                "server_1": requestObj.dispatch_data[interval]["server_1"]["reneging_rate"],
                "server_2": requestObj.dispatch_data[interval]["server_2"]["reneging_rate"]
            }
            results[interval]["jockeying_rates"][anchor] = {
                "server_1": requestObj.dispatch_data[interval]["server_1"]["jockeying_rate"],
                "server_2": requestObj.dispatch_data[interval]["server_2"]["jockeying_rate"]
            }         

            # Clear request log for next run
            request_log.clear()
    '''
    ######## block that works ends here  ########
    
     # After all interval runs, plot the aggregated results with smoothing
    #fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    #window_length = 9  # Must be odd and <= length of data
    #polyorder = 2

    # Plot reneging rates
    '''
    for i, interval in enumerate(intervals):
        for server, style in zip(["server_1", "server_2"], ["-", "--"]):
            y = np.array(all_reneging_rates[i][server])
            x = np.arange(len(y))
            if len(y) >= window_length:
                y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
                x_smooth = x
            elif len(y) >= 3:
                # Use smaller window if needed
                wl = len(y) if len(y)%2==1 else len(y)-1
                y_smooth = savgol_filter(y, window_length=wl, polyorder=2)
                x_smooth = x
            else:
                y_smooth = y
                x_smooth = x
            axs[0].plot(x_smooth, y_smooth, label=f'{server.replace("_", " ").title()} - Interval {interval}s', linestyle=style)
            
    ######## new functionality ########
    
    for i, interval in enumerate(intervals):
        for server, style in zip(["server_1", "server_2"], ["-", "--"]):
            y = np.array(all_reneging_rates[i][server])
            x = np.arange(len(y))
            if len(y) >= window_length:
                y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
            elif len(y) >= 3:
                wl = len(y) if len(y)%2==1 else len(y)-1
                y_smooth = savgol_filter(y, window_length=wl, polyorder=2)
            else:
                y_smooth = y
            axs[0].plot(x, y_smooth, label=f'{server.replace("_", " ").title()} - Interval {interval}s', linestyle=style)
            # Optional: show end marker
            axs[0].scatter([x[-1]], [y_smooth[-1]], color='black')

    axs[0].set_title('Reneging Rates Across Intervals (Smoothed)')
    axs[0].set_ylabel('Reneging Rate')
    axs[0].legend()
    
    '''
    
    # Plot jockeying rates
    '''
    for i, interval in enumerate(intervals):
        for server, style in zip(["server_1", "server_2"], ["-", "--"]):
            y = np.array(all_jockeying_rates[i][server])
            x = np.arange(len(y))
            if len(y) >= window_length:
                y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
                x_smooth = x
            elif len(y) >= 3:
                wl = len(y) if len(y)%2==1 else len(y)-1
                y_smooth = savgol_filter(y, window_length=wl, polyorder=2)
                x_smooth = x
            else:
                y_smooth = y
                x_smooth = x
            axs[1].plot(x_smooth, y_smooth, label=f'{server.replace("_", " ").title()} - Interval {interval}s', linestyle=style)
    
    ######## new functionality ########
    
    for i, interval in enumerate(intervals):
        for server, style in zip(["server_1", "server_2"], ["-", "--"]):
            y = np.array(all_jockeying_rates[i][server])
            x = np.arange(len(y))
            if len(y) >= window_length:
                y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
            elif len(y) >= 3:
                wl = len(y) if len(y)%2==1 else len(y)-1
                y_smooth = savgol_filter(y, window_length=wl, polyorder=2)
            else:
                y_smooth = y
            axs[1].plot(x, y_smooth, label=f'{server.replace("_", " ").title()} - Interval {interval}s', linestyle=style)
            # Optional: show end marker
            axs[1].scatter([x[-1]], [y_smooth[-1]], color='black')
    axs[1].set_title('Jockeying Rates Across Intervals (Smoothed)')
    axs[1].set_xlabel('Number of Requests')
    axs[1].set_ylabel('Jockeying Rate')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    '''
        
    # plot_waiting_time_cdf(waiting_times)
