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
        if len(self.X) > 15:
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
        self.arrival_rates = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
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


    def set_obs (self, queue_id,  serv_rate, intensity, time_in_serv, activity, rewarded, curr_pose, req): # reneged, jockeyed,
        		
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
                }
              

    def get_obs (self):
        
        return dict(self.obs)
        
        
    def set_renege_obs(self, curr_pose, queue_intensity, reneged,time_local_service, time_to_service_end, reward, queueid, activity):		

        self.curr_obs_renege.append(
            {   
                "queue": queueid,
                "at_pose": curr_pose,
                "reneged": reneged,
                #"this_busy": queue_intensity,
                "expected_local_service":time_local_service,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity
            }
        )
        
        
    def get_renege_obs(self, queueid, queue): # , intensity, pose): # get_curr_obs_renege
		
        renegs = sum(1 for req in queue if '_reneged' in req.customerid)
        			    
        # print("\n Num Reneged in ", queueid, "is =", renegs)
	    
        return renegs # self.curr_obs_renege 
  
        
    def set_jockey_obs(self, curr_pose, queue_intensity, jockeyed, time_alt_queue, time_to_service_end, reward, queueid, activity):
        
        self.curr_obs_jockey.append(
            {
                "queue": queueid,
                "at_pose": curr_pose,
                #"this_busy": queue_intensity,
                "jockeyed": jockeyed,
                "expected_local_service":time_alt_queue,
                "time_service_took": time_to_service_end,
                "reward": reward,
                "action":activity   			
            }
        )
        
    
    def get_jockey_obs(self, queueid, intensity, pose):
		
        return self. curr_obs_jockey
           
    
class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, utility_basic, discount_coef, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon, exp_time_service_end=0.0,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, 
                 batchid=np.int16, policy_enabled=True):
                 
        
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
        self.srvrates_1 = 0
        self.srvrates_2 = 0                           
        
        BROADCAST_INTERVAL = 5        
        
        self.policy_enabled = policy_enabled
        self.predictive_model = PredictiveModel()
        self.policy = ServerPolicy(self.predictive_model, min_rate=1.0, max_rate=15.0)
        
        # Schedule the dispatch function to run every minute (or any preferred interval)
        # schedule.every(1).minutes.do(self.dispatch_queue_state, queue=queue_1, queue_name="Server1")
        # schedule.every(1).minutes.do(dispatch_queue_state, queue=queue_2, queue_name="Server2")
        
        # Start the scheduler     
        #scheduler_thread = threading.Thread(target=self.run_scheduler)
        #scheduler_thread.start()
        
        return
    
        
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

        return [self.srvrates_1, self.srvrates_2] # srvrate1, srvrate2]


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
                
                #print("\n Current State of Server",queue_id, " => ", curr_queue_state)
                self.dispatch_queue_state( curr_queue, alt_queue_id, alt_queue, interval) #, curr_queue_state)
                       
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
        service_rate = self.srvrates_1 if queue_id == "1" else self.srvrates_2
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
            steady_state_distribution["max"],
            steady_state_distribution["min"],
            steady_state_distribution["mean"],
        ]

    # After each dispatch or user action, record observation and update model/policy
    def record_user_reaction(self, queue_id, action_label):
        """
        Call this after observing a user action (wait, renege, jockey).
        """
        features = self.extract_features(queue_id)
        self.predictive_model.record_observation(features, action_label)
        self.predictive_model.fit()
        # Optionally: immediately update policy based on new prediction
        self.policy.update_policy(features)


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
                    self.srvrates_1 = self.policy.update_policy(features_srv1)
                    self.srvrates_2 = self.policy.update_policy(features_srv2)
                    
                # else: leave self.srvrates_1 and self.srvrates_2 as initialized/randomized above
               
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

  
    def dispatch_queue_state(self, curr_queue, alt_queue_id, alt_queue, interval): #, curr_queue_state): # curr_queue_id
    
        if interval not in self.dispatch_data:
            self.dispatch_data[interval] = {
                "server_1": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []},
                "server_2": {"num_requests": [], "jockeying_rate": [], "reneging_rate": [], "service_rate": [], "intervals": []}
            }
		
        rate_srv1, rate_srv2 = self.get_server_rates()
		
        if "1" in alt_queue_id:
            curr_queue_id = "2"
            serv_rate = rate_srv1
        else:
            curr_queue_id = "1"
            serv_rate = rate_srv2
        
        curr_queue_state = self.get_queue_state(alt_queue_id)
                
        # print("\n\n From get_queue_state:", curr_queue_state)
                 
        #curr_queue_state["jockeying_rate"] = jockeying_rate
        #curr_queue_state["reneging_rate"] = reneging_rate 
         
        # Compute reneging rate and jockeying rate
        reneging_rate = self.compute_reneging_rate(curr_queue)
        jockeying_rate = self.compute_jockeying_rate(curr_queue)
        num_requests = curr_queue_state['total_customers'] # len(curr_queue)        

        for client in range(len(curr_queue)):
            req = curr_queue[client]  # self.dict_queues_obj[curr_queue_id][client]
            print(f"Dispatching state of server {alt_queue_id} to client {req.customerid} : {curr_queue_state}.")
            
            if "1" in alt_queue_id: # == "1":
                self.makeJockeyingDecision(req, alt_queue_id, curr_queue_id, req.customerid, serv_rate)
                self.makeRenegingDecision(req, alt_queue_id)
                alt_queue_id = str(alt_queue_id) # "Server_"+
            else:
                self.makeJockeyingDecision(req, alt_queue_id, curr_queue_id, req.customerid, serv_rate)
                self.makeRenegingDecision(req, alt_queue_id)
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

            curr_queue_state = self.get_queue_state(queue_id)         
            reneging_rate, jockeying_rate = self.dispatch_queue_state( curr_queue, queue_id, alt_queue, curr_queue_state)       
            # Record the statistics
            serv_rate = self.dict_servers_info[queue_id]
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
		
        #rate_srv1, rate_srv2 = self.get_server_rates()
        # print("\n >>> ", rate_srv1, " === ", rate_srv2, " === ", self.objQueues.get_arrivals_rates())                            
	
        if "1" in queueid:		
            #queue_intensity = self.objQueues.get_arrivals_rates()/ rate_srv1    
            customers_in_queue = self.dict_queues_obj["1"] 
            curr_queue = self.dict_queues_obj[queueid]  
            reneging_rate = self.compute_reneging_rate(curr_queue)
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
            # Instantiate MarkovQueueModel
            #print("\n --> ", self.arr_rate, self.srvrates_1)
            markov_model = MarkovQueueModel(self.arr_rate, self.srvrates_1, max_states=len(customers_in_queue)) # 1000)
            long_run_change_rate = markov_model.long_run_change_rate()
            sample_interchange_time = markov_model.sample_interchange_time() # _steady_state_distribution()
            steady_state_distribution = markov_model._steady_state_distribution()
            steady_state_distribution_list = steady_state_distribution.tolist()  # Convert to list                    
      
        else:
			 
            #queue_intensity = self.objQueues.get_arrivals_rates()/ rate_srv2            
            customers_in_queue = self.dict_queues_obj["2"]
            curr_queue = self.dict_queues_obj[queueid]
            reneging_rate = self.compute_reneging_rate(curr_queue)
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
            # Instantiate MarkovQueueModel
            #print("\n **> ", self.arr_rate, self.srvrates_2)
            markov_model = MarkovQueueModel(self.arr_rate, self.srvrates_2, max_states=len(customers_in_queue)) # 1000)
            long_run_change_rate = markov_model.long_run_change_rate()
            steady_state_distribution = markov_model._steady_state_distribution()
            steady_state_distribution_list = steady_state_distribution.tolist()  # Convert to list
            sample_interchange_time = markov_model.sample_interchange_time()          
		       
        curr_queue_state = {
            "total_customers": len(customers_in_queue),
            #"intensity": queue_intensity,
            "capacity": self.capacity,              
            "long_avg_serv_time": self.get_long_run_avg_service_time(queueid),
            "sample_interchange_time": sample_interchange_time,
            "long_run_change_rate": long_run_change_rate,
            "steady_state_distribution": {
                "max": max(steady_state_distribution_list),
                "min": min(steady_state_distribution_list),
                "mean": sum(steady_state_distribution_list) / len(steady_state_distribution_list),
            },
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
            serv_rate = self.dict_servers_info["1"] # Server1
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
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, req.time_exit-req.time_entrance, reward, len_queue_1, 2, req)   # req.exp_time_service_end,                                    
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
            serv_rate = self.dict_servers_info["2"] # Server2
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
            
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, req.time_exit-req.time_entrance, reward, len_queue_2, 2, req)    # req.exp_time_service_end,                                  
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


    def reqRenege(self, req, queueid, curr_pose, serv_rate, queue_intensity, time_local_service, customerid, time_to_service_end, decision, curr_queue):
        
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
        
                self.objObserv.set_renege_obs(curr_pose, queue_intensity, decision,time_local_service, time_to_service_end, reward, queueid, "reneged")
        
                self.curr_obs_renege.append(self.objObserv.get_renege_obs(queueid, self.queue)) #queueid, queue_intensity, curr_pose))        
        
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
            #print("\n ********* ", request_log[0])
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
            self.curr_obs_jockey.append(self.objObserv.get_jockey_obs(curr_queue_id, queue_intensity, curr_pose))                                      
            self.curr_req = req        
            self.objQueues.update_queue_status(curr_queue_id)# long_avg_serv_time
        
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
        
        if "Server1" in curr_queue_id:
            self.queue = self.dict_queues_obj["1"]  # Server1           
        else:
            self.queue = self.dict_queues_obj["2"]
            
        if curr_queue_id == "1":
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
                        
            queue_states_compared = self.compare_queues(alt_steady_state, curr_steady_state, K=1) 
            
            anchor = getattr(self, "jockey_anchor", "steady_state_distribution")
            
            if anchor == "steady_state_distribution":
                # Use steady-state distribution comparison (as currently implemented)
                queue_states_compared = self.compare_queues(alt_steady_state, curr_steady_state, K=1)
                if queue_states_compared['first_order_dominance']:
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
        self.lambda_ = arrival_rate
        self.mu = service_rate
        #print("\n IN MORSKOV MODEL CLASS: ==> ", arrival_rate, service_rate)
        
        if abs(service_rate) < 1e-8:
            self.rho = 0.0  # fallback value; choose np.inf or another value if more appropriate for your model
        else:
            self.rho = arrival_rate / service_rate
            
        self.max_states = max_states
        self.pi = self._steady_state_distribution()
               
    
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
    
    def simulate(self, T):
        """
        Simulate μ(t) and service times over horizon [0, T].
        
        Returns:
        - events: list of tuples (time, event_type, state)
          event_type = 'jump' or 'service'
        """
        events = []
        next_jump = self.step()
        next_service = self.sample_service_time()
        
        while True:
            # choose which event happens first
            if next_jump + self.current_time <= next_service + self.current_time and self.current_time + next_jump <= T:
                # jump occurs first
                self.current_time += next_jump
                events.append((self.current_time, 'jump', self.current_state))
                next_jump = self.step()
                # schedule next_service relative to new state
                next_service = self.sample_service_time()
            elif self.current_time + next_service <= T:
                # service occurs first
                self.current_time += next_service
                events.append((self.current_time, 'service', self.current_state))
                # schedule next_service from current state
                next_service = self.sample_service_time()
                # adjust next_jump as remaining time
                # no change needed: jump_time remains
            else:
                break
        
        return events



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
    

def plot_avg_waiting_time_over_time(waiting_times, time_stamps, window=10, title="Average Waiting Time Over Time"):
    """
    Plots a moving average of waiting time over the simulation time.
    """
    # Remove Nones if present
    mask = [t is not None for t in time_stamps]
    waiting_times = np.array(waiting_times)[mask]
    time_stamps = np.array(time_stamps)[mask]

    idx_sorted = np.argsort(time_stamps)
    waiting_times = waiting_times[idx_sorted]
    time_stamps = time_stamps[idx_sorted]

    # Compute moving average
    if len(waiting_times) >= window:
        avg_wait = np.convolve(waiting_times, np.ones(window)/window, mode='valid')
        time_avg = time_stamps[window-1:]
    else:
        avg_wait = waiting_times
        time_avg = time_stamps
    plt.figure(figsize=(8,4))
    plt.plot(time_avg, avg_wait, marker='.')
    plt.xlabel("Simulation Time")
    plt.ylabel("Average Waiting Time")
    plt.title(title)
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


def plot_six_panels(results, intervals, jockey_anchors):
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=False)
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
                y = np.array(results[interval]["jockeying_rates"][anchor][server])
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
      
       
######### Globals ########
request_log = []


def main():
	
    utility_basic = 1.0
    discount_coef = 0.1
    requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=True)
    duration = 100 
    
    # Set intervals for dispatching queue states
    intervals = [3, 6, 9]
    wasted_by_interval = []
    per_outcome_by_interval = []
    
    jockey_anchors = ["steady_state_distribution", "inter_change_time"]

    all_results = {anchor: [] for anchor in jockey_anchors}
    
    
    
    # Iterate through each interval in the intervals array
    all_reneging_rates = []
    all_jockeying_rates = []             
    
    def run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, policy_enabled, jockey_anchors):
		
        results = {
            interval: {
                "reneging_rates": {anchor: {} for anchor in jockey_anchors},
                "jockeying_rates": {anchor: {} for anchor in jockey_anchors}
            }
            for interval in intervals
        }
        for interval in intervals:
            for anchor in jockey_anchors:
                requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=policy_enabled)
                requestObj.jockey_anchor = anchor  # Make sure this attribute is respected in RequestQueue
                requestObj.run(duration, interval)
                results[interval]["reneging_rates"][anchor] = {
                    "server_1": list(requestObj.dispatch_data[interval]["server_1"]["reneging_rate"]),
                    "server_2": list(requestObj.dispatch_data[interval]["server_2"]["reneging_rate"]),
                }
                results[interval]["jockeying_rates"][anchor] = {
                    "server_1": list(requestObj.dispatch_data[interval]["server_1"]["jockeying_rate"]),
                    "server_2": list(requestObj.dispatch_data[interval]["server_2"]["jockeying_rate"]),
                }
                # Clear log if needed
                if hasattr(requestObj, "request_log"):
                    requestObj.request_log.clear()
                elif "request_log" in globals():
                    request_log.clear()
        return results
    
    # With policy-driven service rates
    results_policy = run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, policy_enabled=True,  jockey_anchors=jockey_anchors)

    # With static/non-policy-driven service rates
    results_no_policy = run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, policy_enabled=False,  jockey_anchors=jockey_anchors)          

    '''
    for interval in intervals:
        print(f"Running simulation with interval: {interval} seconds")
        #requestObj = RequestQueue(utility_basic, discount_coef)
        requestObj.run(duration, interval)
    
        # Collect reneging and jockeying rates for this interval
        reneging_rates = {
            "server_1": requestObj.dispatch_data[interval]["server_1"]["reneging_rate"],
            "server_2": requestObj.dispatch_data[interval]["server_2"]["reneging_rate"]
        }
        jockeying_rates = {
            "server_1": requestObj.dispatch_data[interval]["server_1"]["jockeying_rate"],
            "server_2": requestObj.dispatch_data[interval]["server_2"]["jockeying_rate"]
        }
    
        all_reneging_rates.append(reneging_rates)
        all_jockeying_rates.append(jockeying_rates)
                

        #total_wasted, per_outcome = requestObj.total_wasted_waiting_time(request_log)
        #wasted_by_interval.append(total_wasted)
        #per_outcome_by_interval.append(per_outcome)
        
        # Make sure request_log is cleared before each run!
        request_log.clear()
    '''
    
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
    
    waiting_times, outcomes, time_stamps = extract_waiting_times_and_outcomes(requestObj)
    
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
        
    plot_waiting_time_cdf(waiting_times)
    '''
     come back to the functions below
    '''
    #plot_boxplot_waiting_times_by_outcome(waiting_times, outcomes)
    #plot_avg_waiting_time_over_time(waiting_times, time_stamps, window=10)
    
    # service rates and jockeying/reneging rates not smooth
    requestObj.plot_rates_by_intervals()
    
    requestObj.plot_reneged_waiting_times_by_interval()
    requestObj.plot_jockeyed_waiting_times_by_interval()
    
    plot_six_panels(results_no_policy, intervals, jockey_anchors)
    
    plot_six_panels(results_policy, intervals, jockey_anchors)       
    
    # --- New: Plot policy and model histories ---
    print("\n[Diagnostics] Plotting policy evolution...")
    plot_policy_history(requestObj.policy.get_policy_history())

    print("\n[Diagnostics] Plotting predictive model fit history...")
    plot_predictive_model_history(requestObj.predictive_model.get_fit_history())
    
    # requestObj.plot_waiting_time_vs_rates_by_interval() #plot_waiting_time_vs_rates()

    # Run the dispatch intervals
    # requestObj.setup_dispatch_intervals(intervals)
    
    ### requestObj.run(duration, intervals[0])
    
    #requestObj.run_scheduler(duration=100)   
    
    # Plot the rates after the simulation
    ### requestObj.plot_rates()       # plot_rates_by_intervals() #       
	 
if __name__ == "__main__":
    main()
