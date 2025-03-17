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
import threading
from tqdm import tqdm
import MarkovStateMachine as msm
from timeit import default_timer as timer
###############################################################################

class Queues(object):
    def __init__(self):
        super().__init__()
        
        self.num_of_queues = 2
        self.dict_queues = {}
        self.dict_servers = {}
        self.arrival_rates = [3,5,7,9,11,13,15]
        rand_idx = random.randrange(len(self.arrival_rates))
        self.sampled_arr_rate = self.arrival_rates[rand_idx] 
        self.queueID = ""             
        
        self.dict_queues = self.generate_queues()
        #self.dict_servers = self.queue_setup_manager()

        self.capacity = 50 #np.inf
        
        
    def queue_setup_manager(self):
                
        # deltalambda controls the difference between the service rate of either queues    
        deltaLambda=random.randint(1, 2)
        
        serv_rate_one=self.sampled_arr_rate + deltaLambda 
        serv_rate_two=self.sampled_arr_rate - deltaLambda

        _serv_rate_one=serv_rate_one / 2
        _serv_rate_two=serv_rate_two / 2
        
        # print("\n .... ", self.dict_servers)
        self.dict_servers["1"] = _serv_rate_one # Server1
        self.dict_servers["2"] = _serv_rate_two # Server2
        
        # print("\n Current Arrival Rate:", self.sampled_arr_rate, "Server1:", _serv_rate_one, "Server2:", _serv_rate_two) 


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

    def __init__(self,time_entrance,pos_in_queue=0,utility_basic=0.0,discount_coef=0.0, outage_risk=0.1, # =timer()
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
        self.scale_local_delay=float(para_local_delay[2])
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
     
    '''
    def estimateMarkovWaitingTime(self):
        # print("   Estimating Markov waiting time...")
        queue_indices=np.arange(self.pos_in_queue-1)+1
        samples=1
        start_belief=np.matrix(np.zeros(self.markov_model.num_states).reshape(1,self.markov_model.num_states)[0],np.float64).T
        start_belief[self.markov_model.current_state]=1.0
        cdf=0        
        while cdf<=self.certainty:
            eff_srv=self.markov_model.integratedEffectiveFeature(samples,
            start_belief)
            cdf=1-sum((eff_srv**i*np.exp(-eff_srv)/np.math.factorial(i) for i in queue_indices))
            # print([eff_srv,cdf])
            samples+=1
        return (samples-1)*self.time_res

    '''
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
        
        
    def get_renege_obs(self, queueid, intensity, pose): # get_curr_obs_renege
		
        pass 
        #for obs in self.curr_obs_renege:
        #    if obs.get("ServerID") == queueid and obs.get("QueueSize") == pose and obs.get("Intensity") == intensity:
        #        print("\n RENENGE OBSERVED: ", obs)
        #        return obs
        #    else:
        #        continue
                                  
        #return self.curr_obs_renege
        
        
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
        
    
    def get_jockey_obs(self, queueid, intensity, pose):
		
        pass
        
        #return self.curr_obs_jockey


class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, utility_basic, discount_coef, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon, exp_time_service_end=0.0,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, batchid=0):
        
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
        self.history = [] # {}
        self.curr_obs_jockey = [] #{}
        self.curr_obs_renege = [] #{}

        self.arr_prev_times = np.array([])

        self.objQueues = Queues()
        # self.objRequest = Request()
        self.objObserv = Observations()

        self.dict_queues_obj = self.objQueues.get_dict_queues()
        self.dict_servers_info = self.objQueues.get_dict_servers()
        self.jockey_threshold = 1
        self.reward = 0.0
        self.curr_state = {} # ["Busy","Empty"]

        self.arr_rate = self.objQueues.get_arrivals_rates()

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
        self.dist_local_delay=dist_local_delay
        self.loc_local_delay=np.random.uniform(low=float(para_local_delay[0]),high=(para_local_delay[1]))
        self.scale_local_delay=float(para_local_delay[2])
        self.max_local_delay=self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay)
        self.max_cloud_delay= np.inf  # float(self.arr_rate/self.serv_rate) #
        
        # self.observations=np.array([]) 
        self.error_loss=1
        
        self.capacity = self.objQueues.get_queue_capacity()
        
        BROADCAST_INTERVAL = 5
        
        # Schedule the dispatch function to run every minute (or any preferred interval)
        # schedule.every(1).minutes.do(self.dispatch_queue_state, queue=queue_1, queue_name="Server1")
        # schedule.every(1).minutes.do(dispatch_queue_state, queue=queue_2, queue_name="Server2")
        
        # Start the scheduler     
        scheduler_thread = threading.Thread(target=self.run_scheduler)
        scheduler_thread.start()
        
        return
    
        
    #def set_customer_id(self):
		
    #    self.customerid = uuid.uuid4()
    
    
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
                #print("\n **** ", list(lst_srv2))
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
		

    def estimateMarkovWaitingTime(self, pos_in_queue): # Original
        # print("   Estimating Markov waiting time...")
        # queue_indices=np.arange(pos_in_queue-1)+1
        
        queue_indices=np.arange(pos_in_queue-1)+1
        samples=1
        start_belief=np.matrix(np.zeros(self.markov_model.num_states).reshape(1,self.markov_model.num_states)[0],np.float64).T
        start_belief[self.markov_model.current_state]=1.0
        cdf=0        
        while cdf<=self.certainty:
            eff_srv=self.markov_model.integratedEffectiveFeature(samples, start_belief, self.get_server_rates())
            # print("\n ***** ", type(eff_srv), eff_srv)
            cdf=1-sum((eff_srv**i*np.exp(-eff_srv)/np.math.factorial(i) for i in queue_indices))
            # print([eff_srv,cdf])
            samples+=1
        
        # return 
        self.avg_delay = (samples-1)*self.time_res
 
        return self.avg_delay


    def estimateMarkovWaitingTimeVer2(self, pos_in_queue, queue_intensity, time_entered):
        """Calculate the amount after a certain time with exponential decay."""
        
        #print("\n WHY THE ZEROS: ", pos_in_queue, " *** " ,queue_intensity, " *** ", time_entered)
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
            self.processEntries(all_entries, i)
            self.time+=self.time_res
            
            self.set_batch_id(i)
            
        return
    
    
    def set_batch_id(self, id):
		
        self.batchid = id
		
		
    def get_batch_id(self):
		
        return self.batchid
	
		
    def get_all_service_times(self):
        
        return self.all_serv_times
        
    
    def get_renege_rate(self, curr_queue_id):
		
        history=self.history[0]
        print("\n RENEGES: ", self.curr_obs_renege)
        print("\n HISTORY: ", self.history)
        #reneging_rate=len(np.where([(entry.reneged) for entry in self.history])[0])/len(self.history)
		
        #return reneging_rate


    def processEntries(self,entries=np.array([]), batchid=np.int16):
		
        #for i in range(len(self.dict_queues_obj["1"])):
        #    print("Clients:", self.dict_queues_obj["1"].customerid)
			
        for entry in entries:
            # print("Processing a new request entry...")
            #self.time=entry[0]            
            if entry[1]==True:
                # print("  Adding a new request into task queue...")                

                req = self.addNewRequest(entry[0], batchid)
                self.arr_prev_times = np.append(self.arr_prev_times, entry[0])
                
            else:                
                q_selector = random.randint(1, 2)
                observer = {}
                
                if q_selector == 1:
					#req = self.dict_queues_obj["1"][0]
                    self.queueID = "1" # Server1
                    self.serveOneRequest(self.dict_queues_obj["1"][0], entry[0], self.queueID) # Server1
                    #schedule.run_pending()                    
                    self.dispatch_queue_state(self.dict_queues_obj["1"], self.queueID, self.dict_queues_obj["2"]) #, req)
                    # observer = self.objObserv.get_obs()
                    # print("\n *** In comparison **** ", observer['QueueSize'], observer['Intensity'], observer['ServRate'])
                    time.sleep(1)
                    
                    #if self.capacity is not None and len(self.dict_queues_obj["1"]) >= self.capacity:
                    #    raise Exception("Queue has reached its capacity limit")
                        # sys.exit(1)
                        #return 
                        
                else:
					#req = self.dict_queues_obj["2"][0]
                    self.queueID = "2"
                    self.serveOneRequest(self.dict_queues_obj["2"][0], entry[0], self.queueID) # Server2
                    #schedule.run_pending()                    
                    self.dispatch_queue_state(self.dict_queues_obj["2"], self.queueID, self.dict_queues_obj["1"]) #, req)
                    # observer = self.objObserv.get_obs()
                    # print("\n ==== In comparison ==== ", observer['QueueSize'], observer['Intensity'], observer['ServRate'])
                    time.sleep(1)
                    
                    #if self.capacity is not None and len(self.dict_queues_obj["2"]) >= self.capacity:
                    #    raise Exception("Queue has reached its capacity limit")
                        # sys.exit(1)
                        #return 
                        
                # print("  Wait to Broadcasting the updated queue information...")
                # print("\n ************* Times Entered ************** ", self.arr_prev_times)
                #self.broadcastQueueInfo()
                    
                    
        return

    # def updateRngLog(self,pos):
    #     pos_reg=np.where(self.rng_pos_reg==pos)[0]
    #     if len(pos_reg>0):
    #         self.rng_counter[pos_reg]+=1
    #     else:
    #         self.rng_pos_reg=np.append(np.append(self.rng_pos_reg[np.where(self.rng_pos_reg>pos)],np.array([pos])),self.rng_pos_reg[np.where(self.rng_pos_reg>pos)])
    #         self.rng_counter=np.append(np.append(self.rng_counter[np.where(self.rng_pos_reg>pos)],np.array([1])),self.rng_counter[np.where(self.rng_pos_reg>pos)])
    #     return

    # def estimateRngRateCorrection(self,pos):
    #     counts=np.sum(self.rng_counter[np.where(self.rng_pos_reg<pos)])
    #     return self.srv_rate+counts/(self.time-self.init_time)


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
            expected_time_to_service_end = self.estimateMarkovWaitingTime(float(pose)) # , queue_intensity, time_entered)
            #time_local_service = self.generateLocalCompUtility(req)

        else:
            pose = lengthQueTwo+1
            server_id = "2" # Server2
            self.customerid = self.get_customer_id()
            self.customerid = "Batch"+str(self.get_batch_id())+"_"+self.customerid
            time_entered = self.time #self.estimateMarkovWaitingTime(lengthQueTwo)
            queue_intensity = self.arr_rate/rate_srv2
            expected_time_to_service_end = self.estimateMarkovWaitingTime(float(pose)) #, queue_intensity, time_entered)
            #time_local_service = self.generateLocalCompUtility(req)
            # time_entered, self.time
            
        req=Request(time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=self.batchid)
                    
        # #markov_model=self.markov_model,  
        self.dict_queues_obj[server_id] = np.append(self.dict_queues_obj[server_id], req)
        
        self.queueID = server_id
        
        self.curr_req = req
    

        '''
            if (self.learning_mode=='truncation') & (self.queue.size>=self.truncation_length):
                hstr_entry=HistoryEntry(False,True,0.0,self.generateLocalCompUtility(req),0)
                self.history=np.append(self.history,hstr_entry)
            elif (req.learning_mode=='transparent') & (req.makeRenegingDecision()):
                hstr_entry=HistoryEntry(False,True,0.0,self.generateLocalCompUtility(req),0)
                self.history=np.append(self.history,hstr_entry)
            else:
                self.queue=np.append(self.queue,req)
        '''
        
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

  
    def dispatch_queue_state(self, curr_queue, curr_queue_id, alt_queue ): #,req):                
        
        if curr_queue_id == "1":
            alt_queue_id = "2"
            curr_queue_state = self.get_queue_state(alt_queue_id)            
			
            for client in range(len(curr_queue)):
                req =  self.dict_queues_obj["1"][client]
                print(f"Dispatching state of server {alt_queue_id} to client {req.customerid} : {curr_queue_state}.")
                
        else:
            alt_queue_id = "1"
            curr_queue_state = self.get_queue_state(alt_queue_id)
            # renege_rate = self.get_renege_rate(self, curr_queue_id)
			
            for client in range(len(curr_queue)):
                req =  self.dict_queues_obj["2"][client]
                print(f"Dispatching state of server {alt_queue_id} to client {req.customerid} : {curr_queue_state}.")
        
        return 
    
    
    def get_long_run_avg_service_time(self, queue_id):
		
        total_service_time = 0
        total_served_requests = 0
    
        if queue_id == "1":
            for req in self.dict_queues_obj["1"]:
                total_service_time += req.exp_time_service_end
                total_served_requests += 1
        else:
            for req in self.dict_queues_obj["2"]:
                total_service_time += req.exp_time_service_end
                total_served_requests += 1
    
        if total_served_requests == 0:
            return 0
    
        return total_service_time / total_served_requests
        
        
    def get_queue_state(self, queueid):
		
        rate_srv1,rate_srv2 = self.get_server_rates()        
		
        if queueid == "1":		
            queue_intensity = self.objQueues.get_arrivals_rates()/ rate_srv1    
            customers_in_queue = self.dict_queues_obj["1"]   
            renege_rate = self.get_renege_rate( queueid)                
            
            state = {
                "total_customers": len(customers_in_queue),
                "intensity": queue_intensity,
                "capacity": self.capacity,
                "renege_rate": renege_rate
            }
        else:
			#serv_rate = self.get_server_rates()[1] #dict_servers_info["2"] 
            queue_intensity = self.objQueues.get_arrivals_rates()/ rate_srv2            
			#customers_in_queue = list(queue.queue)
            customers_in_queue = self.dict_queues_obj["2"]
            renege_rate = self.get_renege_rate( queueid)
			
            state = {
                "total_customers": len(customers_in_queue),
                "intensity": queue_intensity,
                "capacity": self.capacity,
                "renege_rate": renege_rate
            }
            
        return state


    # Run the scheduler in the background
    def run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(1)
  

    def serveOneRequest(self, to_delete, time_entrance, queueID):
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
            
            # take note of the observation ... self.time  queue_id,  serv_rate, intensity, time_in_serv, activity, rewarded, curr_pose
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, self.time-req.time_entrance, reward, len_queue_1, 2)   # req.exp_time_service_end,                                    
            self.history.append(self.objObserv.get_obs())
                
            #time_to_service_end = self.estimateMarkovWaitingTime(float(curr_pose), queue_intensity, reqObj.time_entrance)
            #time_local_service = self.generateLocalCompUtility(req)				                           
                                
            self.arr_prev_times = self.arr_prev_times[1:self.arr_prev_times.size]
            
            self.objQueues.update_queue_status(queueID)
            # req, curr_queue_id, alt_queue_id, customerid, serv_rate
            # Any Customers interested in jockeying or reneging when a request is processed get_curr_obs_jockey
            #print("\n Inside Server 1, calling the  decision procedures")
            self.makeJockeyingDecision(req, self.queueID, "2", req.customerid, serv_rate)
            self.makeRenegingDecision(req, self.queueID)

        else:                        
            req = self.dict_queues_obj["2"][0] # Server2
            serv_rate = self.dict_servers_info["2"] # Server2
            queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
            queueid = "2"   # Server2         
                        
            self.dict_queues_obj["2"] = self.dict_queues_obj["2"][1:self.dict_queues_obj["2"].size] # Server2
            
            reward = self.get_jockey_reward(req)
         
            self.queueID = queueID 
            self.dict_queues_obj["S2"] = self.dict_queues_obj["2"][1:self.dict_queues_obj["2"].size]      # Server2                  
            
            self.objObserv.set_obs(self.queueID, serv_rate, queue_intensity, self.time-req.time_entrance, reward, len_queue_2, 2)    # req.exp_time_service_end,                                  
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
            self.makeJockeyingDecision(req, self.queueID, "1", req.customerid, serv_rate)# Server1
            self.makeRenegingDecision(req, self.queueID)
        
        self.curr_req = req
                                                                  
        return


    def get_jockey_reward(self, req):
		
        reward = 0.0
        if not isinstance(req.customerid, type(None)):	
            if '_jockeyed' in req.customerid:
                if self.avg_delay+req.time_entrance < req.exp_time_service_end:
                    reward = 1.0
                else:
                    reward = 0.0
                    
        return reward
        
    
    def get_history(self):

        return self.history
    

    def get_curr_obs_jockey(self, queueid):
		
        obs_jockey = {}
        
        for obs in self.curr_obs_jockey:
            if obs.get("queue") == queueid and obs.get("at_pose") == pose and obs.get("this_busy") == intensity:
                obs_jockey = obs
                #print("\n JOCKEY OBSERVED: ", obs)
                #return obs
            #else:
            #    return {}	    
                
        return obs_jockey


    def get_curr_obs_renege(self, queueid):
        
        obs_renege = {}
        
        for obs in self.curr_obs_renege:
            if obs.get("queue") == queueid and obs.get("at_pose") == pose and obs.get("this_busy") == intensity:
                obs_renege = obs
                #print("\n RENENGE OBSERVED: ", obs)
                # return obs_renege
            #else:
            #    return {}
         
        return obs_renege

        #return self.curr_obs_renege 
    
    
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

 
    def broadcastQueueInfo(self):
        id_queue=np.array([req.id for req in self.queue])
        if self.learning_mode=='transparent':
            for request in self.queue:
                current_pos=np.where(id_queue==request.id)[0]
                request.serv_rate=self.srv_rate
                request.markov_model=msm.StateMachine(orig=self.markov_model)
        else:
            reneging_triggered=False
            for request in self.queue:
                current_pos=np.where(id_queue==request.id)[0]
                if request.pos_in_queue!=current_pos:
                    if(request.learn(current_pos,self.time)):
                        reneging_triggered=True
                        self.reqRenege(request.id)
            if reneging_triggered:
                self.broadcastQueueInfo()
        return


    def getCurrentCustomerQueue(self, customer_id):

        for customer in self.dict_queues_obj["2"]: # Server2
            if customer_id in customer:
                curr_queue = self.dict_queues_obj["2"]

        for customer in self.dict_queues_obj["1"]: # Server1
            if customer_id in customer:
                curr_queue = self.dict_queues_obj["1"]

        return curr_queue
        
        
    def makeRenegingDecision(self, req, queueid):
        # print("   User making reneging decision...")
        decision=False               
        
        if self.learning_mode=='transparent':
            self.max_cloud_delay=stats.erlang.ppf(self.certainty,a=self.pos_in_queue,loc=0,scale=1/self.serv_rate)
            #self.max_cloud_delay=self.estimateMarkovWaitingTime()
        else:			
            num_observations=min(len(self.get_curr_obs_renege(queueid)),len(self.history)) # if len(self.get_curr_obs_renege()) > 0 else 0 #self.history #self.observations.size
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
                self.max_cloud_delay=stats.erlang.ppf(self.certainty,loc=0,scale=mean_interval,a=req.pos_in_queue)
        
            if self.max_local_delay <= self.max_cloud_delay: # will choose to renege
                decision=True
                reqRenege(self, req, queueid, curr_pose, serv_rate, queue_intensity, self.max_local_delay, customerid, time_to_service_end, decision)
                
                # print("\n Renege Local Delay: ", self.max_local_delay)
                temp=stats.erlang.cdf(np.arange(self.max_local_delay,step=self.time_res),k_erlang,scale=scale_erlang)
                error_loss=np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))-np.sum(np.append([temp[0]],np.diff(temp))*np.exp(-req.pos_in_queue/np.arange(self.max_local_delay,step=self.time_res)))
                
            else:   #will choose to wait and learn -> Can we use the actor-critic here??
                decision=False
                #print('choose to wait')
                temp=stats.erlang.cdf(np.arange(self.max_local_delay,self.APPROX_INF+self.time_res,step=self.time_res),k_erlang,scale=scale_erlang)
                error_loss=np.sum(np.diff(temp)*np.exp(-req.pos_in_queue/np.arange(self.max_local_delay+self.time_res,self.APPROX_INF+self.time_res,step=self.time_res)))-np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))
                
            dec_error_loss = self.error_loss - error_loss
            self.error_loss = error_loss
            
            if dec_error_loss > 1-np.exp(-mean_interval):
                decision = False
            #else:
                #self.optimal_learning_achieved=True
                #print(self.observations)
            #if (not self.optimal_learning_achieved):
                self.min_amount_observations=len(self.get_curr_obs_renege(queueid)) # self.observations.size+1
                # print(self.min_amount_observations)
                
        self.curr_req = req
        
        return decision        


    def reqRenege(self, req, queueid, curr_pose, serv_rate, queue_intensity, time_local_service, customerid, time_to_service_end, decision):
        
        if "Server1" in queueid:
            self.queue = self.dict_queues_obj["1"]  # Server1           
        else:
            self.queue = self.dict_queues_obj["2"] # Server2

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
        
        self.objObserv.set_renege_obs(curr_pose, queue_intensity, decision,time_local_service, time_to_service_end, reward, queueid, 0.0)
        
        self.curr_obs_renege.append(self.objObserv.get_renege_obs(queueid, queue_intensity, curr_pose))
        print("\n RENEGED OSERVED: => ", self.curr_obs_renege)
        
        self.curr_req = req
        
        self.objQueues.update_queue_status(queueid)

    
    def reqJockey(self, curr_queue_id, dest_queue_id, req, customerid, serv_rate, dest_queue, exp_delay, decision):
		
        if "Server1" in curr_queue_id:
            self.queue = self.dict_queues_obj["1"]  # Server1           
        else:
            self.queue = self.dict_queues_obj["2"]
            
        for t in range(len(self.queue)):
            #print("\n ==> ", self.queue[t].customerid, customerid)
            
            if self.queue[t].customerid == customerid:
            #if req.customerid  # self.arr_rate
                curr_pose = t
            else:
                continue
                #return
        
        np.delete(curr_queue_id, curr_pose) # np.where(id_queue==req_id)[0][0])
        reward = 1.0
        req.time_entrance = self.time # timer()
        dest_queue = np.append( dest_queue, req)
        
        self.queueID = curr_queue_id
        # decision = True #=1.0,False=0.0
        
        req.customerid = req.customerid+"_jockeyed"
        
        if curr_queue_id == "1": # Server1
            queue_intensity = self.arr_rate/self.dict_servers_info["1"] # Server1
            
        else:
            queue_intensity = self.arr_rate/self.dict_servers_info["2"] # Server2
        
        reward = self.get_jockey_reward(req)
        
        
        time_alt_queue = self.estimateMarkovWaitingTime(float(len(dest_queue)+1)) #, queue_intensity, req.time_entrance)
                   
        # self.objObserv.set_obs(curr_queue_id, False, self.dict_servers_info[curr_queue_id], queue_intensity, decision,self.time-req.time_entrance, expectedJockeyWait, reward, len(dest_queue))            
        print("\n I have moved ", customerid, " from ",curr_queue_id, " to ", dest_queue_id ,"\n Time checks: expected vs. actual ", self.avg_delay ," ==== ",self.time-req.time_entrance) #self.objObserv.get_obs())                 
            
        self.objObserv.set_jockey_obs(curr_pose, queue_intensity, decision, time_alt_queue, req.exp_time_service_end, reward, 1.0)
        
        self.curr_obs_jockey.append(self.objObserv.get_jockey_obs(curr_queue_id, queue_intensity, curr_pose))  
                            
            
        #if "Server1" in id_queue:
        #    id_dest_queue = "Server2"
        #else:
        #    id_dest_queue = "Server1"
        
        #self.makeJockeyingDecision(req, id_queue, id_dest_queue, which_customer, serv_rate) #dest_queue)
        
        self.curr_req = req
        
        self.objQueues.update_queue_status(curr_queue_id)
        
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

        for t in range(len(curr_queue)):
            if curr_queue[t].customerid == customerid:
                curr_pose = t
		
        #if expectedJockeyWait < self.estimateMarkovWaitingTime(len(dest_queue)+1, queue_intensity, req.time_entrance):
        self.avg_delay = self.estimateMarkovWaitingTime(len(dest_queue)+1) #, queue_intensity, req.time_entrance)
        
        self.curr_req = req
        # print("\n Jockeys Avg Delay: ", self.avg_delay)
        if req.time_entrance < self.avg_delay:
            decision = True
            self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision)

        # ToDo:: There is also the case of the customer willing to take the risk
        #        and jockey regardless of the the predicted loss -> Customer does not
        #        care anymore whether they incur a loss because they have already joined anyway
        #        such that reneging returns more loss than the jockeying decision

        #else:
            #decision = False #=0.0,True=1.0
            # ToDo:: revisit this for the case of jockeying.
            #        Do not use the local cloud delay

            #reward = 0.0
            #self.objObserv.set_obs(curr_queue_id, False, self.dict_servers_info[curr_queue_id], queue_intensity, decision, timer()-req.time_entrance, req.exp_time_service_end, reward, len(curr_queue), "jockeyed")
            #self.objObserv.set_obs(curr_queue_id, self.dict_servers_info[curr_queue_id], queue_intensity,
            #     timer()-req.time_entrance, reward, len(curr_queue), 0.)                                                                    
            
            # self.history.append(self.objObserv.get_obs())
            # self.curr_obs_jockey.append(self.objObserv.get_obs())  
            #self.queueID = curr_queue_id                      
        
        #if curr_pose >= 1:
        #    self.serveOneRequest(curr_queue[0], req.time_entrance, self.queueID)
        
        return decision
        

def main():
	
    utility_basic = 1.0
    discount_coef = 0.1
    requestObj = RequestQueue(utility_basic, discount_coef)
    duration = 3
    requestObj.run(duration)                
	
    
if __name__ == "__main__":
    main()
