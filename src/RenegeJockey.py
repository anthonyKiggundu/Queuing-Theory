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
from tqdm import tqdm
from jockey import *
import MarkovStateMachine as msm
from plotly.validators.histogram2dcontour import _zmid
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
        
        #self.dict_queues = self.generate_queues()
        #self.dict_servers = self.queue_setup_manager()

        self.queue_states = ["Busy","Empty"]
        
        
    def queue_setup_manager(self):
                
        # deltalambda controls the difference between the service rate of either queues    
        deltaLambda=random.randint(1, 2)
        
        serv_rate_one=self.sampled_arr_rate + deltaLambda 
        serv_rate_two=self.sampled_arr_rate - deltaLambda

        _serv_rate_one=serv_rate_one / 2
        _serv_rate_two=serv_rate_two / 2
        
        # print("\n .... ", self.dict_servers)
        self.dict_servers["Server1"] = _serv_rate_one
        self.dict_servers["Server2"] = _serv_rate_two
        
        # print("\n Current Arrival Rate:", self.sampled_arr_rate, "Server1:", _serv_rate_one, "Server2:", _serv_rate_two) 


    def get_dict_servers(self):

        self.queue_setup_manager()
        return self.dict_servers        


    def get_curr_preferred_queues (self):
        # queues = Queues()
        #self.all_queues = self.generate_queues() #queues.generate_queues()

        curr_queue = self.dict_queues.get("Server1")
        alter_queue = self.dict_queues.get("Server2")

        return (curr_queue, alter_queue)

    
    def generate_queues(self):
        
        for i in range(self.num_of_queues):
            code_string = "Server%01d" % (i+1)
            queue_object = np.array([])
            self.dict_queues.update({code_string: queue_object})

        # return self.dict_queues

    def get_dict_queues(self):

        self.generate_queues()
        return self.dict_queues

    def get_arrivals_rates(self):

        return self.sampled_arr_rate

    
    
class Request:

    LEARNING_MODES=['stochastic','transparent' ] # [ online','fixed_obs', 'truncation','preemption']
    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self,time_entrance=pyg.time.get_ticks(),pos_in_queue=0,utility_basic=0.0,discount_coef=0.0,outage_risk=0.1,
                 customerid="", learning_mode='online',min_amount_observations=1,time_res=1.0, #markov_model=a2c.A2C,serv_rate=1.0,
                 dist_local_delay=stats.expon,para_local_delay=[1.0,2.0,10.0]): 
        
        # self.id=id #uuid.uuid1()
        customerid = "Customer_"+str(pos_in_queue+1)
        self.customerid = customerid
        # time_entrance = self.estimateMarkovWaitingTime()
        self.time_entrance=time_entrance #[0] # ToDo:: still need to find out why this turns out to be an array
        # self.time_last_observation=float(time_entrance)
        self.pos_in_queue=int(pos_in_queue)
        self.utility_basic=float(utility_basic)
        self.discount_coef=float(discount_coef)
        self.certainty=1.0-float(outage_risk)
        self.certainty=float(outage_risk)

        # self.arr_prev_times = np.array([])
        # print("\n TIME ENTERED: ", self.time_entrance, time_entrance, self.arr_prev_times)
        # arr_prev_times = np.append(arr_prev_times,  self.time_entrance)

        if (self.certainty<=0) or (self.certainty>=1):
            raise ValueError('Invalid outage risk threshold! Please select between (0,1)')
        #if Request.LEARNING_MODES.count(learning_mode)==0:
        #   raise ValueError('Invalid learning mode! Please select from '+str(Request.learning_modes))
        #else:
        #    self.learning_mode=str(learning_mode)
            
        self.min_amount_observations=int(min_amount_observations)
        self.time_res=float(time_res)
        # self.markov_model=markov_model #msm.StateMachine(orig=markov_model)
        #if learning_mode=='transparent':
        #   self.serv_rate=self.markov_model.feature
        #else:
        #   self.serv_rate=float(serv_rate)
        queueObj = Queues()
        #queue_srv_rates = queueObj.queue_setup_manager()

        #queueObj.generate_queues()
        #queueObj.queue_setup_manager()

        queue_srv_rates = queueObj.get_dict_servers()

        if queue_srv_rates.get("Server1"):
            self.serv_rate = queue_srv_rates.get("Server1")
        else:
            self.serv_rate = queue_srv_rates.get("Server2")
              # srv1 = queueObjs.get("Server1")

        #if len(self.arr_prev_times) <= 0:
        #    diff = self.time_entrance #arr_prev_times[0]
        #    print("\n Diff: ", diff)
        #elif len(self.arr_prev_times) > 0:
        #    print("\n Rates --> ",  self.arr_prev_times, " ------  ",  self.arr_prev_times[len(self.arr_prev_times)-1])

        self.dist_local_delay=dist_local_delay
        self.loc_local_delay=np.random.uniform(low=float(para_local_delay[0]),high=(para_local_delay[1]))
        self.scale_local_delay=float(para_local_delay[2])
        self.max_local_delay=self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay)
        self.max_cloud_delay=float(queueObj.get_arrivals_rates()/self.serv_rate) # np.inf
       
        # print("\n ****** ",self.loc_local_delay, " ---- " , self.time_entrance-arr_prev_times[len(arr_prev_times)-1])
        self.observations=np.array([])
        self.error_loss=1
        self.optimal_learning_achieved=False

        # arr_prev_times = np.append(arr_prev_times,  self.time_entrance)

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
            #self.max_cloud_delay=stats.erlang.ppf(self.certainty,a=self.pos_in_queue,loc=0,scale=1/self.serv_rate)
            self.max_cloud_delay=self.estimateMarkovWaitingTime()
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
        #elif self.learning_mode=='preemption':
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
            self.history = np.append(self.history,obs_entry)
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
    

class Observations:
    def __init__(self, reneged=False, serv_rate=0.0, queue_intensity=0.0, jockeyed=False, time_waited=0.0,end_utility=0.0, reward=0.0, queue_size=0): # reward=0.0, 
        self.reneged=bool(reneged)
        self.serv_rate = serv_rate
        self.queue_intensity = queue_intensity
        self.jockeyed=bool(jockeyed)
        self.time_waited=float(time_waited)
        self.end_utility=float(end_utility)
        self.reward= reward # id_queue
        self.queue_size=int(queue_size)
        self.obs = {} # OrderedDict() #{} # self.get_obs()

        # self.set_obs(self.reneged, self.serv_rate, self.queue_intensity, self.jockeyed, 
        #                       self.time_waited, self.end_utility, self.reward, self.min_num_observations )

        return


    def set_obs (self, queue_id, reneged, serv_rate, intensity, jockeyed, time_in_serv, utility, rewarded, curr_pose):
        self.obs = {
			        "ServerID": queue_id,
                    "EndUtility":utility,
                    "Intensity":intensity,
                    "Jockey":jockeyed,
                    "QueueSize":curr_pose,
                    "Renege":reneged,
                    "Reward":rewarded,
                    "ServRate":serv_rate,
                    "Waited":time_in_serv,
                }


    def get_obs (self):
        
        return self.obs


class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, utility_basic, discount_coef, # markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0):
        
        # self.mar:kov_model=msm.StateMachine(orig=markov_model)
        # self.arr_rate=float(arr_rate) arr_rate, queue=np.array([])
        self.customerid = customerid
        self.utility_basic=float(utility_basic)
        self.local_utility = 0.0
        self.compute_counter = 0
        self.avg_delay = 0.0
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
        # self.history = np.array([])
        self.decision_rule=str(decision_rule)
        self.truncation_length=float(truncation_length)
        self.preempt_timeout=float(preempt_timeout)
        self.preempt_timer=self.preempt_timeout
        self.time_res=float(time_res)
        self.dict_queues_obj = {}
        self.dict_servers_info = {}
        self.history = {}
        self.curr_obs_jockey = {}
        self.curr_obs_renege = {}

        self.arr_prev_times = np.array([])

        self.objQueues = Queues()
        self.objRequest = Request()
        self.objObserv = Observations()

        self.dict_queues_obj = self.objQueues.get_dict_queues()
        self.dict_servers_info = self.objQueues.get_dict_servers()
        self.jockey_threshold = 1
        self.reward = 0.0
        self.queue_states = ["Busy","Empty"]

        self.arr_rate = self.objQueues.get_arrivals_rates()

        self.objObserve = Observations()
        self.all_times = []
        self.all_serv_times = []
        self.queueID = ""

        # self.rng_pos_reg=np.array([])
        self.rng_counter=np.array([])
        # if self.markov_model.feature!=None:
        #    self.srv_rate=self.markov_model.feature
        return


    def estimateMarkovWaitingTimeOriginal(self, pos_in_queue):
        # print("   Estimating Markov waiting time...")
        # queue_indices=np.arange(pos_in_queue-1)+1
        
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
        
        # return 
        self.avg_delay = (samples-1)*self.time_res
 
        return self.avg_delay


    def estimateMarkovWaitingTime(self, pos_in_queue, queue_intensity, time_entered):
        """Calculate the amount after a certain time with exponential decay."""
        
        self.avg_delay = pos_in_queue * math.exp(-queue_intensity * time_entered)

        return self.avg_delay


        '''
        samples=1
        num_states = len(self.queue_states)

        start_belief=np.matrix(np.zeros(num_states).reshape(1,num_states)[0],np.float64).T
        start_belief[self.markov_model.current_state]=1.0
        cdf=0
        while cdf<=self.certainty:
            eff_srv=self.markov_model.integratedEffectiveFeature(samples, start_belief)
            cdf=1-sum((eff_srv**i*np.exp(-eff_srv)/np.math.factorial(i) for i in queue_indices))
            # print([eff_srv,cdf])
            samples+=1
        return (samples-1)*self.time_res
        '''

    def get_times_entered(self):
        
        
        #self.arr_prev_times = np.append(self.arr_prev_times, time_entered)
        print("\n ************ ", self.arr_prev_times)


    # staticmethod
    def get_queue_sizes(self):
        q1_size = len(self.dict_queues_obj.get("Server1"))
        q2_size = len(self.dict_queues_obj.get("Server2"))

        return (q1_size, q2_size)

    def get_server_rates(self):
        srvrate1 = self.dict_servers_info.get("Server1")
        srvrate2 = self.dict_servers_info.get("Server2")

        return (srvrate1, srvrate2)


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
        #stop_time=self.time+duration

        # print("\n Directly jumping into the run ,,,,,,,") # , self.dict_queues_obj)

        # print("\n ============= ", self.dict_queues_obj["Server1"], "\n ************* ", self.dict_queues_obj["Server2"])
        #while self.time<=stop_time:
        if progress_bar!=None:
            loop=tqdm(range(steps),leave=False,desc='     Current run')
        else:
            loop=range(steps)
        for i in loop:
            if progress_log:
                print("Step",i,"/",steps)
            # ToDo:: is line below the same as the update_parameters() in the a2c.py    
            # self.markov_model.updateState()

            srv_1 = self.dict_queues_obj.get("Server1")
            srv_2 = self.dict_queues_obj.get("Server2")

            if len(srv_1) < len(srv_2):
                self.queue = srv_2
                self.srv_rate = self.dict_servers_info.get("Server2")

            else:            
                self.queue = srv_1
                self.srv_rate = self.dict_servers_info.get("Server1")

            # self.srv_rate=self.markov_model.feature           
            # Here during the run the customer assess the new estimate
            # of the markovian time left to completing the task
            # If this time is more then renege
            
            '''
            if (self.queue.size>0): # and self.learning_mode=='preemption'):
                #self.preempt_timer-=self.time_res
                #print("\n Actual Size --->> ", self.queue.size, " and ", self.preempt_timer, self.time_res)
                #if self.preempt_timer<=0:
                    print("\n ****Hello Sunshine, I just left your bad queue ****** ")
                    self.reqRenege(self.queue[0].id)
                    self.preempt_timer=self.preempt_timeout

            elif (abs(len(srv_1) - len(srv_2)) >= self.jockey_threshold):
                print("\n .............. Hello Jockeying..............")
                self.reqJockey()
            '''
                    
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
            self.processEntries(all_entries)
            self.time+=self.time_res
        return
    
    
    def get_all_service_times(self):
        
        return self.all_serv_times


    def processEntries(self,entries=np.array([])):
        for entry in entries:
            # print("Processing a new request entry...")
            #self.time=entry[0]
            # print("\n Who is entry -> ",entry, entry[0])
            if entry[1]==True:
                # print("  Adding a new request into task queue...")
                #print("\n Entry: ", entry[0])
                self.addNewRequest(entry[0])
                self.arr_prev_times = np.append(self.arr_prev_times, entry[0])
            else:
                # print("\n Entry: ", entry[0])
                q_selector = random.randint(1, 2)
                print("  Serving a pending request...in queue ", q_selector)
                if q_selector == 1:
                    self.queueID = "Server1"
                    self.serveOneRequest(self.dict_queues_obj["Server1"][0], entry[0], self.queueID)
                else:
                    self.queueID = "Server2"
                    self.serveOneRequest(self.dict_queues_obj["Server2"][0], entry[0], self.queueID)

                print("  Wait to Broadcasting the updated queue information...")
                # print("\n ************* Times Entered ************** ", self.arr_prev_times)
                # self.broadcastQueueInfo()
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
        then a reward is given, else a penalty.
    '''

    def getRenegeRewardPenalty (self):

        if self.local_utility < self.avg_delay:
            self.reward = 1
        else:
            self.reward = 0

        return self.reward

    def generateLocalCompUtility(self, req):
        #req=Request(req)
        self.compute_counter = self.compute_counter + 1
        # local_delay=req.dist_local_delay.rvs(loc=req.loc_local_delay,scale=req.scale_local_delay)
        local_delay=req.dist_local_delay.rvs(loc=req.loc_local_delay,scale=2.0) #req.scale_local_delay)
        # print("\n Local :", local_delay, req.time_entrance, self.time)
        delay=float(self.time-req.time_entrance)+local_delay        
        self.local_utility = float(req.utility_basic*np.exp(-delay*req.discount_coef))

        self.avg_delay = (self.local_utility + self.avg_delay)/self.compute_counter

        return self.local_utility
    
    
    def generateExpectedJockeyCloudDelay (self, req, id_queue):
        #id_queue = np.array([req.id for req in self.queue])
        # req = self.queue[np.where(id_queue==req_id)[0][0]]

        total_jockey_delay = 0.0
        
        init_delay = float(self.time - req.time_entrance)
        
        if id_queue == "Server1":  
            curr_queue =self.dict_queues_obj["Server1"]        
            alt_queue = self.dict_queues_obj["Server2"]
            pos_in_alt_queue = len(alt_queue)+1
            # And then compute the expected delay here using Little's Law
            expected_delay_in_alt_queue_pose = float(pos_in_alt_queue/self.arr_rate) #self.sampled_arr_rate)
            total_jockey_delay = expected_delay_in_alt_queue_pose + init_delay
        else:
            curr_queue =self.dict_queues_obj["Server2"]        
            alt_queue = self.dict_queues_obj["Server1"]
            pos_in_alt_queue = len(alt_queue)+1
            # And then compute the expected delay here using Little's Law
            expected_delay_in_alt_queue_pose = float(pos_in_alt_queue/self.arr_rate) # self.sampled_arr_rate)
            total_jockey_delay = expected_delay_in_alt_queue_pose + init_delay
            #self.queue= queue
            
        return total_jockey_delay
               

    def addNewRequest(self, time_entered):
        # Join the shorter of either queues
               
        lengthQueOne = len(self.dict_queues_obj["Server1"])
        lengthQueTwo = len(self.dict_queues_obj["Server2"])        

        if lengthQueOne < lengthQueTwo:
            # time_entered = self.estimateMarkovWaitingTime(lengthQueOne)
            pose = lengthQueOne+1
            server_id = "Server1"
            self.customerid = "Customer_"+str(pose+1)

            #req=Request(time_entrance=time_entered,pos_in_queue=lengthQueOne+1,utility_basic=self.utility_basic,
            #        discount_coef=self.discount_coef,outage_risk=self.outage_risk,learning_mode=self.learning_mode,
            #        min_amount_observations=self.min_amount_observations,time_res=self.time_res, #markov_model=self.markov_model,
            #        dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay)

            #self.dict_queues_obj["Server1"] = np.append(self.dict_queues_obj["Server1"], req)

        else:
            pose = lengthQueTwo+1
            server_id = "Server2"
            self.customerid = "Customer_"+str(pose+1)
            # time_entered = self.estimateMarkovWaitingTime(lengthQueTwo)

        req=Request(time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #markov_model=self.markov_model,
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay)

        self.dict_queues_obj[server_id] = np.append(self.dict_queues_obj[server_id], req)
    

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
        
        return


    def getCustomerID(self):

        return self.customerid


    def getCustomerPose(self):

        pass       
 

    def serveOneRequest(self, to_delete, time_entrance, queueID):
        #randomly select which queue to process at a time t+1
        q_selector = random.randint(1, 2)
        reward = self.getRenegeRewardPenalty()

        if "Server1" in queueID:                
            req =  self.dict_queues_obj["Server1"][0]
            serv_rate = self.dict_servers_info["Server1"]
            queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
            queueid = "Server1"
            len_queue_1,len_queue_2 = self.get_queue_sizes() 
            # hstr_entry = Observations(False,serv_rate, queue_intensity, False,self.time-time_entrance,self.generateLocalCompUtility(req),reward, req.min_amount_observations)
            # self.objObserv.set_obs(False,serv_rate, queue_intensity, False,self.time-time_entrance,self.generateLocalCompUtility(req),reward, len_queue_1)
            self.dict_queues_obj["Server1"] = self.dict_queues_obj["Server1"][1:self.dict_queues_obj["Server1"].size]

            for customer in range(len(self.dict_queues_obj["Server1"])):
                curr_pose = customer+1 #self.dict_queues_obj["Server1"]
                reqObj = self.dict_queues_obj["Server1"][customer]
                #print("\n",curr_pose," ****CHECK 1 ",reqObj.time_entrance) #, " --------- ",t[curr_pose][0])
                time_to_service_end = self.estimateMarkovWaitingTime(float(curr_pose), queue_intensity, reqObj.time_entrance)
                time_local_service = self.generateLocalCompUtility(req)
                if time_local_service < time_to_service_end:
                    #print("\n ****Hello Sunshine, I just left your bad queue ****** ", reqObj.customerid)
                    self.reqRenege(reqObj, queueID, curr_pose, serv_rate, queue_intensity, time_local_service, reqObj.customerid)
                    #self.objObserv.set_obs(True, serv_rate, queue_intensity, False,self.time-reqObj.time_entrance,time_local_service, reward, curr_pose)
                    #self.curr_obs.update({
                    #     "EndUtility":time_local_service,
                    #    "Intensity": queue_intensity,
                    #    "Jockey": False,
                    #    "QueueSize": curr_pose,
                    #    "Renege": True, 
                    #    "Reward": reward,
                    #    "ServRate": serv_rate,                                                 
                    #    "Waited":self.time-reqObj.time_entrance                        
                    #})
                    #self.history=np.append(self.history,self.objObserv.get_obs())
                    #self.preempt_timer=self.preempt_timeout
                    # True, serv_rate, queue_intensity, False,self.time-reqObj.time_entrance,time_local_service, reward, curr_pose

                elif (abs(len(self.dict_queues_obj["Server1"]) - len(self.dict_queues_obj["Server2"])) >= self.jockey_threshold):
                    #print("\n .............. Hello Jockeying..............")
                    #self.objObserv.set_obs(False, serv_rate, queue_intensity, True,self.time-reqObj.time_entrance,time_local_service, reward, curr_pose)
                    #self.curr_obs.update({
                    #    "EndUtility": time_local_service,
                    #    "Intensity": queue_intensity,                                            
                    #    "Jockey": True,
                    #    "QueueSize": curr_pose,
                    #    "Renege": False,
                    #    "Reward": reward,
                    #    "ServRate": serv_rate,
                    #    "Waited": self.time-reqObj.time_entrance                       
                    #})
                    self.reqJockey(queueid, reqObj, reqObj.customerid, serv_rate)
                    #self.history=np.append(self.history,self.objObserv.get_obs())
                    # False, serv_rate, queue_intensity, True,self.time-reqObj.time_entrance,time_local_service, reward, curr_pose
                
            
                
                
            self.arr_prev_times = self.arr_prev_times[1:self.arr_prev_times.size]

        else:                        
            req = self.dict_queues_obj["Server2"][0]
            serv_rate = self.dict_servers_info["Server2"]
            queue_intensity = self.objQueues.get_arrivals_rates()/ serv_rate
            queueid = "Server2"
            # hstr_entry = Observations(False, serv_rate, queue_intensity, False,self.time-time_entrance,self.generateLocalCompUtility(req), reward, req.min_amount_observations)            
            #self.objObserv.set_obs(False,serv_rate, queue_intensity, False,self.time-time_entrance,self.generateLocalCompUtility(req),reward, req.min_amount_observations)
                        
            self.dict_queues_obj["Server2"] = self.dict_queues_obj["Server2"][1:self.dict_queues_obj["Server2"].size]

            for customer in range(len(self.dict_queues_obj["Server2"])):
                curr_pose = customer+1
                reqObj = self.dict_queues_obj["Server2"][customer]
                #t = self.get_all_times()
                #print("\n ",curr_pose," ****CHECK 2 ",reqObj.time_entrance) #, " 00000 000 ",t[curr_pose][0])
                time_to_service_end = self.estimateMarkovWaitingTime(float(curr_pose), queue_intensity, reqObj.time_entrance)
                time_local_service = self.generateLocalCompUtility(req)
                if time_local_service < time_to_service_end:
                    #print("\n ****Hello Sunshine, I just left your bad queue ****** ", reqObj.customerid)
                    self.reqRenege(reqObj, queueID, curr_pose, serv_rate, queue_intensity, time_local_service, reqObj.customerid) #, time_to_service_end) #self.queue[0].id)
                    #self.objObserv.set_obs(True, serv_rate, queue_intensity, False,self.time-reqObj.time_entrance,time_local_service, reward, curr_pose)
                    #self.curr_obs.update({
                    #    "EndUtility":time_local_service,
                    #   "Intensity": queue_intensity,
                    #    "Jockey": False,
                    #    "QueueSize": curr_pose,
                    #    "Renege": True, 
                    #    "Reward": reward,
                    #    "ServRate": serv_rate,                                                 
                    #    "Waited":self.time-reqObj.time_entrance                                                                         
                    #})
                    #self.preempt_timer=self.preempt_timeout
                    # self.history=np.append(self.history,self.objObserv.get_obs())

                elif (abs(len(self.dict_queues_obj["Server1"]) - len(self.dict_queues_obj["Server2"])) >= self.jockey_threshold):
                    #print("\n .............. Hello Jockeying ..............")
                    #self.objObserv.set_obs(False, serv_rate, queue_intensity, True,self.time-reqObj.time_entrance,time_local_service, reward, curr_pose)
                    #self.curr_obs.update({
                    #    "EndUtility": time_local_service,
                    #    "Intensity": queue_intensity,                                            
                    #    "Jockey": True,
                    #    "QueueSize": curr_pose,
                    #    "Renege": False,
                    #    "Reward": reward,
                    #    "ServRate": serv_rate,
                    #    "Waited": self.time-reqObj.time_entrance                                                                    
                    #})
                    
                    self.reqJockey(queueid, reqObj, reqObj.customerid, serv_rate)
                    # self.history=np.append(self.history,self.objObserv.get_obs())
                
           # print("\n Current Observation: ", self.get_obs())
                
            self.arr_prev_times = self.arr_prev_times[1:self.arr_prev_times.size]

        return

    
    def get_history(self):

        return self.history
    

    def get_curr_obs_jockey(self):

        return self.curr_obs_jockey


    def get_curr_obs_renege(self):

        return self.curr_obs_renege
    
    
    def get_curr_queue_id(self):
        
        return self.queueID

 
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

        for customer in self.dict_queues_obj["Server2"]:
            if customer_id in customer:
                curr_queue = self.dict_queues_obj["Server2"]

        for customer in self.dict_queues_obj["Server1"]:
            if customer_id in customer:
                curr_queue = self.dict_queues_obj["Server1"]

        return curr_queue


    def reqRenege(self, req, queueid, curr_pose, serv_rate, queue_intensity, time_local_service, customerid): #, time_to_service_end):
        #id_queue=np.array([req.id for req in self.queue])
        if "Server1" in queueid:
            self.queue = self.dict_queues_obj["Server1"] 
            #self.getCurrentCustomerQueue(req_id)
        else:
            self.queue = self.dict_queues_obj["Server2"]

        # find customer in queue by index
        # index = np.argwhere(self.queue==req_id)
        # req = self.queue[index]
        reward = self.getRenegeRewardPenalty()
        #hstr_entry = self.objObserve(True,False,self.time-req.time_entrance,self.generateLocalCompUtility(req), reward, curr_pose)
        self.objObserv.set_obs(queueid, True, serv_rate, queue_intensity, False,self.time-req.time_entrance,self.generateLocalCompUtility(req), reward, len(self.queue))

        for t in range(len(self.queue)):
            if self.queue[t].customerid == customerid:
                curr_pose = t
        
        #self.history.update({"ServerID":queueid,
        #                    "Status":self.objObserv.get_obs()                                 
        #                })
        
        self.history.update({queueid:self.objObserv.get_obs()})
        
        self.queue = np.delete(self.queue, curr_pose) # index)
        
        
        self.curr_obs_renege.update({
			"ServerID": queueid, #self.queue,
            "EndUtility": time_local_service,
            "Intensity": queue_intensity,                                            
            "Jockey": False,
            "QueueSize": curr_pose,
            "Renege": True,
            "Reward": reward,
            "ServRate": serv_rate,
            "Waited": self.time-req.time_entrance  
            })    
        
        return
    
    def reqJockey(self, id_queue, req, which_customer, serv_rate):
        #id_queue=np.array([req.id for req in self.queue])
        #req=self.queue[np.where(id_queue==req_id)[0][0]]

        if "Server1" in id_queue:
            id_dest_queue = "Server2"
        else:
            id_dest_queue = "Server1"
        
        self.makeJockeyingDecision(req, id_queue, id_dest_queue, which_customer, serv_rate) #dest_queue)
        
        return


    def makeJockeyingDecision(self, req, curr_queue_id, alt_queue_id, customerid, serv_rate):
        # We make this decision if we have already joined the queue
        # First we analyse our current state -> which server, server intensity and expected remaining latency
        # Then we get information about the state of the alternative queue
        # Evaluate input from the actor-critic once we get in the alternative queue
        decision=False
        expectedJockeyWait = self.generateExpectedJockeyCloudDelay(req, curr_queue_id)
        
        queue_intensity = self.arr_rate/self.dict_servers_info[alt_queue_id]

        curr_queue = self.dict_queues_obj.get(curr_queue_id)
        dest_queue = self.dict_queues_obj.get(alt_queue_id)

        for t in range(len(curr_queue)):
            if curr_queue[t].customerid == customerid:
                curr_pose = t


        if expectedJockeyWait < self.estimateMarkovWaitingTime(len(dest_queue)+1, queue_intensity, req.time_entrance):
            np.delete(curr_queue, curr_pose) # np.where(id_queue==req_id)[0][0])
            reward = 1.0
            dest_queue = np.append( dest_queue, req)
            decision = True            
            self.objObserv.set_obs(curr_queue_id, False, self.dict_servers_info[curr_queue_id], queue_intensity, decision,self.time-req.time_entrance, expectedJockeyWait, reward, len(curr_queue))
            
            print("\n I have moved ", customerid, " from ",curr_queue_id, " to ", alt_queue_id ,"\n Observation: ", self.objObserv.get_obs())
            
            #obs_entry = self.objObserve(False,self.dict_servers_info[curr_queue_id], queue_intensity, decision,self.time-req.time_entrance, expectedJockeyWait, reward, curr_pose)
            
            #self.history = np.append(self.history,self.objObserve.get_obs()) #obs_entry)
            #self.history.update({"ServerID":curr_queue_id,
            #                    "Status":self.objObserv.get_obs()
            #                })
        
            self.history.update({curr_queue_id:self.objObserv.get_obs()})
            
            self.curr_obs_jockey.update({
				"ServerID": curr_queue_id,
                "EndUtility": expectedJockeyWait,
                "Intensity": queue_intensity,                                            
                "Jockey": True,
                "QueueSize": curr_pose,
                "Renege": False,
                "Reward": reward,
                "ServRate": serv_rate,
                "Waited": self.time-req.time_entrance  
            }) 
            
            #print("\n Current Observation in Jockey: ", self.objObserve.get_obs())

        # ToDo:: There is also the case of the customer willing to take the risk
        #        and jockey regardless of the the predicted loss -> Customer does not
        #        care anymore whether they incur a loss because they have already joined anyway
        #        such that reneging returns more loss than the jockeying decision

        else:
            decision = False
            # ToDo:: revisit this for the case of jockeying.
            #        Do not use the local cloud delay

            reward = 0.0
            self.objObserv.set_obs(curr_queue_id, False, self.dict_servers_info[curr_queue_id], queue_intensity, decision, self.time-req.time_entrance, expectedJockeyWait, reward, len(curr_queue))
            
            # print("\n", customerid," has RENEGED ", "\n Observation: ", self.objObserv.get_obs())
            
            #obs_entry = self.objObserve.get_obs()
            # self.history = np.append(self.history,self.objObserve.get_obs()) # obs_entry)
            
            #self.history.update({"ServerID":curr_queue_id,
            #                    "Status":self.objObserv.get_obs()
            #                })

            self.history.update({curr_queue_id:self.objObserv.get_obs()})
            
            self.curr_obs_jockey.update({
				"ServerID": curr_queue_id,
                "EndUtility": expectedJockeyWait,
                "Intensity": queue_intensity,                                            
                "Jockey": False,
                "QueueSize": len(dest_queue),
                "Renege": False,
                "Reward": reward,
                "ServRate": serv_rate,
                "Waited": self.time-req.time_entrance  
            })

        return decision

