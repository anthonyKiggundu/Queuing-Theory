###############################################################################
# Author: anthony.kiggundu@dfki.de

import numpy as np
import scipy.stats as stats
import uuid
import A2C as a2c
import time
from tqdm import tqdm
from jockey import *
###############################################################################

class Request:

    LEARNING_MODES=['online','fixed_obs','transparent','truncation','preemption']
    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self,time_entrance,pos_in_queue,utility_basic,discount_coef,outage_risk=0.1,
                 learning_mode='online',min_amount_observations=1,time_res=1.0,markov_model=a2c.A2C,
                 serv_rate=1.0,dist_local_delay=stats.expon,para_local_delay=[1.0,2.0,10.0]):
        
        self.id=uuid.uuid1()
        self.time_entrance=time_entrance
        # self.time_last_observation=float(time_entrance)
        self.pos_in_queue=int(pos_in_queue)
        self.utility_basic=float(utility_basic)
        self.discount_coef=float(discount_coef)
        self.certainty=1.0-float(outage_risk)
        # self.certainty=float(outage_risk)
        if (self.certainty<=0) or (self.certainty>=1):
            raise ValueError('Invalid outage risk threshold! Please select between (0,1)')
        if Request.LEARNING_MODES.count(learning_mode)==0:
            raise ValueError('Invalid learning mode! Please select from '+str(Request.learning_modes))
        else:
            self.learning_mode=str(learning_mode)
        self.min_amount_observations=int(min_amount_observations)
        self.time_res=float(time_res)
        self.markov_model=markov_model #msm.StateMachine(orig=markov_model)
        if learning_mode=='transparent':
            self.serv_rate=self.markov_model.feature
        else:
            self.serv_rate=float(serv_rate)
        self.dist_local_delay=dist_local_delay
        self.loc_local_delay=np.random.uniform(low=float(para_local_delay[0]),high=(para_local_delay[1]))
        self.scale_local_delay=float(para_local_delay[2])
        self.max_local_delay=self.dist_local_delay.ppf(self.certainty,loc=self.loc_local_delay,scale=self.scale_local_delay)
        self.max_cloud_delay=np.inf
        self.observations=np.array([])
        self.error_loss=1
        self.optimal_learning_achieved=False
        return


    def learn(self,new_pos,new_time):
        steps_forward=self.pos_in_queue-int(new_pos)
        # self.time_last_observation=float(new_time)
        self.pos_in_queue=int(new_pos)
        self.observations=np.append(self.observations,(new_time-self.time_entrance-np.sum(self.observations))/steps_forward)
        return self.makeRenegingDecision()


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
        if self.learning_mode=='truncation':
            decision=False
        elif self.learning_mode=='preemption':
            decision=False
        elif self.learning_mode=='transparent':
            decision=(self.max_local_delay<=self.max_cloud_delay)
        elif self.learning_mode=='fixed_obs':
            decision=(self.max_local_delay<=self.max_cloud_delay) & (num_observations>=self.min_amount_observations)
        elif scale_erlang==0:
            decision=False
        else: # mode='learning' , scale_erlang>0
            if self.max_local_delay<=self.max_cloud_delay: # will choose to renege
                decision=True
                #print('choose to rng')
                temp=stats.erlang.cdf(np.arange(self.max_local_delay,step=self.time_res),k_erlang,scale=scale_erlang)
                error_loss=np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))-np.sum(np.append([temp[0]],np.diff(temp))*np.exp(-self.pos_in_queue/np.arange(self.max_local_delay,step=self.time_res)))
            else:   #will choose to wait and learn
                decision=False
                #print('choose to wait')
                temp=stats.erlang.cdf(np.arange(self.max_local_delay,self.APPROX_INF+self.time_res,step=self.time_res),k_erlang,scale=scale_erlang)
                error_loss=np.sum(np.diff(temp)*np.exp(-self.pos_in_queue/np.arange(self.max_local_delay+self.time_res,self.APPROX_INF+self.time_res,step=self.time_res)))-np.exp(-self.dist_local_delay.mean(loc=self.loc_local_delay,scale=self.scale_local_delay))
            dec_error_loss=self.error_loss-error_loss
            self.error_loss=error_loss
            if dec_error_loss>1-np.exp(-mean_interval):
                decision=False
            else:
                self.optimal_learning_achieved=True
                #print(self.observations)
            if (not self.optimal_learning_achieved):
                self.min_amount_observations=self.observations.size+1
                # print(self.min_amount_observations)
        return decision
    
    # Extensions for the Actor-Critic modeling
    def makeJockeyingDecision(self):
        # We make this decision if we have already joined the queue 
        # First we analyse our current state -> which server, server intensity and expected remaining latency
        # Then we get information about the state of the alternative queue 
        # Evaluate input from the actor-critic once we get in the alternative queue
        pass
    

class Observations:
    def __init__(self, reneged=False, jockeyed=False, time_waited=0.0,end_utility=0.0,min_num_observations=0):
        self.reneged=bool(reneged)
        self.jockeyed=bool(jockeyed)
        self.time_waited=float(time_waited)
        #self.end_utility=float(end_utility)
        self.reward=float(reward)
        self.min_num_observations=int(min_num_observations)
        return

class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, arr_rate, utility_basic, discount_coef, markov_model=msm.StateMachine(orig=None),
                 time=0.0, queue=np.array([]), outage_risk=0.1, learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon,
                 para_local_delay=[1.0,2.0,10.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0):
        
        self.markov_model=msm.StateMachine(orig=markov_model)
        self.arr_rate=float(arr_rate)
        self.utility_basic=float(utility_basic)
        self.discount_coef=float(discount_coef)
        self.outage_risk=float(outage_risk)
        self.time=float(time)
        self.init_time=self.time
        self.queue=queue
        self.learning_mode=str(learning_mode)
        self.alt_option=str(alt_option)
        self.min_amount_observations=int(min_amount_observations)
        self.dist_local_delay=dist_local_delay
        self.para_local_delay=list(para_local_delay)
        self.history=np.array([])
        self.decision_rule=str(decision_rule)
        self.truncation_length=float(truncation_length)
        self.preempt_timeout=float(preempt_timeout)
        self.preempt_timer=self.preempt_timeout
        self.time_res=float(time_res)
        self.dict_servers_info = ImpatientTenantEnv.queue_setup_manager()
        self.dict_queues_obj = ImpatientTenantEnv.generate_queues()
        # self.rng_pos_reg=np.array([])
        self.rng_counter=np.array([])
        if self.markov_model.feature!=None:
            self.srv_rate=self.markov_model.feature
        return


    def run(self,duration,progress_bar=True,progress_log=False):
        steps=int(duration/self.time_res)
        #stop_time=self.time+duration
        #while self.time<=stop_time:
        if progress_bar!=None:
            loop=tqdm(range(steps),leave=False,desc='     Current run')
        else:
            loop=range(steps)
        for i in loop:
            if progress_log:
                print("Step",i,"/",steps)
            self.markov_model.updateState()
            self.srv_rate=self.markov_model.feature           
            if (self.queue.size>0 and self.learning_mode=='preemption'):
                self.preempt_timer-=self.time_res
                if self.preempt_timer<=0:
                    self.reqRenege(self.queue[0].id)
                    self.preempt_timer=self.preempt_timeout
            service_intervals=np.random.exponential(1/self.srv_rate,max(int(self.srv_rate*self.time_res*5),2)) # to ensure they exceed one sampling interval
            service_intervals=service_intervals[np.where(np.add.accumulate(service_intervals)<=self.time_res)[0]]
            service_intervals=service_intervals[0:np.min([len(service_intervals),self.queue.size])]
            arrival_intervals=np.random.exponential(1/self.arr_rate, max(int(self.arr_rate*self.time_res*5),2))

            arrival_intervals=arrival_intervals[np.where(np.add.accumulate(arrival_intervals)<=self.time_res)[0]]
            service_entries=np.array([[self.time+i,False] for i in service_intervals]) # False for service
            service_entries=service_entries.reshape(int(service_entries.size/2),2)
            # print(arrival_intervals)
            time.sleep(2)
            arrival_entries=np.array([[self.time+i,True] for i in arrival_intervals]) # True for request
            print(arrival_entries) ####
            time.sleep(2)
            arrival_entries=arrival_entries.reshape(int(arrival_entries.size/2),2)
            # print(arrival_entries)
            time.sleep(2)
            all_entries=np.append(service_entries,arrival_entries,axis=0)
            all_entries=all_entries[np.argsort(all_entries[:,0])]
            print(all_entries) ####
            time.sleep(2)
            self.processEntries(all_entries)
            self.time+=self.time_res
        return

    def processEntries(self,entries=np.array([])):
        for entry in entries:
            # print("Processing a new request entry...")
            #self.time=entry[0]
            if entry[1]==True:
                # print("  Adding a new request into task queue...")
                self.addNewRequest()
            else:
                # print("  Serving a pending request...")
                self.serveOneRequest()
                # print("  Broadcasting the updated queue information...")
                self.broadcastQueueInfo()
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

    def generateLocalCompUtility(self,req):
        #req=Request(req)
        local_delay=req.dist_local_delay.rvs(loc=req.loc_local_delay,scale=req.scale_local_delay)
        delay=float(self.time-req.time_entrance)+local_delay
        return float(req.utility_basic*np.exp(-delay*req.discount_coef))

    def addNewRequest(self):
        # Join the shorter of either queues
               
        req=Request(time_entrance=self.time,pos_in_queue=self.queue.size,utility_basic=self.utility_basic,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res,markov_model=self.markov_model,
                    serv_rate=self.srv_rate,dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay)
        
        lengthQueOne = len(self.dict_queues_obj["Server1"])
        lengthQueTwo = len(self.dict_queues_obj["Server2"])
        
        if lengthQueOne < lengthQueTwo:
            self.dict_queues_obj["Server1"] = np.append(req, self.dict_queues_obj["Server1"])
        else:
            self.dict_queues_obj["Server2"] = np.append(req, self.dict_queues_obj["Server2"])
        
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

    def serveOneRequest(self):
        #randomly select which queue to process at a time t+1
        q_selector = random.randint(1, 2)
        
        if q_selector == 1:                
            queueID = list(self.dict_servers_info.keys())[0]
            req=self.dict_queues_obj.values()[0]
            hstr_entry=HistoryEntry(False,False,self.time-req.time_entrance,self.generateLocalCompUtility(req),req.min_amount_observations)
            self.history=np.append(self.history,hstr_entry)
            # self.queue=self.queue[1:self.queue.size]
            self.dict_queues_obj["Server1"] = np.append(1, self.dict_queues_obj["Server1"].size)
        else:                        
            queueID = list(self.dict_servers_info.keys())[1] 
            req=self.queue[0]
            hstr_entry=HistoryEntry(False,False,self.time-req.time_entrance,self.generateLocalCompUtility(req),req.min_amount_observations)
            self.history=np.append(self.history,hstr_entry)
            #self.queue=self.queue[1:self.queue.size]           
            self.dict_queues_obj["Server2"] = np.append(1, self.dict_queues_obj["Server2"].size)
        
        return

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

    def reqRenege(self,req_id):
        id_queue=np.array([req.id for req in self.queue])
        req=self.queue[np.where(id_queue==req_id)[0][0]]
        # self.updateRngLog(req.pos_in_queue)
        hstr_entry=Observations(True,False,self.time-req.time_entrance,self.generateLocalCompUtility(req),req.min_amount_observations)
        self.history=np.append(self.history,hstr_entry)
        np.delete(self.queue,np.where(id_queue==req_id)[0][0])
        return
    
    def reqJockey(self, req_id):
        id_queue=np.array([req.id for req in self.queue])
        req=self.queue[np.where(id_queue==req_id)[0][0]]
        
        for key in self.dict_queues_obj:
            if key in id_queue:
                dest_queue = self.dict_queues_obj[key]
                        
        np.delete(self.queue,np.where(id_queue==req_id)[0][0])
        
        # ToDo:: After deleting from the current queue,
        #        move to the end of the preferred queue
        #        Record observation from this activity and rewards/penalty
        dest_queue = np.append( dest_queue, req)
        obs_entry=Observations(True,False,self.time-req.time_entrance,self.generateLocalCompUtility(req),req.min_amount_observations)
        self.history=np.append(self.history,obs_entry)
        
        return 
        
