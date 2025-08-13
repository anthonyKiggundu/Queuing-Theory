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
import itertools
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
from termcolor import colored
from collections import Counter
from collections import defaultdict

# --- Add for predictive modeling ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Global variables
GLOBAL_HISTORIES = []
GLOBAL_SIMULATION_HISTORIES_NOPOLICY = []
GLOBAL_SIMULATION_HISTORIES_POLICY = []
GLOBAL_SIMULATION_HISTORIES_POLICY_QUEUE_LEN = []

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
        self.model = LogisticRegression(max_iter=200, class_weight='balanced')
        self.scaler = StandardScaler()
        self.X = []
        self.y = []
        self.is_fitted = False
        self.fit_history = []  # <--- model fit history
        

    def record_observation(self, features, reaction):
        """features: 1D array-like, reaction: str from ACTIONS"""
        if reaction not in self.ACTIONS:
            print(f"[PredictiveModel] Warning: Unknown action '{reaction}'. Valid actions: {self.ACTIONS}")
            return
            
        # print(f"[PredictiveModel] Recording action: {reaction} | Features: {features}")
        self.X.append(features)
        self.y.append(self.ACTION_IDX[reaction])        
            

    def fit(self):
        from collections import Counter
		
        X_arr = np.array(self.X, dtype=np.float64)
        y_arr = np.array(self.y)
        mask = np.all(np.isfinite(X_arr), axis=1)
        if not np.all(mask):
            print(f"Skipping {np.sum(~mask)} samples with NaN/inf in features")
        X_arr = X_arr[mask]
        y_arr = y_arr[mask]
        # Fit if there are at least 8 samples and two classes present
        #if len(X_arr) > 7 and len(set(y_arr)) >= 2:
        
        if len(X_arr) > 3 and len(set(y_arr)) >= 2:
            X_scaled = self.scaler.fit_transform(X_arr)
            self.model.fit(X_scaled, y_arr)
            self.is_fitted = True
            class_counts = Counter(y_arr)
            # print(f"[PredictiveModel] Fitting model with {len(X_arr)} samples. Class counts: {class_counts}")
            # Save diagnostics info for this fit
            self.fit_history.append({
                "n_samples": len(X_arr),
                "classes": dict(class_counts),
                "coef_": self.model.coef_.copy(),
                "intercept_": self.model.intercept_.copy(),
            })
        else:
            print(f"[PredictiveModel] Not enough data to fit: {len(X_arr)} samples, {len(set(y_arr))} classes")
            

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
class ServerPolicyOld:
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
        self.w_renege = -0.5   # Penalty for reneging (bad for server)
        self.w_jockey = -2.0   # Penalty for jockeying (loss of customer)
        self.history = []

    def utility(self, proba):
        """Expected utility given predicted action probabilities."""
        
        t = self.w_wait * proba[0] + self.w_renege * proba[1] + self.w_jockey * proba[2]
        print("\n What is in TTT ", t, self.w_wait * proba[0], self.w_renege * proba[1], self.w_jockey * proba[2])
        return (self.w_wait * proba[0] +
                self.w_renege * proba[1] +
                self.w_jockey * proba[2])

    def update_policy(self, queue_state_features, max_rate=None): # queue_size=None):
        """
        Update service rate to maximize expected utility based on model prediction.
        """
        
        if max_rate is not None:
            self.max_rate = max_rate
            
        proba = self.model.predict_proba(queue_state_features)
        # print("\n --> ", proba)
        util = self.utility(proba)
        #print("="*60)
        #print(f"[ServerPolicy] Features: {queue_state_features}")
        #print(f"[ServerPolicy] Proba: {proba}")
        #print(f"[ServerPolicy] Utility: {util}")
        #print(f"[ServerPolicy] Model Fitted: {getattr(self.model, 'is_fitted', False)}")
        prev_rate = self.current_service_rate
        print(f"[ServerPolicy] Previous Rate: {prev_rate}")                

        # Simple rule: if utility is low (many reneges/jockeys), increase rate; else decrease
        # Only adapt if the model is fitted
        if getattr(self.model, 'is_fitted', False):
            if util < 0:
                new_rate = self.current_service_rate * random.uniform(1,1.2)
                self.current_service_rate = min(new_rate, self.max_rate) # 1.15
                print(f"[ServerPolicy-UTIL <0] Updated Rate: {self.current_service_rate}, {self.current_service_rate * random.uniform(1,1.2)}")
            elif util > 0: # .2
                new_rate = self.current_service_rate * random.uniform(0.5,1.0)
                self.current_service_rate = max(new_rate, self.min_rate) # 0.95
                print(f"[ServerPolicy-UTIL >2] Updated Rate: {self.current_service_rate}, {self.current_service_rate * random.uniform(0.5,1.0)}")
                
            #print(f"[ServerPolicy] Updated Rate: {self.current_service_rate}")  
        else:
            print("[ServerPolicy] Model not fitted, rate unchanged.")

        self.history.append({
            "features": queue_state_features,
            "proba": proba,
            "utility": util,
            "prev_rate": prev_rate,
            "new_rate": self.current_service_rate
        })
        return self.current_service_rate


class ServerPolicy:
    def __init__(self, predictive_model, min_rate=1.0, max_rate=15.0, max_rate_factor=0.5):
        self.model = predictive_model
        self.current_service_rate = min_rate
        self.min_rate = min_rate
        self.max_rate = max_rate        
        self.max_rate_factor = max_rate_factor
        self.w_wait = 1.0
        self.w_renege = -1.0
        self.w_jockey = -0.5
        self.history = []
        
        
    def compute_dynamic_max_rate(self, queue_size):
        # Example: base it on queue size; tweak factor as needed
        return min(max(3.0, queue_size * self.max_rate_factor), 15.0)
        
         
    def utility(self, proba):
        """Expected utility given predicted action probabilities."""
        
        #t = self.w_wait * proba[0] + self.w_renege * proba[1] + self.w_jockey * proba[2]
        #print("\n What is in TTT ", t, self.w_wait * proba[0], self.w_renege * proba[1], self.w_jockey * proba[2])
        return (self.w_wait * proba[0] +
                self.w_renege * proba[1] +
                self.w_jockey * proba[2])

    def adjust_weights(self):
        # Count actions in the predictive model
        from collections import Counter
        y = getattr(self.model, 'y', [])
        action_counts = Counter(y)
        total = sum(action_counts.values())
        if total == 0:
            return  # Not enough data

        # Example logic: if renege is dominant, reduce its penalty
        renege_ratio = action_counts.get(self.model.ACTION_IDX['renege'], 0) / total
        jockey_ratio = action_counts.get(self.model.ACTION_IDX['jockey'], 0) / total
        wait_ratio = action_counts.get(self.model.ACTION_IDX['wait'], 0) / total

        # Dynamically adjust penalties to encourage more "wait"
        # The more renege/jockey, the softer the penalty (less negative)
        self.w_renege = -max(0.2, 1.5 * (1 - renege_ratio))
        self.w_jockey = -max(0.1, 1.0 * (1 - jockey_ratio))
        # Optionally, increase reward for waiting if rare
        if wait_ratio < 0.3:
            self.w_wait = min(2.0, 1.0 + (0.5 - wait_ratio))
        else:
            self.w_wait = 1.0

        # print(f"[ServerPolicy] Adjusted weights: w_wait={self.w_wait:.2f}, w_renege={self.w_renege:.2f}, w_jockey={self.w_jockey:.2f}")

    def update_policy(self, queue_state_features, queue_size=None): # max_rate=None): #
        # Call adjust_weights before each update
            
        self.adjust_weights()
        proba = self.model.predict_proba(queue_state_features)
        util = self.utility(proba)
        
        #print("="*60)
        #print(f"[ServerPolicy] Features: {queue_state_features}")
        #print(f"[ServerPolicy] Proba: {proba}")
        #print(f"[ServerPolicy] Utility: {util}")
        #print(f"[ServerPolicy] Model Fitted: {getattr(self.model, 'is_fitted', False)}")
        prev_rate = self.current_service_rate
        #print(f"[ServerPolicy] Previous Rate: {prev_rate}")
        
        # Dynamically update max_rate if queue_size is provided
        if queue_size is not None:
            self.max_rate = self.compute_dynamic_max_rate(queue_size)
        
        if getattr(self.model, 'is_fitted', False):
            if util < 0:
                new_rate = self.current_service_rate * 1.15
                self.current_service_rate = min(new_rate, self.max_rate)
                
            elif util > 0.2:
                new_rate = self.current_service_rate * 0.95
                self.current_service_rate = max(new_rate, self.min_rate)                
                
        self.history.append({
            "features": queue_state_features,
            "proba": proba,
            "utility": util,
            "prev_rate": prev_rate,
            "new_rate": self.current_service_rate,
            "weights": (self.w_wait, self.w_renege, self.w_jockey)
        })
        
        return self.current_service_rate
        
        
    def get_policy_history(self):
        return self.history


########################################################################
#        Optimal policy for controlling two-server queueing            #
#           systems with jockeying - prototype                         #
########################################################################

'''	
mu1=2.0, mu2=2.0,   # service rates
r1=7.0, r2=7.0,     # admission rewards for routing to server1/2
h1=1.0, h2=1.0,     # holding costs per job per unit time
c1=2.0, c2=2.0,     # service costs (when serving own job)
c12=3.0, c21=3.0,   # jockeying service costs (server1 serves jockey from 2 = c21, etc.)
C_R=5.0,            # reneging cost (penalty) when a job abandons
alpha=0.1):         # discount / uniformization parameter (not directly used)
'''

"""
Two separate MDP solvers:
  1) compute_optimal_jockeying_policy(...)   -- jockeying/routing/service decisions (NO reneging)
  2) compute_optimal_reneging_policy(...)   -- reneging decisions only (routing fixed or provided)

Both use value iteration on truncated grid {0..N_x} x {0..N_y}.
"""

from math import inf


class Params:
    def __init__(self, lam=2.0, mu1=2.0, mu2=2.0,
                 r1=7.0, r2=7.0,
                 h1=1.0, h2=1.0,
                 c1=2.0, c2=2.0,
                 c12=3.0, c21=3.0,
                 C_R=5.0):
        self.lam = lam
        self.mu1 = mu1
        self.mu2 = mu2
        self.r1 = r1
        self.r2 = r2
        self.h1 = h1
        self.h2 = h2
        self.c1 = c1
        self.c2 = c2
        self.c12 = c12
        self.c21 = c21
        self.C_R = C_R
        
        
    def update_from_queue(self, queue):
        """Update rates from a RequestQueue instance."""
        self.lam = queue.arr_rate
        self.mu1 = queue.srvrates_1
        self.mu2 = queue.srvrates_2
        

class CombinedPolicySolver:
    """
    This class combines the logic of JockeyingPolicySolver and RenegingPolicySolver.
    It precomputes both policies and exposes a unified decision API for use in RequestQueue.
    """
    def __init__(self, params, N_x=30, N_y=30, max_iters=2000, tol=1e-6, verbose=False):
        self.params = params
        self.N_x = N_x
        self.N_y = N_y
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose
        self._compute_policies()

    def _compute_policies(self):
        # Jockeying policy
        self.jockeying_policy = self._solve_jockeying_policy()
        # Reneging policy
        self.reneging_policy = self._solve_reneging_policy()

    def _solve_jockeying_policy(self):
        # Same as JockeyingPolicySolver.solve
        lam = self.params.lam
        mu1 = self.params.mu1
        mu2 = self.params.mu2
        r1 = self.params.r1
        r2 = self.params.r2
        h1 = self.params.h1
        h2 = self.params.h2
        c1 = self.params.c1
        c2 = self.params.c2
        c12 = self.params.c12
        c21 = self.params.c21

        Nx = self.N_x
        Ny = self.N_y
        V = np.zeros((Nx+1, Ny+1))
        Vnew = np.zeros_like(V)
        X1 = np.arange(Nx+1)[:,None]
        X2 = np.arange(Ny+1)[None,:]
        hmat = h1*X1 + h2*X2

        def getV(i,j):
            i = min(max(i,0),Nx)
            j = min(max(j,0),Ny)
            return V[i,j]

        def T0_val(x1,x2):
            return max(getV(x1,x2), r1 + getV(x1+1,x2), r2 + getV(x1,x2+1))

        def T1_val(x1,x2):
            if x1==0 and x2==0: return getV(x1,x2)
            cand = [getV(x1,x2)]
            if x1>0: cand.append(getV(x1-1,x2) - c1)
            if x2>0: cand.append(getV(x1,x2-1) - c21)
            return max(cand)

        def T2_val(x1,x2):
            if x1==0 and x2==0: return getV(x1,x2)
            cand = [getV(x1,x2)]
            if x2>0: cand.append(getV(x1,x2-1) - c2)
            if x1>0: cand.append(getV(x1-1,x2) - c12)
            return max(cand)

        if self.verbose: print("Starting jockeying-only VI on grid",Nx,"x",Ny)
        
        it=0
        
        while it < self.max_iters:
            for i in range(Nx+1):
                for j in range(Ny+1):
                    t0 = T0_val(i,j)
                    t1 = T1_val(i,j)
                    t2 = T2_val(i,j)
                    
                    if np.any(np.isnan(Vnew)) or np.any(np.isinf(Vnew)):
                        print("NaN or Inf detected in Vnew at iteration", it)
                        Vnew = np.nan_to_num(Vnew, nan=0.0, posinf=0.0, neginf=0.0)
                        
                    Vnew[i,j] = -hmat[i,j] + lam*t0 + mu1*t1 + mu2*t2
            diff = np.max(np.abs(Vnew-V))
            V[:,:] = Vnew[:,:]
            it +=1
            if self.verbose and (it<5 or it%100==0): print(f"iter {it} diff {diff:.3e}")
            if diff < self.tol: break
        if self.verbose: print("Jockeying VI done at iter",it,"diff",diff)

        # extract argmax policy tables
        a0 = np.zeros_like(V, dtype=int); a1 = np.zeros_like(V, dtype=int); a2 = np.zeros_like(V, dtype=int)
        for i in range(Nx+1):
            for j in range(Ny+1):
                # a0
                vals0 = [getV(i,j), r1+getV(i+1,j), r2+getV(i,j+1)]
                a0[i,j] = int(np.argmax(vals0))
                # a1
                vals1 = [getV(i,j)]
                acts1 = [0]
                
                if i>0: 
                    vals1.append(getV(i-1,j)-c1)
                    acts1.append(1)
                    
                if j>0: 
                    vals1.append(getV(i,j-1)-c21)
                    acts1.append(2)
                    
                a1[i,j] = acts1[int(np.argmax(vals1))]
                # a2
                vals2 = [getV(i,j)]; acts2=[0]
                if j>0: 
                    vals2.append(getV(i,j-1)-c2)
                    acts2.append(1)
                if i>0: 
                    vals2.append(getV(i-1,j)-c12)
                    acts2.append(2)
                    
                a2[i,j] = acts2[int(np.argmax(vals2))]
                
        return dict(a0=a0, a1=a1, a2=a2)


    def _solve_reneging_policy(self):
        lam = self.params.lam
        mu1 = self.params.mu1
        mu2 = self.params.mu2
        h1 = self.params.h1
        h2 = self.params.h2
        C_R = self.params.C_R

        Nx = self.N_x
        Ny = self.N_y
        V = np.zeros((Nx+1,Ny+1))
        Vnew = np.zeros_like(V)
        X1 = np.arange(Nx+1)[:,None]
        X2 = np.arange(Ny+1)[None,:]
        hmat = h1*X1 + h2*X2

        def getV(i,j):
            i=min(max(i,0),Nx)
            j=min(max(j,0),Ny)
            return V[i,j]

        def T0_val(x1,x2):
            # default: route to shorter queue, ties -> S1
            rout = 1 if x1<=x2 else 2
            if rout==1:
                return max(getV(x1,x2), self.params.r1 + getV(x1+1,x2))
            else:
                return max(getV(x1,x2), self.params.r2 + getV(x1,x2+1))

        def T1_val(x1,x2):
            cand = [getV(x1,x2)]
            if x1>0: cand.append(getV(x1-1,x2))  # serve own (no extra cost here)
            if x1>0: cand.append(-C_R + getV(x1-1,x2))  # renege q1
            if x2>0: cand.append(-C_R + getV(x1,x2-1))  # renege q2
            return max(cand)

        def T2_val(x1,x2):
            cand = [getV(x1,x2)]
            if x2>0: cand.append(getV(x1,x2-1))
            if x1>0: cand.append(-C_R + getV(x1-1,x2))
            if x2>0: cand.append(-C_R + getV(x1,x2-1))
            return max(cand)

        if self.verbose: print("Starting reneging-only VI on grid",Nx,"x",Ny)
        it=0
        while it < self.max_iters:
            for i in range(Nx+1):
                for j in range(Ny+1):
                    t0 = T0_val(i,j)
                    t1 = T1_val(i,j)
                    t2 = T2_val(i,j)
                    
                    if np.any(np.isnan(Vnew)) or np.any(np.isinf(Vnew)):
                        print("NaN or Inf detected in Vnew at iteration", it)
                        Vnew = np.nan_to_num(Vnew, nan=0.0, posinf=0.0, neginf=0.0)
                        
                    Vnew[i,j] = -hmat[i,j] + lam*t0 + mu1*t1 + mu2*t2
            diff = np.max(np.abs(Vnew-V))
            V[:,:] = Vnew[:,:]; it+=1
            if self.verbose and (it<5 or it%100==0): print(f"iter {it} diff {diff:.3e}")
            if diff<self.tol: break
        if self.verbose: print("Reneging VI done at iter",it,"diff",diff)

        # extract renege policy tables: at each state, does best action include a renege?
        renege_srv1 = np.zeros_like(V, dtype=bool)
        renege_srv2 = np.zeros_like(V, dtype=bool)
        for i in range(Nx+1):
            for j in range(Ny+1):
                vals = []
                acts = []
                vals.append(getV(i,j)); acts.append("idle")
                if i>0: 
                    vals.append(getV(i-1,j))
                    acts.append("serve")
                if i>0: 
                    vals.append(-C_R + getV(i-1,j))
                    acts.append("renege_q1")
                if j>0: 
                    vals.append(-C_R + getV(i,j-1))
                    acts.append("renege_q2")
                    
                best = acts[int(np.argmax(vals))]
                renege_srv1[i,j] = (best.startswith("renege"))
                vals=[]; acts=[]
                vals.append(getV(i,j)); acts.append("idle")
                if j>0: 
                    vals.append(getV(i,j-1))
                    acts.append("serve")
                if i>0:
                    vals.append(-C_R + getV(i-1,j))
                    acts.append("renege_q1")
                if j>0:
                    vals.append(-C_R + getV(i,j-1))
                    acts.append("renege_q2")
                    
                best2 = acts[int(np.argmax(vals))]
                renege_srv2[i,j] = (best2.startswith("renege"))
                
        return dict(at_srv1=renege_srv1, at_srv2=renege_srv2)
        

    # Unified API for RequestQueue to use:
    def get_jockey_decision(self, queue_idx, x1, x2):
        """
        Returns jockeying action for a server at queue_idx (1 or 2) and state (x1,x2).
        """
        if queue_idx == 1:
            return self.jockeying_policy['a1'][x1, x2]
        elif queue_idx == 2:
            return self.jockeying_policy['a2'][x1, x2]
        else:
            raise ValueError("queue_idx must be 1 or 2")

    def get_renege_decision(self, queue_idx, x1, x2):
        """
        Returns True if reneging is optimal at this state for the given queue_idx.
        """
        if queue_idx == 1:
            return self.reneging_policy['at_srv1'][x1, x2]
        elif queue_idx == 2:
            return self.reneging_policy['at_srv2'][x1, x2]
        else:
            raise ValueError("queue_idx must be 1 or 2")

    def get_routing_decision(self, x1, x2):
        """
        Returns 0=reject, 1=route to S1, 2=route to S2 for new arrivals at state (x1, x2)
        """
        return self.jockeying_policy['a0'][x1, x2]



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
                 customerid="", learning_mode='online',min_amount_observations=1,time_res=1.0,markov_model=msm.StateMachine(orig=None), time_exit=0.0,
                 exp_time_service_end=0.0, serv_rate=1.0, dist_local_delay=stats.expon,para_local_delay=[0.0,0.05,1.0], batchid=0, policy_type=None ):  #markov_model=a2c.A2C, 
        
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
        self.time_exit = time_exit
        self.service_time = service_time # self.objQueues.get_dict_servers()
        #self.certainty=float(outage_risk)
        self.policy_type = policy_type


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
        self.anchor_counter = 0

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
                self.min_amount_observations=self.observations.size+1
                
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
        		
        if  "1" in queue_id: # Server1
            _id_ = "1"
        else:
            _id_ = "2"
			
        self.obs = {
			        "ServerID": _id_, #queue_id,
                    "customerid": req.customerid,
                    "at_pose": curr_pose,
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
                "expected_local_service":time_local_service,
                "Waited": time_to_service_end,
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
                "jockeyed": jockeyed,
                "expected_local_service":time_alt_queue,
                "Waited": time_to_service_end,
                "reward": reward,
                "action":activity,
                "queue_length": queue_length  			
            }
        )
        
    
    def get_jockey_obs(self, queueid, intensity, pose):
		
        return self.curr_obs_jockey                  
           
    
class RequestQueue:

    APPROX_INF = 1000 # an integer for approximating infinite

    def __init__(self, utility_basic, discount_coef, markov_model=msm.StateMachine(orig=None),
                 time=0.0, outage_risk=0.1, customerid="",learning_mode='online', decision_rule='risk_control',
                 alt_option='fixed_revenue', min_amount_observations=1, dist_local_delay=stats.expon, time_exit=0.0, exp_time_service_end=0.0,
                 para_local_delay=[0.01,0.1,1.0], truncation_length=np.Inf, preempt_timeout=np.Inf, time_res=1.0, 
                 batchid=np.int16, policy_enabled=True, use_e_greedy=False, params=None):
                 
        
        self.dispatch_data = {}
        self.params = params
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
        self.time_exit = time_exit      
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
        self.raw_srvrates_1 = 1.0
        self.raw_srvrates_2 = 1.0                          
        
        BROADCAST_INTERVAL = 5
        self.all_requests = []  
        self.anchor_counter = 0  # For round-robin anchor selection
        #self.anchor_counts = Counter()   # equal distribution    
        
        self.policy_enabled = policy_enabled
        self.predictive_model = PredictiveModel()
        self.policy = ServerPolicy(self.predictive_model, min_rate=1.0, max_rate=12.0)
        
        #if not hasattr(self, 'simulation_start_time'):
        self.simulation_start_time = self.time # time.time()
        self.use_e_greedy = use_e_greedy
        # self.policy_type = policy_type          
                
        return      
    
    
    #def decide_policy(self, *args, **kwargs):
    #    if self.policy_type == "egreedy":
    #        # Use e-greedy policy logic
    #        # For example:
    #        queue_idx, mode = self.egreedy_router.select_queue()
    #        # (Do something with queue_idx or mode)
    #        return queue_idx, mode
    #    elif self.policy_type == "rule":
    #        # Use rule-based policy logic
    #        # For example:
    #        next_queue = self.rule_based_policy.choose_queue(*args, **kwargs)
    #        return next_queue
    #    else:
    #        raise ValueError(f"Unknown policy_type: {self.policy_type}")
            
    
    #def assign_policy_type(self):
    #    """
    #    Randomly assign a policy type per request (A/B test).
    #    """
    #    return random.choice(self.policy_types)
            
       
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
		
        return self.srvrates_1 if queue_id == "1" else self.srvrates_2


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

                alt_queue_id = "2" if "1" in queue_id else "1"
                alt_queue = self.dict_queues_obj[alt_queue_id]         
                            
                self.dispatch_queue_state( curr_queue, queue_id, alt_queue, alt_queue_id, interval) #, curr_queue_state)
                                   
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
        curr_queue_state = self.get_queue_state(queue_id, service_rate)
        reneging_rate = curr_queue_state["reneging_rate"]
        jockeying_rate = curr_queue_state["jockeying_rate"]
        sample_interchange_time = curr_queue_state["markov_model_inter_change_time"]
        steady_state_distribution = curr_queue_state["markov_model_service_rate"]
        
        features = [
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
         
        features = np.array(features, dtype=np.float64)
        
        # Replace any inf/-inf with 0.0
        if not np.all(np.isfinite(features)):
            print(f"[extract_features] Warning: Non-finite feature(s) detected for queue {queue_id}: {features}")
            features[~np.isfinite(features)] = 0.0

        return features


    # After each dispatch or user action, record observation and update model/policy
    def record_user_reaction(self, queue_id, action_label):
        """
        Call this after observing a user action (wait, renege, jockey).
        """
        features = self.extract_features(queue_id)
        self.predictive_model.record_observation(features, action_label)
        self.predictive_model.fit()
        queue_size = len(self.dict_queues_obj.get(queue_id, []))
        # Optionally: immediately update policy based on new prediction
        self.policy.update_policy(features, queue_size)


    def run(self,duration, interval, progress_bar=True,progress_log=False):
		
        steps=int(duration/self.time_res)
        
        self.srvrates1_history = []
        self.srvrates2_history = []

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
                
                deltaLambda=random.randint(1, 2)                
        
                serv_rate_one = self.arr_rate + deltaLambda 
                serv_rate_two = self.arr_rate - deltaLambda

                self.srvrates_1 = serv_rate_one / 2                
                self.srvrates_2 = serv_rate_two / 2                
             
                srv_1 = self.dict_queues_obj.get("1") # Server1
                srv_2 = self.dict_queues_obj.get("2") 
                print("\n Arrival rate: ", self.arr_rate, "Rates 1: ----", self.srvrates_1,  "Rates 2: ----", self.srvrates_2)
                
                self.raw_srvrates_1 = serv_rate_one / 2
                self.raw_srvrates_2 = serv_rate_two / 2
                
                if self.params is not None:
                    self.params.update_from_queue(self)
                    
                print("\n PARAMOS UPDATOS: ", self.params.update_from_queue(self))
                                 
                # --- Predictive modeling: adjust service rates using learned policy ---
                
                features_srv1 = self.extract_features("1")
                features_srv2 = self.extract_features("2")
                
                queue_size_1 = len(self.dict_queues_obj.get("1", []))
                queue_size_2 = len(self.dict_queues_obj.get("2", []))
                
                if self.policy_enabled:
                    self.srvrates_1 = self.policy.update_policy(features_srv1, queue_size=queue_size_1)
                    self.srvrates_2 = self.policy.update_policy(features_srv2, queue_size=queue_size_2)
                    #self.srvrates_1 = self.policy.update_policy(features_srv1, max_rate=self.arr_rate)
                    #self.srvrates_2 = self.policy.update_policy(features_srv2, max_rate=self.arr_rate)
                else:
                    self.srvrates_1 = self.raw_srvrates_1
                    self.srvrates_2 = self.raw_srvrates_2                                  
               
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
                self.processEntries(all_entries, i, interval) #, duration)
                self.time+=self.time_res                           
            
                self.set_batch_id(i)
                                
                GLOBAL_HISTORIES.append(list(self.get_history())) 
                
                self.srvrates1_history.append(self.srvrates_1)
                self.srvrates2_history.append(self.srvrates_2)
                
                
            #print("\n Length in Run: ", len(GLOBAL_HISTORIES))    
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        finally:
            self.running = False  # Stop the dispatch thread
            dispatch_thread.join()
            print("Simulation completed.")    
        
        #GLOBAL_SIMULATION_HISTORIES.append(list(self.history))  # Save full history for this run
                    
        return
    

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
        
        for entry in entries:          
            if entry[1]==True:              
                req = self.addNewRequest(entry[0], batchid)
                self.arr_prev_times = np.append(self.arr_prev_times, entry[0])
                
            else:               
                q_selector = random.randint(1, 2)
                if q_selector == 1:					
                    self.queueID = "1" # Server1                    
                    curr_queue_len = len(self.dict_queues_obj[self.queueID])
                    if  curr_queue_len > 0:       
                        self.serveOneRequest(self.queueID, interval) # Server1 = self.dict_queues_obj["1"][0], entry[0],                                                                                      
                        time.sleep(random.uniform(0.01, 0.05))  # Random delay between 0.1 and 0.5 seconds                        
                        
                else:					
                    self.queueID = "2" 
                    curr_queue_len = len(self.dict_queues_obj[self.queueID])
                    if  curr_queue_len > 0:                    
                        self.serveOneRequest(self.queueID, interval) 
                        time.sleep(random.uniform(0.01, 0.05))
                                        
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
            
        policy_type = self.assign_policy_type() 
          
        req=Request(time_entrance=time_entered, pos_in_queue=pose, utility_basic=self.utility_basic, service_time=expected_time_to_service_end,
                    discount_coef=self.discount_coef,outage_risk=self.outage_risk,customerid=self.customerid, learning_mode=self.learning_mode,
                    min_amount_observations=self.min_amount_observations,time_res=self.time_res, #exp_time_service_end=expected_time_to_service_end, 
                    dist_local_delay=self.dist_local_delay,para_local_delay=self.para_local_delay, batchid=batchid, policy_type=policy_type) # =self.batchid
         
        self.all_requests.append(req)           
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
		
        if "1" in queue_id:
            #curr_queue_id = "1"
            serv_rate = self.raw_srvrates_1 # self.get_service_rates(queue_id) # rate_srv1
        else:
            #curr_queue_id = "2"
            serv_rate = self.raw_srvrates_2 # self.get_service_rates(queue_id) # rate_srv2
        
        curr_queue_state = self.get_queue_state(alt_queue_id, serv_rate)

        # Compute reneging rate and jockeying rate
        reneging_rate = self.compute_reneging_rate(curr_queue)
        jockeying_rate = self.compute_jockeying_rate(curr_queue)
        num_requests = curr_queue_state['total_customers'] # len(curr_queue) 
         
        anchor = random.choice(["markov_model_inter_change_time", "markov_model_service_rate"])
        #print("\n Entering loop now ....")      
        for client in range(len(curr_queue)):
            #print("\n ANCHOR Now :=> ", anchor )
            req = curr_queue[client] 
            self.makeJockeyingDecision(req, queue_id, alt_queue_id, req.customerid, serv_rate, anchor)
            self.makeRenegingDecision(req, alt_queue_id, req.customerid, anchor)
            
            # print(f"Dispatching state of server {alt_queue_id} to client {req.customerid} : {curr_queue_state}.")
            
            #if "1" in alt_queue_id: # == "1":
            #    self.makeJockeyingDecision(req, alt_queue_id, queue_id, req.customerid, serv_rate)
            #    self.makeRenegingDecision(req, alt_queue_id, req.customerid)
            #    alt_queue_id = str(alt_queue_id) # "Server_"+
            #else:
            #elif "" in alt_queue_id:
            #    self.makeJockeyingDecision(req, alt_queue_id, queue_id, req.customerid, serv_rate)
            #    self.makeRenegingDecision(req, alt_queue_id, req.customerid)
            #    alt_queue_id = str(alt_queue_id) # "Server_"+
        
        # Append rates to interval-specific dispatch data
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["intervals"].append(interval)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["num_requests"].append(num_requests)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["jockeying_rate"].append(jockeying_rate)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["reneging_rate"].append(reneging_rate)
        self.dispatch_data[interval][f"server_{alt_queue_id}"]["service_rate"].append(serv_rate)  
        
    
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
            

    def run_scheduler(self): # , duration=None
        """
        Run the scheduler to dispatch queue status at different intervals.
        """
        self.setup_dispatch_intervals()
        start_time = time.time()
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    
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
    
                       
    def get_queue_state(self, queueid, srv_rate):
		
        #serv_rates = [self.srvrates_1, self.srvrates_2]
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
            #queue_intensity = self.objQueues.get_arrivals_rates()/ rate_srv1    
            customers_in_queue = self.dict_queues_obj["1"] 
            curr_queue = self.dict_queues_obj[queueid]  
            reneging_rate = self.compute_reneging_rate(curr_queue)
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
            # Instantiate MarkovQueueModel          
            #markov_model = MarkovQueueModel(self.arr_rate, self.srvrates_1, max_states=len(customers_in_queue)) # 1000)
            markov_model = MarkovQueueModel(self.arr_rate, self.raw_srvrates_1, max_states=len(customers_in_queue)) # 1000)
            
            # servmarkov_model = MarkovModulatedServiceModel([self.srvrates_1,self.srvrates_2], trans_matrix) # [srvrate1,srvrate2], trans_matrix)
            servmarkov_model = MarkovModulatedServiceModel([self.raw_srvrates_1, self.raw_srvrates_2], trans_matrix) # [srvrate1,srvrate2], trans_matrix)           
            # sample_interchange_time = markov_model.compute_expected_time_between_changes(self.arr_rate, self.srvrates_1, N=100)
            sample_interchange_time = markov_model.compute_expected_time_between_changes(self.arr_rate, self.raw_srvrates_1,  N=100)
            arr_rate1, arr_rate2 = servmarkov_model.arrival_rates_divisor(self.arr_rate, self.raw_srvrates_1, self.raw_srvrates_2) 
            steady_state_distribution = servmarkov_model.best_queue_delay(arr_rate1, self.raw_srvrates_1, arr_rate2, self.raw_srvrates_2)                               
      
        elif "2" in queueid:
       
            customers_in_queue = self.dict_queues_obj["2"]
            curr_queue = self.dict_queues_obj[queueid]
            reneging_rate = self.compute_reneging_rate(curr_queue)
            jockeying_rate = self.compute_jockeying_rate(curr_queue)
    
            # markov_model = MarkovQueueModel(self.arr_rate, self.srvrates_2, max_states=len(customers_in_queue)) 
            markov_model = MarkovQueueModel(self.arr_rate, self.raw_srvrates_2, max_states=len(customers_in_queue)) 
            servmarkov_model = MarkovModulatedServiceModel([self.raw_srvrates_1, self.raw_srvrates_2], trans_matrix)  #([srvrate1,srvrate2], trans_matrix)
            # servmarkov_model = MarkovModulatedServiceModel([self.raw_srvrates_1, self.raw_srvrates_2], trans_matrix)  #([srvrate1,srvrate2], trans_matrix)
            sample_interchange_time = markov_model.compute_expected_time_between_changes(self.arr_rate, self.raw_srvrates_2, N=100)
            
            arr_rate1, arr_rate2 = servmarkov_model.arrival_rates_divisor(self.arr_rate, self.raw_srvrates_1, self.raw_srvrates_2) #_steady_state_distribution()
            steady_state_distribution = servmarkov_model.best_queue_delay(arr_rate1, self.raw_srvrates_1, arr_rate2, self.raw_srvrates_2)            
		       
        curr_queue_state = {
            "total_customers": len(customers_in_queue),            
            "capacity": self.capacity,              
            "long_avg_serv_time": self.get_long_run_avg_service_time(queueid),
            "markov_model_inter_change_time": sample_interchange_time,            
            "markov_model_service_rate": steady_state_distribution, #{              
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


        #else:
        elif "2" in queueID:                       
            req = self.dict_queues_obj["2"][0] # Server2
            serv_rate = serv_rate = self.get_service_rates(queueID) #self.dict_servers_info["2"] # Server2
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
                                                    
                                    
            self.arr_prev_times = self.arr_prev_times[1:self.arr_prev_times.size]  
            self.objQueues.update_queue_status(queueID)

        
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
          
    
    
    def makeRenegingDecision(self, req, queueid, customer_id, anchor, t_max=10.0, num_points=1000):	 #  T_local,                     	

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
            # renege = (t_change_1 >= T_local) and (t_change_2 >= T_local)
            
            return  t_change_1, t_change_2 # renege,

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
            alt_queue_state = self.get_queue_state(alt_queue_id, serv_rate)
            alt_interchange_time = alt_queue_state["markov_model_inter_change_time"] 
            # T_local = stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=0.75/req.pos_in_queue) # queue_interchange_time
            #self. max_cloud_delay = self.calculate_max_cloud_delay(req.pos_in_queue, queue_intensity, req)  
            #self.max_cloud_delay=stats.erlang.ppf(self.certainty, loc=0, scale=curr_arriv_rate, a=req.pos_in_queue)
            T_local = self.generateLocalCompUtility(req)
            
            # use Little's law to compute expected wait in alternative queue
            # instead of Little's Law, use the expected waiting time in steady state using (1 - queue_intensity e^{-(serv_rate - curr_arriv_rate)})
            #expected_wait_in_alt_queue = 1 - queue_intensity * math.exp(-(serv_rate - curr_arriv_rate)) # float(len(self.dict_queues_obj["2"])/curr_arriv_rate)
            #T_queue = expected_wait_in_alt_queue # queue_interchange_time + expected_wait_in_alt_queue
        #else:
        elif "2" in queueid:
            serv_rate = self.get_service_rates(queueid) # self.srvrates_2 # self.dict_servers_info["2"] 
            queue =  self.dict_queues_obj["2"]
            alt_queue_id = "1"
            curr_arriv_rate = self.objQueues.get_arrivals_rates()
            #queue_intensity = curr_arriv_rate/ serv_rate    
            alt_queue_state = self.get_queue_state(alt_queue_id, serv_rate)  
            alt_interchange_time = alt_queue_state["markov_model_inter_change_time"]  
            # T_local = stats.erlang.ppf(self.certainty,a=req.pos_in_queue,loc=0,scale=0.75/req.pos_in_queue)  
            T_local = self.generateLocalCompUtility(req)                                                         
        
        #options = ["steady_state_distribution", "inter_change_time"]
        #anchor = options[self.anchor_counter % 2]
        #self.anchor_counter += 1
        '''
        options = ["steady_state_distribution", "inter_change_time"]

        # Choose the anchor with fewer selections so far; if equal, pick randomly
        counts = self.anchor_counts
        if counts[options[0]] > counts[options[1]]:
            anchor = options[1]
        elif counts[options[0]] < counts[options[1]]:
            anchor = options[0]
        else:    
            anchor = random.choice(options)

        self.anchor_counts[anchor] += 1
        '''
        
        #anchor = random.choice(["steady_state_distribution", "inter_change_time"])
        #print("\n ** ANCHOR IN RENEGE: ** ", anchor )
        
        mu1 = self.get_service_rates("1") # self.srvrates_1 # self.dict_servers_info["1"]
        mu2 = self.get_service_rates("2") # self.srvrates_2 # self.dict_servers_info["2"]
        
        # print(f"Anchor in makeRenegeDecison: {anchor}")
            
        if "markov_model_service_rate" in anchor:
            
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
                            serv_rate = self.srvrates_1
                        else:
                            serv_rate = self.srvrates_2
                        # Compute remaining waiting time (number of requests ahead * average service time)
                        # For M/M/1, expected remaining time = position * 1/service_rate
                        remaining_wait_time = pos * (1.0 / serv_rate) if serv_rate > 0 else 1e4  # avoid zero division

                        # If remaining wait exceeds T_local, renege
                        renege = (remaining_wait_time > T_local)
                        #print("\n I took the remaining time -> ", renege)
                        if renege:
                            decision = True
                            self.reqRenege(req_in_queue, queueid, pos, serv_rate, T_local, req_in_queue.customerid, req_in_queue.service_time, decision, queue, anchor)
                            found = True
                            break

                if not found:
                    #print(f"Request ID {customer_id} not found in queue {queueid}. Continuing with processing...")
                    return False
           
            
            # For high throughput as objective, a low interchange_time shows a better state, if stability is the objective, a high value is better                       
        #elif anchor == "inter_change_time":
        elif "markov_model_inter_change_time" in anchor:

            # Compute values
            #rate = compute_rate_of_change(lambda_, mu)
            #time_between_changes = compute_expected_time_between_changes(lambda_, mu)
            lambda1, lambda2 = arrival_rates_divisor(curr_arriv_rate, mu1, mu2)
            
            t_change_1 , t_change_2 = should_renege_using_tchange(lambda1, mu1, lambda2, mu2, T_local, len(queue)) # renege, 

            #print(f"Rate of queue length change: {rate:.4f} events/sec")
            #print(f"Expected time between changes in Queue 1: {t_change_1:.4f} sec and Expected time between changes in Queue 2: {t_change_2:.4f} sec")
            found = False                       
					
            for pos, req_in_queue in enumerate(queue):
                                
                if customer_id in req_in_queue.customerid:
                    #renege, t_change_1 , t_change_2 = should_renege_using_tchange(lambda1, mu1, lambda2, mu2, T_local, len(queue))
                    # Get the current service rate for the queue
                    # This was working before 
                    
                    if "1" in queueid:
                        serv_rate = mu1 # self.get_service_rates(queueid) # self.srvrates_1  # self.dict_servers_info["1"] get_service_rate
                        remaining_wait_time = pos * t_change_1 
                    #else:
                    elif "2" in queueid:
                        serv_rate = mu2 # self.get_service_rates(queueid) # self.srvrates_2  # self.dict_servers_info["2"]
                        remaining_wait_time = pos * t_change_2 
                    # Compute remaining waiting time (number of requests ahead * average service time)
                    # For M/M/1, expected remaining time = position * 1/service_rate
                    # remaining_wait_time = pos * (1.0 / serv_rate) if serv_rate > 0 else 1e4  # avoid zero division

                    # If remaining wait exceeds T_local, renege
                    renege = (remaining_wait_time > T_local)
                    
            
                    if renege: #alt_queue_state["sample_interchange_time"] > curr_queue_state["sample_interchange_time"]:
                        decision = True 
                        self.reqRenege( req, queueid, pos, serv_rate, T_local, req.customerid, req.service_time, decision, queue, anchor)
                
            if not found:
                #print(f"Request ID {customer_id} not found in queue {queueid}. Continuing with processing...")
                return False


    def reqRenege(self, req, queueid, curr_pose, serv_rate, time_local_service, customerid, time_to_service_end, decision, curr_queue, anchor):
        
        if decision:
            self.record_user_reaction(queueid, "renege")
            
        if "1" in queueid:
            self.queue = self.dict_queues_obj["1"]  
            queue_intensity = self.arr_rate/self.get_service_rates(queueid)          
        else:
            self.queue = self.dict_queues_obj["2"] 
            queue_intensity = self.arr_rate/self.get_service_rates(queueid)
            
        if curr_pose >= len(curr_queue):
            return
            
        else:
            # print("\n Error: ** ", len(self.queue), curr_pose)
            if len(self.queue) > curr_pose:
                self.queue = np.delete(self.queue, curr_pose) # index)  
                self.log_request(req.time_entrance, "reneged", req.time_res) # , self.queue) # req.queue)    arrival_time=    outcome= exit_time= queue=
                #print("\n ********* ", request_log[0])
                self.queueID = queueid  
        
                #req.customerid = req.customerid+"_reneged"
                if not req.customerid.endswith("_reneged"):
                    req.customerid = req.customerid+"_reneged"
                req.reneged = True
                req.time_exit = self.time  # or the relevant exit time
        
                # In the case of reneging, you only get a reward if the time.entrance plus
                # the current time minus the time _to_service_end is greater than the time_local_service
        
                reward = self.getRenegeRewardPenalty(req, time_local_service, time_to_service_end) 
                print(colored("%s", 'red') %(anchor) + ":" + colored("%s", 'green') % (req.customerid) + " in Server %s" %(queueid) + " reneging now to Local")                                   
                
                self.objObserv.set_obs (queueid,  serv_rate, queue_intensity, time_to_service_end, "reneged", reward, curr_pose, req, len(self.queue))
                self.history.append(self.objObserv.get_obs())
                # print("History event in Reneging:", self.history[-1])
                self.objObserv.set_renege_obs(curr_pose, decision,time_local_service, time_to_service_end, reward, queueid, "reneged", len(self.queue))
        
                self.curr_obs_renege.append(self.objObserv.get_renege_obs(queueid, self.queue)) #queueid, queue_intensity, curr_pose))        
                # self.history.append(self.objObserv.get_renege_obs(queueid, self.queue))
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
     
            
    def reqJockey(self, curr_queue_id, dest_queue_id, req, customerid, serv_rate, dest_queue, exp_delay, decision, curr_pose, curr_queue, anchor):		        
        
        # Do not allow jockeying for requests that have already reneged
        if '_reneged' in req.customerid:
            return  # Ignore this request
            
        if decision:
            self.record_user_reaction(curr_queue_id, "jockey")
                
        if curr_pose >= len(curr_queue):
            return
            
        else:	
            np.delete(curr_queue, curr_pose) # np.where(id_queue==req_id)[0][0])
            self.log_request(req.time_entrance, "jockeyed", req.time_res ) #, curr_queue) # req.queue) arrival_time= outcome= exit_time= queue=
            #print("\n ********* ", request_log[0])
            reward = 1.0
            req.time_entrance = self.time # timer()
            dest_queue = np.append( dest_queue, req)
        
            self.queueID = curr_queue_id        
        
            # req.customerid = req.customerid+"_jockeyed"
            if not req.customerid.endswith("_jockeyed"):
                req.customerid = req.customerid+"_jockeyed"
            req.jockeyed = True
            req.time_exit = self.time  # or the relevant exit time
        
            if "1" in curr_queue_id: # Server1
                queue_intensity = self.arr_rate/self.get_service_rates(curr_queue_id) # self.dict_servers_info["1"] # Server1
                self.queue = self.dict_queues_obj["1"]
            
            else:
                queue_intensity = self.arr_rate/self.get_service_rates(curr_queue_id) # self.dict_servers_info["2"] # Server2
                self.queue = self.dict_queues_obj["2"]
        
            reward = self.get_jockey_reward(req)
                  
            # print("\n Moving ", customerid," from Server ",curr_queue_id, " to Server ", dest_queue_id ) 
            print(colored("%s", 'red') %(anchor) + ":" + colored("%s", 'green') % (req.customerid) + " in Server %s" %(curr_queue_id) + " jockeying now, to Server %s" % (colored(dest_queue_id,'green')))  
                                
            self.objObserv.set_obs (curr_queue_id,  serv_rate, queue_intensity, exp_delay, "jockeyed", reward, curr_pose, req, len(self.queue))
            
            self.history.append(self.objObserv.get_obs()) # curr_queue_id, queue_intensity, curr_pose))
            # print("History event in Jockeying:", self.history[-1])
            self.objObserv.set_jockey_obs(curr_pose,  decision, exp_delay, req.exp_time_service_end, reward, 1.0, "jockeyed", len(self.queue)) # time_alt_queue        
            #self.curr_obs_jockey.append(self.objObserv.get_jockey_obs(curr_queue_id, queue_intensity, curr_pose))
            #self.history.append(self.objObserv.get_jockey_obs(curr_queue_id, queue_intensity, curr_pose))                                      
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
    
    
    def makeJockeyingDecision(self, req, curr_queue_id, alt_queue_id, customerid, serv_rate, anchor):
        # We make this decision if we have already joined the queue
        # First we analyse our current state -> which server, server intensity and expected remaining latency
        # Then we get information about the state of the alternative queue
        # Evaluate input from the actor-critic once we get in the alternative queue
        
        if getattr(req, "policy_type", "rule") == "egreedy":
            # --- e-greedy renege logic here (call your e-greedy policy logic) ---
            pass
        elif getattr(req, "policy_type", "rule") == "rule":
            # --- rule-based renege logic here ---
            pass
            

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
                	
        # print("\n Rates: ", self.srvrates_1,self.srvrates_2 )
        if "1" in curr_queue_id:
            self.queue = self.dict_queues_obj["1"]  # Server1     
            serv_rate = self.get_service_rates(curr_queue_id) #self.dict_servers_info["1"]          get_service_rate    
            alt_queue_id = "2"   
            curr_arriv_rate = self.objQueues.get_arrivals_rates()  
            curr_queue_state = self.get_queue_state("1", serv_rate)
            alt_queue_state = self.get_queue_state("2", serv_rate)    
        #else:
        elif "2" in curr_queue_id:
            self.queue = self.dict_queues_obj["2"]
            serv_rate = self.get_service_rates(curr_queue_id) # self.dict_servers_info["2"]              
            alt_queue_id = "1"   
            curr_arriv_rate = self.objQueues.get_arrivals_rates() 
            curr_queue_state = self.get_queue_state("2", serv_rate)
            alt_queue_state = self.get_queue_state("1", serv_rate)

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
            
            '''
               Observe the state of the current queue and compare that with the state of the
               other queue and jockey if the other queue is better than the current one.
               The better state is defined by first-order stochatsic dorminance and the jockeying rate (orprobability)
            ''' 
                        
            #queue_states_compared = self.compare_queues(alt_steady_state, curr_steady_state, K=1) 
            
            # options = random.choice(["steady_state_distribution", "inter_change_time"])
            #anchor = options[self.anchor_counter % 2]
            #self.anchor_counter += 1                       
            
            #print(f"Anchor in makeJockeyDecison: {anchor}")
            if "markov_model_service_rate" in anchor: 
                # Use steady-state distribution comparison (as currently implemented)
                #queue_states_compared = mmc_wait_cdf_and_tail(curr_arriv_rate, serv_rate, c, t): #self.compare_queues(alt_steady_state, curr_steady_state, K=1)
                lambda1, lambda2 = arrival_rates_divisor(curr_arriv_rate, self.get_service_rates("1"), self.get_service_rates("2")) #"1"], self.dict_servers_info["2"])
                # current_queue, lambda1, mu1, lambda2, mu2
                jockey_flag = should_jockey_flag(curr_queue_id, lambda1, self.get_service_rates("1"), lambda2, self.get_service_rates("2"))
                
                if "1" in curr_queue_id:
                    serv_rate = self.srvrates_1
                else:
                    serv_rate = self.srvrates_2
					
                if jockey_flag: #queue_states_compared['first_order_dominance']:
                    decision = True
                    self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.dict_queues_obj.get(curr_queue_id), anchor)
            
            # For high throughput as objective, a low interchange_time shows a better state, if stability is the objective, a high value is better                       
            #elif anchor == "inter_change_time":
            elif  "markov_model_inter_change_time" in anchor:
				
                if "1" in curr_queue_id:
                    serv_rate = self.srvrates_1
                else:
                    serv_rate = self.srvrates_2
					
                if alt_queue_state["markov_model_inter_change_time"] > curr_queue_state["markov_model_inter_change_time"]:
                    decision = True 
                    self.reqJockey(curr_queue_id, alt_queue_id, req, req.customerid, serv_rate, dest_queue, self.avg_delay, decision, curr_pose, self.dict_queues_obj.get(curr_queue_id), anchor)
  
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
                    

    def compute_steady_state_probs(self, rho, N=100):
        """Compute steady-state probabilities for M/M/1 with truncation at N."""
        return np.array([(1 - rho) * rho**n for n in range(N + 1)])

    def compute_rate_of_change(self, lambda_, mu, N=100):
        """Compute average rate at which queue length changes."""
        if mu == 0:
            # No service possible: no queue-length decrease, only arrivals
            return 0.0  # or handle as appropriate for your model
            
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
    

import pandas as pd
import seaborn as sns

def plot_boxplot_waiting_times_by_outcome_2x2(histories_nopolicy, histories_policy):
    """
    2x2 subplot: rows = [No Policy, Policy], cols = [anchor1, anchor2]
    Each panel: boxplot of waiting times by outcome (served/jockeyed/reneged)
    Accepts histories_nopolicy and histories_policy as produced by run_simulations_with_results.
    """
    # Helper to flatten history into DataFrame
    def flatten(histories, policy_label):
        data = []
        for h in histories:
            anchor = h.get('anchor')
            for obs in h.get('history', []):
                waited = obs.get('Waited')
                cid = obs.get('customerid', '')
                if waited is not None:
                    if '_reneged' in cid:
                        outcome = 'reneged'
                    elif '_jockeyed' in cid:
                        outcome = 'jockeyed'
                    else:
                        outcome = 'served'
                    data.append({
                        'Policy': policy_label,
                        'Anchor': anchor,
                        'Outcome': outcome,
                        'Waiting Time': waited
                    })
        return pd.DataFrame(data)

    df_nopolicy = flatten(histories_nopolicy, "No Policy")
    df_policy   = flatten(histories_policy, "Policy")
    df = pd.concat([df_nopolicy, df_policy], ignore_index=True)

    anchors = sorted(df['Anchor'].dropna().unique())
    policies = ["No Policy", "Policy"]
    palette = {
        'served': 'green',
        'jockeyed': 'blue',
        'reneged': 'red'
    }
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for i, policy in enumerate(policies):
        for j, anchor in enumerate(anchors):
            ax = axs[i, j]
            subset = df[(df['Policy'] == policy) & (df['Anchor'] == anchor)]
            # Ensure all outcomes are shown even if some are missing
            sns.boxplot(
                data=subset,
                x='Outcome',
                y='Waiting Time',
                ax=ax,
                showmeans=True,
                meanprops={"marker":"o","markerfacecolor":"black", "markeredgecolor":"black"},
                order=['served', 'jockeyed', 'reneged'],
                palette=palette #'pastel'
            )
            ax.set_title(f"{policy} | Anchor: {anchor}")
            ax.set_xlabel("Outcome")
            if j == 0:
                ax.set_ylabel("Waiting Time")
            else:
                ax.set_ylabel("")
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    # plt.suptitle("Waiting Time by Outcome, Policy, and Anchor", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    

def plot_boxplot_waiting_times_by_outcome_by_interval_and_anchor(histories_nopolicy, histories_policy):
    """
    For each interval and each anchor, creates a separate plot (2 panels: No Policy, Policy).
    Each panel: boxplot of waiting times by outcome (served/jockeyed/reneged).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Helper to flatten history into DataFrame
    def flatten(histories, policy_label):
        data = []
        for h in histories:
            anchor = h.get('anchor')
            interval = h.get('interval')
            for obs in h.get('history', []):
                waited = obs.get('Waited')
                cid = obs.get('customerid', '')
                if waited is not None:
                    if '_reneged' in cid:
                        outcome = 'reneged'
                    elif '_jockeyed' in cid:
                        outcome = 'jockeyed'
                    else:
                        outcome = 'served'
                    data.append({
                        'Policy': policy_label,
                        'Anchor': anchor,
                        'Interval': interval,
                        'Outcome': outcome,
                        'Waiting Time': waited
                    })
        return pd.DataFrame(data)

    df_nopolicy = flatten(histories_nopolicy, "No Policy")
    df_policy   = flatten(histories_policy, "Policy")
    df = pd.concat([df_nopolicy, df_policy], ignore_index=True)

    anchors = sorted(df['Anchor'].dropna().unique())
    intervals = sorted(df['Interval'].dropna().unique())
    policies = ["No Policy", "Policy"]
    palette = {
        'served': 'green',
        'jockeyed': 'blue',
        'reneged': 'red'
    }

    for interval in intervals:
        for anchor in anchors:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            for i, policy in enumerate(policies):
                ax = axs[i]
                subset = df[(df['Interval'] == interval) & (df['Anchor'] == anchor) & (df['Policy'] == policy)]
                sns.boxplot(
                    data=subset,
                    x='Outcome',
                    y='Waiting Time',
                    ax=ax,
                    showmeans=True,
                    meanprops={"marker":"o","markerfacecolor":"black", "markeredgecolor":"black"},
                    order=['served', 'jockeyed', 'reneged'],
                    palette=palette
                )
                ax.set_title(f"{policy}")
                ax.set_xlabel("Outcome")
                if i == 0:
                    ax.set_ylabel("Waiting Time")
                else:
                    ax.set_ylabel("")
                ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            fig.suptitle(f"Waiting Time by Outcome\nInterval: {interval}, Anchor: {anchor}", fontsize=15)
            plt.tight_layout(rect=[0, 0.03, 1, 0.92])
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


def plot_six_panels_with_service_rates(results, intervals, jockey_anchors):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, len(intervals), figsize=(6*len(intervals), 8), sharex=False)
    colors = {
        "markov_model_service_rate": "blue",
        "markov_model_inter_change_time": "green"
    }
    linestyles = {
        "markov_model_service_rate": "-",
        "markov_model_inter_change_time": "--"
    }

    # Accumulate handles and labels for one bottom legend
    all_handles = []
    all_labels = []

    for col, interval in enumerate(intervals):
        # Reneging rate plots
        ax_ren = axs[0, col]
        ax2_ren = ax_ren.twinx()
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                # Reneging Rates
                y = np.array(results[interval]["reneging_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_ren.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)

                # Service Rates (if present)
                srv_key = "service_rates"
                if (
                    srv_key in results[interval]
                    and anchor in results[interval][srv_key]
                    and server in results[interval][srv_key][anchor]
                ):
                    srv_y = np.array(results[interval][srv_key][anchor][server])
                    srv_line, = ax2_ren.plot(
                        x, srv_y,
                        label=f"Service Rate | {anchor} | {server.replace('_',' ').title()}",
                        color=colors[anchor],
                        linestyle='-' if server == "server_1" else ':',
                        alpha=0.5
                    )
                    srv_label = f"Service Rate | {anchor} | {server.replace('_',' ').title()}"
                    if srv_label not in all_labels:
                        all_handles.append(srv_line)
                        all_labels.append(srv_label)

        ax_ren.set_title(f"Reneging Rates | Interval {interval}s")
        ax_ren.set_xlabel("Steps")
        ax_ren.set_ylabel("Reneging Rate")
        ax2_ren.set_ylabel("Service Rate", color="gray", fontsize=10)
        ax2_ren.tick_params(axis='y', labelcolor='gray')

        # Jockeying rate plots
        ax_jky = axs[1, col]
        ax2_jky = ax_jky.twinx()
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                # Jockeying Rates
                y = np.array(results[interval]["jockeying_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_jky.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)

                # Service Rates (if present)
                srv_key = "service_rates"
                if (
                    srv_key in results[interval]
                    and anchor in results[interval][srv_key]
                    and server in results[interval][srv_key][anchor]
                ):
                    srv_y = np.array(results[interval][srv_key][anchor][server])
                    srv_line, = ax2_jky.plot(
                        x, srv_y,
                        label=f"Service Rate | {anchor} | {server.replace('_',' ').title()}",
                        color=colors[anchor],
                        linestyle='-' if server == "server_1" else ':',
                        alpha=0.5
                    )
                    srv_label = f"Service Rate | {anchor} | {server.replace('_',' ').title()}"
                    if srv_label not in all_labels:
                        all_handles.append(srv_line)
                        all_labels.append(srv_label)

        ax_jky.set_title(f"Jockeying Rates | Interval {interval}s")
        ax_jky.set_xlabel("Steps")
        ax_jky.set_ylabel("Jockeying Rate")
        ax2_jky.set_ylabel("Service Rate", color="gray", fontsize=10)
        ax2_jky.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    # Only one legend at the bottom!
    fig.legend(
        all_handles, 
        all_labels,
        loc='lower center',
        ncol=min(4, len(all_labels)), 
        bbox_to_anchor=(0.5, -0.025),
        fontsize=11
    )
    plt.show()
 

def plot_six_panels(results, intervals, jockey_anchors):

    fig, axs = plt.subplots(2, len(intervals), figsize=(6*len(intervals), 8), sharex=False)
    colors = {
        "markov_model_service_rate": "blue",
        "markov_model_inter_change_time": "green"
    }
    linestyles = {
        "markov_model_service_rate": "-",
        "markov_model_inter_change_time": "--"
    }

    # Accumulate handles and labels for single legend
    all_handles = []
    all_labels = []

    for col, interval in enumerate(intervals):
        # Reneging rate plots
        ax_ren = axs[0, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                y = np.array(results[interval]["reneging_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_ren.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)

        ax_ren.set_title(f"Reneging Rates | Interval {interval}s")
        ax_ren.set_xlabel("Steps")
        ax_ren.set_ylabel("Reneging Rate")

        # Jockeying rate plots
        ax_jky = axs[1, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                y = np.array(results[interval]["jockeying_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_jky.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)

        ax_jky.set_title(f"Jockeying Rates | Interval {interval}s")
        ax_jky.set_xlabel("Steps")
        ax_jky.set_ylabel("Jockeying Rate")

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
        
    # Only one legend at the bottom!
    fig.legend(
        all_handles, 
        all_labels,
        loc='lower center',
        ncol=min(4, len(all_labels)), 
        bbox_to_anchor=(0.5, 0.035), #-0.025),
        fontsize=10,
        frameon=True,
        borderaxespad=1,
        bbox_transform=fig.transFigure
    )
    plt.subplots_adjust(top=0.91, bottom=0.15)
    
    plt.show()


def plot_six_panels_combo(results, intervals, jockey_anchors, histories_egreedy=None):

    fig, axs = plt.subplots(2, len(intervals), figsize=(6*len(intervals), 8), sharex=False)
    colors = {
        "markov_model_service_rate": "blue",
        "markov_model_inter_change_time": "green",
        "egreedy_reneging": "darkorange",  # New color for e-greedy reneging
        "egreedy_jockeying": "purple",     # New color for e-greedy jockeying
    }
    linestyles = {
        "markov_model_service_rate": "-",
        "markov_model_inter_change_time": "--",
        "egreedy": "-."
    }

    all_handles = []
    all_labels = []

    for col, interval in enumerate(intervals):
        ax_ren = axs[0, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                y = np.array(results[interval]["reneging_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_ren.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)
        # E-greedy (reneging) - dark orange
        if histories_egreedy is not None:
            for h in histories_egreedy:
                if h['interval'] == interval:
                    anchor = h['anchor']
                    y = [obs.get('Waited', 0) for obs in h['history'] if '_reneged' in obs.get('customerid', '')]
                    x = np.arange(len(y))
                    if len(y) > 0:
                        line, = ax_ren.plot(
                            x, y,
                            label=f"E-greedy Reneging | {anchor}",
                            color=colors["egreedy_reneging"],
                            linestyle=linestyles["egreedy"]
                        )
                        label = f"E-greedy Reneging | {anchor}"
                        if label not in all_labels:
                            all_handles.append(line)
                            all_labels.append(label)

        ax_ren.set_title(f"Reneging Rates | Interval {interval}s")
        ax_ren.set_xlabel("Steps")
        ax_ren.set_ylabel("Reneging Rate")

        ax_jky = axs[1, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                y = np.array(results[interval]["jockeying_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_jky.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)
        # E-greedy (jockeying) - purple
        if histories_egreedy is not None:
            for h in histories_egreedy:
                if h['interval'] == interval:
                    anchor = h['anchor']
                    y = [obs.get('Waited', 0) for obs in h['history'] if '_jockeyed' in obs.get('customerid', '')]
                    x = np.arange(len(y))
                    if len(y) > 0:
                        line, = ax_jky.plot(
                            x, y,
                            label=f"E-greedy Jockeying | {anchor}",
                            color=colors["egreedy_jockeying"],
                            linestyle=linestyles["egreedy"]
                        )
                        label = f"E-greedy Jockeying | {anchor}"
                        if label not in all_labels:
                            all_handles.append(line)
                            all_labels.append(label)

        ax_jky.set_title(f"Jockeying Rates | Interval {interval}s")
        ax_jky.set_xlabel("Steps")
        ax_jky.set_ylabel("Jockeying Rate")

    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.legend(
        all_handles, 
        all_labels,
        loc='lower center',
        ncol=min(4, len(all_labels)), 
        bbox_to_anchor=(0.5, 0.035),
        fontsize=10,
        frameon=True,
        borderaxespad=1,
        bbox_transform=fig.transFigure
    )
    plt.subplots_adjust(top=0.91, bottom=0.15)
    plt.show()


def plot_rates_vs_service_rates(results, intervals, jockey_anchors):

    fig, axs = plt.subplots(2, len(intervals), figsize=(6*len(intervals), 8), sharex=False)
    colors = {
        "markov_model_service_rate": "blue",
        "markov_model_inter_change_time": "green"
    }
    linestyles = {
        "server_1": "-",
        "server_2": ":"
    }

    all_handles = []
    all_labels = []
    marker_handles = []
    marker_labels = []

    for col, interval in enumerate(intervals):
        # Reneging rate vs service rate
        ax_ren = axs[0, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                y = np.array(results[interval]["reneging_rates"][anchor][server])
                s = np.array(results[interval]["service_rates"][anchor][server])
                mask = (~np.isnan(s)) & (~np.isnan(y))
                if np.sum(mask) == 0:
                    continue
                # Sort by service rate
                sort_idx = np.argsort(s[mask])
                s_sorted = s[mask][sort_idx]
                y_sorted = y[mask][sort_idx]
                line, = ax_ren.plot(
                    s_sorted, y_sorted,
                    label=f"Reneging | {anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[server]
                )
                label = f"Reneging | {anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)
                # Plot marker at minimal reneging rate
                if len(y_sorted) > 0:
                    min_idx = np.nanargmin(y_sorted)
                    marker, = ax_ren.plot(
                        s_sorted[min_idx], y_sorted[min_idx],
                        marker='*', color='red', markersize=12, linestyle='None',
                        label="Min Reneging Rate" if not marker_labels or "Min Reneging Rate" not in marker_labels else "")
                    if "Min Reneging Rate" not in marker_labels:
                        marker_handles.append(marker)
                        marker_labels.append("Min Reneging Rate")

        ax_ren.set_title(f"Reneging Rate vs Service Rate\nInterval {interval}s")
        ax_ren.set_xlabel("Service Rate")
        ax_ren.set_ylabel("Reneging Rate")
        ax_ren.grid(True, linestyle='--', alpha=0.5)

        # Jockeying rate vs service rate
        ax_jky = axs[1, col]
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                y = np.array(results[interval]["jockeying_rates"][anchor][server])
                s = np.array(results[interval]["service_rates"][anchor][server])
                mask = (~np.isnan(s)) & (~np.isnan(y))
                if np.sum(mask) == 0:
                    continue
                sort_idx = np.argsort(s[mask])
                s_sorted = s[mask][sort_idx]
                y_sorted = y[mask][sort_idx]
                line, = ax_jky.plot(
                    s_sorted, y_sorted,
                    label=f"Jockeying | {anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[server]
                )
                label = f"Jockeying | {anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)
                # Plot marker at minimal jockeying rate
                if len(y_sorted) > 0:
                    min_idx = np.nanargmin(y_sorted)
                    marker, = ax_jky.plot(
                        s_sorted[min_idx], y_sorted[min_idx],
                        marker='o', color='red', markersize=10, linestyle='None',
                        label="Min Jockeying Rate" if not marker_labels or "Min Jockeying Rate" not in marker_labels else "")
                    if "Min Jockeying Rate" not in marker_labels:
                        marker_handles.append(marker)
                        marker_labels.append("Min Jockeying Rate")

        ax_jky.set_title(f"Jockeying Rate vs Service Rate\nInterval {interval}s")
        ax_jky.set_xlabel("Service Rate")
        ax_jky.set_ylabel("Jockeying Rate")
        ax_jky.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    # Combine all legend items
    fig.legend(
        all_handles + marker_handles, all_labels + marker_labels,
        loc='lower center', ncol=min(4, len(all_labels + marker_labels)),
        bbox_to_anchor=(0.5, -0.01), fontsize=11
    )
    fig.suptitle("Reneging and Jockeying Rates as Functions of Service Rate", fontsize=16)
    plt.show()
    

def plot_six_panels_bad(results, intervals, jockey_anchors):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, len(intervals), figsize=(6*len(intervals), 8), sharex=False)
    colors = {
        "markov_model_service_rate": "blue",
        "markov_model_inter_change_time": "green"
    }
    linestyles = {
        "markov_model_service_rate": "-",
        "markov_model_inter_change_time": "--"
    }

    # Accumulate handles and labels for one bottom legend
    all_handles = []
    all_labels = []
    all_srv_handles = []
    all_srv_labels = []

    for col, interval in enumerate(intervals):
        # Reneging rate plots
        ax_ren = axs[0, col]
        ax2_ren = ax_ren.twinx()
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                # Reneging Rates
                y = np.array(results[interval]["reneging_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_ren.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)

                # Service Rates (if present)
                srv_key = "service_rates"
                if srv_key in results[interval] and anchor in results[interval][srv_key] and server in results[interval][srv_key][anchor]:
                    srv_y = np.array(results[interval][srv_key][anchor][server])
                    srv_line, = ax2_ren.plot(
                        x, srv_y,
                        label=f"Service Rate | {anchor} | {server.replace('_',' ').title()}",
                        color=colors[anchor],
                        linestyle='-' if server == "server_1" else ':',
                        alpha=0.5
                    )
                    srv_label = f"Service Rate | {anchor} | {server.replace('_',' ').title()}"
                    if srv_label not in all_srv_labels:
                        all_srv_handles.append(srv_line)
                        all_srv_labels.append(srv_label)

        ax_ren.set_title(f"Reneging Rates | Interval {interval}s")
        ax_ren.set_xlabel("Steps")
        ax_ren.set_ylabel("Reneging Rate")
        ax2_ren.set_ylabel("Service Rate", color="gray", fontsize=10)
        ax2_ren.tick_params(axis='y', labelcolor='gray')

        # Jockeying rate plots
        ax_jky = axs[1, col]
        ax2_jky = ax_jky.twinx()
        for anchor in jockey_anchors:
            for server in ["server_1", "server_2"]:
                # Jockeying Rates
                y = np.array(results[interval]["jockeying_rates"][anchor][server])
                x = np.arange(len(y))
                line, = ax_jky.plot(
                    x, y,
                    label=f"{anchor} | {server.replace('_',' ').title()}",
                    color=colors[anchor],
                    linestyle=linestyles[anchor] if server == "server_1" else ':'
                )
                label = f"{anchor} | {server.replace('_',' ').title()}"
                if label not in all_labels:
                    all_handles.append(line)
                    all_labels.append(label)

                # Service Rates (if present)
                srv_key = "service_rates"
                if srv_key in results[interval] and anchor in results[interval][srv_key] and server in results[interval][srv_key][anchor]:
                    srv_y = np.array(results[interval][srv_key][anchor][server])
                    srv_line, = ax2_jky.plot(
                        x, srv_y,
                        label=f"Service Rate | {anchor} | {server.replace('_',' ').title()}",
                        color=colors[anchor],
                        linestyle='-' if server == "server_1" else ':',
                        alpha=0.5
                    )
                    srv_label = f"Service Rate | {anchor} | {server.replace('_',' ').title()}"
                    if srv_label not in all_srv_labels:
                        all_srv_handles.append(srv_line)
                        all_srv_labels.append(srv_label)

        ax_jky.set_title(f"Jockeying Rates | Interval {interval}s")
        ax_jky.set_xlabel("Steps")
        ax_jky.set_ylabel("Jockeying Rate")
        ax2_jky.set_ylabel("Service Rate", color="gray", fontsize=10)
        ax2_jky.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    # One single legend for all at the bottom
    fig.legend(
        all_handles + all_srv_handles, 
        all_labels + all_srv_labels,
        loc='lower center', ncol=min(4, len(all_labels + all_srv_labels)), 
        bbox_to_anchor=(0.5, -0.02), fontsize=11
    )
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
            if isinstance(obs, dict):
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
                        
            elif isinstance(obs, list):
                # If obs is a list, iterate through its items
                for o in obs:
                    if not isinstance(o, dict):
                        continue
                    waited = o.get('Waited', None)
                    cid = o.get('customerid', '')
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
        fig.suptitle(f"{title}\nAnchor: {anchor}", fontsize=16)

        for j, interval in enumerate(intervals):
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
            if j == 0:
                ax.set_ylabel('Avg Waiting Time (Moving Avg)')
            ax.set_title(f"Interval: {interval}")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()        
       

def plot_rates_vs_requests(results, intervals, anchors, metric="jockeying_rates"):
    """
    Plot jockeying or reneging rates vs. number of requests (simulation step), for each anchor and interval.
    Args:
        results: the nested data structure with rates
        intervals: list of intervals (e.g., [3, 6, 9])
        anchors: list of anchor names (e.g., ["steady_state_distribution", "inter_change_time"])
        metric: either 'jockeying_rates' or 'reneging_rates'
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    servers = ["server_1", "server_2"]

    fig, axs = plt.subplots(1, len(intervals), figsize=(6*len(intervals), 5), sharey=True)
    if len(intervals) == 1:
        axs = [axs]
    for j, interval in enumerate(intervals):
        ax = axs[j]
        for i, anchor in enumerate(anchors):
            for k, server in enumerate(servers):
                y = results[interval][metric][anchor][server]
                x = np.arange(1, len(y)+1)  # Number of requests as 1-based index
                label = f"{anchor}, {server.replace('_', ' ').title()}"
                ax.plot(x, y, label=label, color=colors[i*2 + k])
        ax.set_title(f"{metric.replace('_',' ').capitalize()} (Interval: {interval}s)")
        ax.set_xlabel("Number of Requests")
        ax.set_ylabel(metric.replace('_',' ').capitalize())
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
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

    from collections import defaultdict

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

            # --- Fix: Union of all x bins, align both y series ---
            all_xs = sorted(set(bin_xs_p) | set(bin_xs_np))
            y_policy = np.full(len(all_xs), np.nan)
            y_nopolicy = np.full(len(all_xs), np.nan)
            if len(bin_xs_p) > 0:
                idx_map_p = {x: i for i, x in enumerate(bin_xs_p)}
                for i, x in enumerate(all_xs):
                    if x in idx_map_p:
                        y_policy[i] = bin_ys_p[idx_map_p[x]]
            if len(bin_xs_np) > 0:
                idx_map_np = {x: i for i, x in enumerate(bin_xs_np)}
                for i, x in enumerate(all_xs):
                    if x in idx_map_np:
                        y_nopolicy[i] = bin_ys_np[idx_map_np[x]]

            ax = axs[row, col]
            # Policy curve
            if np.any(~np.isnan(y_policy)):
                ax.plot(all_xs, y_policy, '-', label='Policy', color=color_map['Policy'])
            # No policy curve
            if np.any(~np.isnan(y_nopolicy)):
                ax.plot(all_xs, y_nopolicy, '--', label='No Policy', color=color_map['No Policy'])

            # Also scatter the raw points
            if len(xs_policy) > 0:
                ax.scatter(xs_policy, ys_policy, color=color_map['Policy'], alpha=0.3, marker='o')
            if len(xs_np) > 0:
                ax.scatter(xs_np, ys_np, color=color_map['No Policy'], alpha=0.3, marker='x')

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




def plot_avg_wait_by_queue_length_grouped_by_anchor(
    histories_policy=None, histories_nopolicy=None, window=1, 
    title="Avg Waiting Time vs Queue Length (Grouped by Anchor)"
):
    """
    Plots average waiting time vs queue length for both Policy and No-Policy runs,
    grouped by anchor and interval, for outcomes 'jockeyed' and 'reneged'.
    """
    # Use global variables if not provided
    if histories_policy is None:
        global GLOBAL_SIMULATION_HISTORIES_POLICY
        histories_policy = GLOBAL_SIMULATION_HISTORIES_POLICY
    if histories_nopolicy is None:
        global GLOBAL_SIMULATION_HISTORIES_NOPOLICY
        histories_nopolicy = GLOBAL_SIMULATION_HISTORIES_NOPOLICY

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
                if not isinstance(obs, dict):
                    continue
                waited = obs.get('Waited', None)
                cid = obs.get('customerid', '')
                queue_length = obs.get('queue_length', None)
                if waited is not None and queue_length is not None and not np.isnan(queue_length):
                    if (outcome == 'reneged' and '_reneged' in cid) or \
                       (outcome == 'jockeyed' and '_jockeyed' in cid):
                        xs.append(queue_length)
                        ys.append(waited)
        return np.array(xs), np.array(ys)

    def binned_average(xs, ys, bin_xs, window=1):
        # xs: (queue_lengths), ys: (wait_times), bin_xs: all possible queue_lengths
        # Returns: arrays of mean waiting time for each bin_x, NaN if no data for that bin
        bins = defaultdict(list)
        for x, y in zip(xs, ys):
            bins[int(x)].append(y)
        bin_ys = []
        for x in bin_xs:
            if x in bins and bins[x]:
                bin_ys.append(np.mean(bins[x]))
            else:
                bin_ys.append(np.nan)
        bin_ys = np.array(bin_ys)
        # Optionally smooth with moving average (ignore NaNs in window)
        if window > 1 and len(bin_ys) >= window:
            pad = window // 2
            ys_smooth = []
            for i in range(len(bin_ys)):
                left = max(0, i - pad)
                right = min(len(bin_ys), i + pad + 1)
                window_vals = bin_ys[left:right]
                window_vals = window_vals[~np.isnan(window_vals)]
                if len(window_vals) > 0:
                    ys_smooth.append(np.mean(window_vals))
                else:
                    ys_smooth.append(np.nan)
            bin_ys = np.array(ys_smooth)
        return np.array(bin_xs), bin_ys

    color_map = {
        'Policy': 'tab:blue',
        'No Policy': 'tab:orange'
    }

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.3 * nrows), squeeze=False, sharey='row')

    for row, interval in enumerate(all_intervals):
        for col, (anchor, outcome) in enumerate(itertools.product(all_anchors, outcomes)):
            ax = axs[row, col]
            plotted = False

            # Get all queue lengths (bins) present in either policy or no policy for this subplot
            xs_policy, ys_policy = extract_by_anchor_interval(histories_policy, anchor, interval, outcome)
            xs_np, ys_np = extract_by_anchor_interval(histories_nopolicy, anchor, interval, outcome)
            all_bins = set(xs_policy.tolist() + xs_np.tolist())
            if len(all_bins) == 0:
                all_bins = set([0])  # placeholder if empty
            bin_xs = np.array(sorted(all_bins))

            # Policy
            bin_xs_p, bin_ys_p = binned_average(xs_policy, ys_policy, bin_xs, window)
            if np.any(~np.isnan(bin_ys_p)):
                ax.plot(bin_xs_p, bin_ys_p, '-', label='Policy', color=color_map['Policy'])
                plotted = True

            # No Policy
            bin_xs_np, bin_ys_np = binned_average(xs_np, ys_np, bin_xs, window)
            if np.any(~np.isnan(bin_ys_np)):
                ax.plot(bin_xs_np, bin_ys_np, '--', label='No Policy', color=color_map['No Policy'])
                plotted = True

            # Optionally: scatter the raw points for context
            ax.scatter(xs_policy, ys_policy, color=color_map['Policy'], alpha=0.2, marker='o')
            ax.scatter(xs_np, ys_np, color=color_map['No Policy'], alpha=0.2, marker='x')

            # Labels
            if row == 0:
                ax.set_title(f"{anchor}\n{outcome.title()}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Interval: {interval}\nAvg Waiting Time", fontsize=11)
            ax.set_xlabel('Queue Length')
            ax.grid(True, linestyle='--', alpha=0.5)
            if plotted:
                ax.legend(fontsize=9)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
    
    # Place this after all subplot plotting
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    fig.suptitle(title, fontsize=17)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()    


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def surface_plot_multi_interval(request_objs, intervals):
    n = len(intervals)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(7 * ncols, 6 * nrows))

    for idx, (requestObj, interval) in enumerate(zip(request_objs, intervals)):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

        mu_i_traj = getattr(requestObj, 'srvrates1_history', None)
        mu_j_traj = getattr(requestObj, 'srvrates2_history', None)

        # Only plot if we have real history
        if mu_i_traj is not None and len(mu_i_traj) > 0 and mu_j_traj is not None and len(mu_j_traj) > 0:
            mu_i_traj = np.asarray(mu_i_traj)
            mu_j_traj = np.asarray(mu_j_traj)
            mu_i_opt = mu_i_traj[-1]
            mu_j_opt = mu_j_traj[-1]

            # Get sensible axis limits
            mu_i_min = np.min(mu_i_traj) * 0.9
            mu_i_max = np.max(mu_i_traj) * 1.1
            mu_j_min = np.min(mu_j_traj) * 0.9
            mu_j_max = np.max(mu_j_traj) * 1.1
            # Handle case of constant value (avoid min==max)
            if np.allclose(mu_i_min, mu_i_max):
                mu_i_min = mu_i_opt * 0.9
                mu_i_max = mu_i_opt * 1.1
            if np.allclose(mu_j_min, mu_j_max):
                mu_j_min = mu_j_opt * 0.9
                mu_j_max = mu_j_opt * 1.1

            n_points = 50
            mu_i_vals = np.linspace(mu_i_min, mu_i_max, n_points)
            mu_j_vals = np.linspace(mu_j_min, mu_j_max, n_points)
            MI, MJ = np.meshgrid(mu_i_vals, mu_j_vals)

            def objective(mu_i, mu_j):
                return (1.0 / mu_i + 1.0 / mu_j) + 0.5 * np.abs(mu_i - mu_j)

            F = objective(MI, MJ)
            surf = ax.plot_surface(MI, MJ, F, cmap='viridis', linewidth=0, antialiased=True, alpha=0.7)

            # Plot trajectory
            traj_obj = objective(mu_i_traj, mu_j_traj)
            ax.plot(mu_i_traj, mu_j_traj, traj_obj, color='red', lw=2, label='Dynamic Path')
            # Start and optimal
            ax.scatter(mu_i_traj[0], mu_j_traj[0], traj_obj[0], color='orange', s=30, label='Start')
            ax.scatter(mu_i_opt, mu_j_opt, traj_obj[-1], color='red', s=50, label='Final/Optimal')
            # Annotate
            ax.text(mu_i_opt, mu_j_opt, traj_obj[-1],
                    f"$\\mu_1$={mu_i_opt:.2f}\n$\\mu_2$={mu_j_opt:.2f}", color='black', fontsize=10)
        else:
            # If not enough data, show a warning
            ax.text2D(0.1, 0.5, "No trajectory data", transform=ax.transAxes, color='red', fontsize=12)
            ax.set_xlim(0.9, 1.2)
            ax.set_ylim(0.9, 1.2)
            ax.set_zlim(1.5, 2.5)

        ax.set_xlabel(r'$\mu_1$')
        ax.set_ylabel(r'$\mu_2$')
        ax.set_zlabel('Objective Value')
        ax.set_title(f'Surface, Interval={interval}s')
        ax.legend()
        if 'surf' in locals():
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

    plt.tight_layout()
    plt.show()
    

def surface_plot_comparison(
    surface_datas,  # List of dicts: each with X, Y, Z, interval, opt_path, opt_final, nonopt_path, nonopt_final, static_final
    intervals #,
    # fig_title="Surface Plot Comparison"
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    n = len(surface_datas)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(14, 11))
    #fig.suptitle(fig_title, fontsize=16)

    for idx, (data, interval) in enumerate(zip(surface_datas, intervals)):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        X, Y, Z = data['X'], data['Y'], data['Z']

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75, edgecolor='none')
        mappable = surf
        fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, pad=0.1)

        # OPTIMIZED PATH
        ax.plot(
            data['opt_path'][:,0], data['opt_path'][:,1], data['opt_path'][:,2],
            color='orange', lw=2.5, label='Optimized Path', zorder=20
        )
        # Optimized Final Marker
        ax.scatter(
            data['opt_final'][0], data['opt_final'][1], data['opt_final'][2],
            color='yellow', edgecolor='black', marker='*', s=320, zorder=100, 
            label='Optimized Final', depthshade=False
        )

        # NON-OPTIMIZED PATH
        ax.plot(
            data['nonopt_path'][:,0], data['nonopt_path'][:,1], data['nonopt_path'][:,2],
            color='purple', lw=2.5, label='Non-Optimized Path', zorder=20
        )
        ax.scatter(
            data['nonopt_final'][0], data['nonopt_final'][1], data['nonopt_final'][2],
            color='cyan', edgecolor='black', marker='o', s=200, zorder=100, 
            label='Non-Opt Final', depthshade=False
        )

        # Static Final (if provided)
        if 'static_final' in data:
            ax.scatter(
                data['static_final'][0], data['static_final'][1], data['static_final'][2],
                color='magenta', edgecolor='black', marker='^', s=200, zorder=100,
                label='Static Final', depthshade=False
            )

        # Annotate points for visibility and value
        f_opt = data['opt_final']
        ax.text(
            f_opt[0], f_opt[1], f_opt[2]+0.25,
            f"Opt\nμ₁={f_opt[0]:.2f}\nμ₂={f_opt[1]:.2f}\nObj={f_opt[2]:.2f}",
            color='black', fontsize=10, weight='bold', zorder=200,
            bbox=dict(facecolor="yellow", alpha=0.7, boxstyle="round,pad=0.2")
        )
        f_nonopt = data['nonopt_final']
        ax.text(
            f_nonopt[0], f_nonopt[1], f_nonopt[2]+0.25,
            f"Non-Opt\nμ₁={f_nonopt[0]:.2f}\nμ₂={f_nonopt[1]:.2f}\nObj={f_nonopt[2]:.2f}",
            color='black', fontsize=10, weight='bold', zorder=200,
            bbox=dict(facecolor="cyan", alpha=0.7, boxstyle="round,pad=0.2")
        )
        if 'static_final' in data:
            f_static = data['static_final']
            ax.text(
                f_static[0], f_static[1], f_static[2]+0.25,
                f"Static\nμ₁={f_static[0]:.2f}\nμ₂={f_static[1]:.2f}\nObj={f_static[2]:.2f}",
                color='black', fontsize=10, weight='bold', zorder=200,
                bbox=dict(facecolor="magenta", alpha=0.7, boxstyle="round,pad=0.2")
            )

        ax.set_xlabel('$\mu_1$')
        ax.set_ylabel('$\mu_2$')
        ax.set_zlabel('Objective Value')
        ax.set_title(f"Surface, Interval={interval}s")

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def bird_surface_plot_comparison(
    surface_datas,
    intervals
    #fig_title="Surface Plot Comparison"
):
    
    from mpl_toolkits.mplot3d import Axes3D    
    from matplotlib.lines import Line2D
    from scipy.interpolate import interp1d

    n = len(surface_datas)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(14, 11))
    # fig.suptitle(fig_title, fontsize=16)

    # Add path handles only once, marker handles for every interval
    legend_handles = [
        Line2D([0], [0], color='orange', lw=2.5, label='Optimized Path'),
        Line2D([0], [0], color='red', lw=2.5, label='Non-Optimized Path'),
    ]
    legend_labels = [
        'Optimized Path',
        'Non-Optimized Path',
    ]

    for idx, (data, interval) in enumerate(zip(surface_datas, intervals)):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        X, Y, Z = data['X'], data['Y'], data['Z']
        ax.view_init(elev=75, azim=-60)
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75, edgecolor='none')

        # Move the colorbar to the left of the surface plot
        pos = ax.get_position()
        cbar_width = 0.012  # Width of colorbar as fraction of figure width
        cbar_pad = 0.015    # Padding between colorbar and plot
        cbar_height = pos.height * 0.85
        cbar_bottom = pos.y0 + (pos.height - cbar_height) / 2
        cbar_left = pos.x0 - cbar_width - cbar_pad
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        fig.colorbar(surf, cax=cbar_ax, orientation='vertical')
        cbar_ax.set_ylabel('Objective Value', fontsize=9)

        # Plot all paths first
        opt_path = data['opt_path']
        if len(opt_path) >= 4:
            t = np.linspace(0, 1, len(opt_path))
            interp_points = 100
            tt = np.linspace(0, 1, interp_points)
            x_spline = interp1d(t, opt_path[:,0], kind='cubic')
            y_spline = interp1d(t, opt_path[:,1], kind='cubic')
            z_spline = interp1d(t, opt_path[:,2], kind='cubic')
            x_smooth = x_spline(tt)
            y_smooth = y_spline(tt)
            z_smooth = z_spline(tt)
            ax.plot(x_smooth, y_smooth, z_smooth, color='orange', lw=2.5)
        else:
            ax.plot(opt_path[:,0], opt_path[:,1], opt_path[:,2], color='orange', lw=1.5)
        # Non-Optimized Path: Red
        nonopt_path = data['nonopt_path']
        ax.plot(
            nonopt_path[:,0], nonopt_path[:,1], nonopt_path[:,2],
            color='red', lw=1.5
        )

        # Now plot all markers LAST so they are on top
        # Optimized Final
        ax.scatter(
            data['opt_final'][0], data['opt_final'][1], data['opt_final'][2],
            color='red', edgecolor='black', marker='*', s=320, 
            depthshade=False, zorder=1000
        )
        # Non-Optimized Final
        ax.scatter(
            data['nonopt_final'][0], data['nonopt_final'][1], data['nonopt_final'][2],
            color='cyan', edgecolor='black', marker='o', s=200,
            depthshade=False, zorder=1000
        )
        # Static Final (if provided)
        if 'static_final' in data:
            ax.scatter(
                data['static_final'][0], data['static_final'][1], data['static_final'][2],
                color='magenta', edgecolor='black', marker='^', s=200,
                depthshade=False, zorder=1000
            )

        # Prepare marker value entries for legend
        f_opt = data['opt_final']
        f_nonopt = data['nonopt_final']
        interval_label = f"Interval={interval}s"
        opt_label = f"Optimized Final ({interval_label}): μ₁={f_opt[0]:.2f}, μ₂={f_opt[1]:.2f}, Obj={f_opt[2]:.2f}"
        nonopt_label = f"Non-Opt Final ({interval_label}): μ₁={f_nonopt[0]:.2f}, μ₂={f_nonopt[1]:.2f}, Obj={f_nonopt[2]:.2f}"

        legend_handles.append(Line2D([0], [0], marker='*', color='w',
                                     label=opt_label,
                                     markerfacecolor='red', markeredgecolor='black',
                                     markersize=18, linewidth=0))
        legend_labels.append(opt_label)

        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     label=nonopt_label,
                                     markerfacecolor='cyan', markeredgecolor='black',
                                     markersize=13, linewidth=0))
        legend_labels.append(nonopt_label)

        # Axis labels
        ax.set_xlabel('$\mu_1$')
        ax.set_ylabel('$\mu_2$')
        ax.set_zlabel('Objective Value', fontsize=10)  # Remove default zlabel

        # Add "Objective Value" label at the top right (now unobstructed)
        #ax.text2D(
        #    0.97, 0.93,
        #    "Objective Value",
        #    transform=ax.transAxes,
        #    ha='right', va='center', fontsize=10, fontweight='normal'
        #)
        ax.set_title(f"Surface, Interval={interval}s")

    # Place legend at the bottom center, and ensure it's not cut off
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.035),  # Move further up from the bottom edge
        fontsize=10,
        frameon=True,
        ncol=2,
        borderaxespad=1,
        bbox_transform=fig.transFigure
    )
    # Reduce top space, increase bottom space for the legend
    plt.subplots_adjust(top=0.91, bottom=0.15)
    plt.show()
    

def build_surface_datas_for_comparison_original(policy_objs, static_objs, request_objs_egreedy, intervals, objective_func):
    """
    Returns a list of dicts (surface_datas) for use in surface_plot_comparison,
    one per interval, each containing all required keys for plotting.
    Each dict has keys: X, Y, Z, opt_path, opt_final, nonopt_path, nonopt_final, (optionally static_final)
    - policy_objs: list of RequestQueue objects for each interval (policy-enabled)
    - static_objs: list of RequestQueue objects for each interval (no-policy)
    - intervals: list of intervals matching the above
    - objective_func: function of (mu1, mu2) returning Z
    """

    surface_datas = []
    for rq_pol, rq_stat, interval in zip(policy_objs, static_objs, intervals):
        # Get histories of service rates for both paths
        mu1_pol = np.asarray(getattr(rq_pol, 'srvrates1_history', []))
        mu2_pol = np.asarray(getattr(rq_pol, 'srvrates2_history', []))
        mu1_stat = np.asarray(getattr(rq_stat, 'srvrates1_history', []))
        mu2_stat = np.asarray(getattr(rq_stat, 'srvrates2_history', []))

        # Concatenate all for surface limits
        all_mu1 = np.concatenate([mu1_pol, mu1_stat]) if mu1_pol.size and mu1_stat.size else np.array([1,2])
        all_mu2 = np.concatenate([mu2_pol, mu2_stat]) if mu2_pol.size and mu2_stat.size else np.array([1,2])

        mu1_min, mu1_max = np.min(all_mu1)*0.9, np.max(all_mu1)*1.1
        mu2_min, mu2_max = np.min(all_mu2)*0.9, np.max(all_mu2)*1.1

        if np.allclose(mu1_min, mu1_max): mu1_min, mu1_max = all_mu1[-1]*0.9, all_mu1[-1]*1.1
        if np.allclose(mu2_min, mu2_max): mu2_min, mu2_max = all_mu2[-1]*0.9, all_mu2[-1]*1.1

        n_points = 50
        mu1_vals = np.linspace(mu1_min, mu1_max, n_points)
        mu2_vals = np.linspace(mu2_min, mu2_max, n_points)
        X, Y = np.meshgrid(mu1_vals, mu2_vals)
        Z = objective_func(X, Y)

        # Prepare paths (N,3)
        opt_path = np.column_stack([mu1_pol, mu2_pol, objective_func(mu1_pol, mu2_pol)]) if mu1_pol.size and mu2_pol.size else np.empty((0,3))
        nonopt_path = np.column_stack([mu1_stat, mu2_stat, objective_func(mu1_stat, mu2_stat)]) if mu1_stat.size and mu2_stat.size else np.empty((0,3))
        # Final points
        opt_final = opt_path[-1] if len(opt_path) else np.array([0,0,0])
        nonopt_final = nonopt_path[-1] if len(nonopt_path) else np.array([0,0,0])

        surface_datas.append({
            'X': X, 'Y': Y, 'Z': Z,
            'interval': interval,
            'opt_path': opt_path,
            'opt_final': opt_final,
            'nonopt_path': nonopt_path,
            'nonopt_final': nonopt_final
            # Optionally, add 'static_final' if you have a third type
        })
    return surface_datas
    
 
def build_surface_datas_for_comparison(policy_objs, static_objs, request_objs_egreedy, intervals, objective_func):
    """
    Returns a list of dicts (surface_datas) for use in surface_plot_comparison,
    one per interval, each containing all required keys for plotting.
    Each dict has keys: X, Y, Z, opt_path, opt_final, nonopt_path, nonopt_final,
    eg_path, eg_final (optionally static_final)
    - policy_objs: list of RequestQueue objects for each interval (policy-enabled)
    - static_objs: list of RequestQueue objects for each interval (no-policy)
    - request_objs_egreedy: list of RequestQueue objects for each interval (e-greedy)
    - intervals: list of intervals matching the above
    - objective_func: function of (mu1, mu2) returning Z
    """

    surface_datas = []
    for idx, (rq_pol, rq_stat, interval) in enumerate(zip(policy_objs, static_objs, intervals)):
        # Get histories of service rates for all three paths
        mu1_pol = np.asarray(getattr(rq_pol, 'srvrates1_history', []))
        mu2_pol = np.asarray(getattr(rq_pol, 'srvrates2_history', []))
        mu1_stat = np.asarray(getattr(rq_stat, 'srvrates1_history', []))
        mu2_stat = np.asarray(getattr(rq_stat, 'srvrates2_history', []))

        # E-greedy path for this interval, if present
        if request_objs_egreedy is not None and idx < len(request_objs_egreedy):
            rq_eg = request_objs_egreedy[idx]
            mu1_eg = np.asarray(getattr(rq_eg, 'srvrates1_history', []))
            mu2_eg = np.asarray(getattr(rq_eg, 'srvrates2_history', []))
        else:
            mu1_eg = np.array([])
            mu2_eg = np.array([])

        # Concatenate all for surface limits
        all_mu1 = np.concatenate([mu1_pol, mu1_stat, mu1_eg]) if (mu1_pol.size or mu1_stat.size or mu1_eg.size) else np.array([1,2])
        all_mu2 = np.concatenate([mu2_pol, mu2_stat, mu2_eg]) if (mu2_pol.size or mu2_stat.size or mu2_eg.size) else np.array([1,2])

        mu1_min, mu1_max = np.min(all_mu1)*0.9, np.max(all_mu1)*1.1
        mu2_min, mu2_max = np.min(all_mu2)*0.9, np.max(all_mu2)*1.1

        if np.allclose(mu1_min, mu1_max): mu1_min, mu1_max = all_mu1[-1]*0.9, all_mu1[-1]*1.1
        if np.allclose(mu2_min, mu2_max): mu2_min, mu2_max = all_mu2[-1]*0.9, all_mu2[-1]*1.1

        n_points = 50
        mu1_vals = np.linspace(mu1_min, mu1_max, n_points)
        mu2_vals = np.linspace(mu2_min, mu2_max, n_points)
        X, Y = np.meshgrid(mu1_vals, mu2_vals)
        Z = objective_func(X, Y)

        # Prepare paths (N,3)
        opt_path = np.column_stack([mu1_pol, mu2_pol, objective_func(mu1_pol, mu2_pol)]) if mu1_pol.size and mu2_pol.size else np.empty((0,3))
        nonopt_path = np.column_stack([mu1_stat, mu2_stat, objective_func(mu1_stat, mu2_stat)]) if mu1_stat.size and mu2_stat.size else np.empty((0,3))
        eg_path = np.column_stack([mu1_eg, mu2_eg, objective_func(mu1_eg, mu2_eg)]) if mu1_eg.size and mu2_eg.size else np.empty((0,3))

        # Final points
        opt_final = opt_path[-1] if len(opt_path) else np.array([0,0,0])
        nonopt_final = nonopt_path[-1] if len(nonopt_path) else np.array([0,0,0])
        eg_final = eg_path[-1] if len(eg_path) else np.array([0,0,0])

        surface_datas.append({
            'X': X, 'Y': Y, 'Z': Z,
            'interval': interval,
            'opt_path': opt_path,
            'opt_final': opt_final,
            'nonopt_path': nonopt_path,
            'nonopt_final': nonopt_final,
            'eg_path': eg_path,
            'eg_final': eg_final
        })
    return surface_datas

# Example objective function (adjust as needed for your model)
def objective_func(mu_i, mu_j):
    return (1.0 / mu_i + 1.0 / mu_j) + 0.5 * np.abs(mu_i - mu_j)
    
# Usage in your main or analysis code:
# surface_datas = build_surface_datas_for_comparison(request_objs_opt, request_objs_nonopt, intervals, example_objective_func)
# surface_plot_comparison(surface_datas, intervals) 
     
       
######### Globals ########
request_log = []


def main():
	
    utility_basic = 1.0
    discount_coef = 0.1
    
    #params = Params(
    #        lam=utility_basic["lam"], mu1=utility_basic["mu1"], mu2=utility_basic["mu2"],
    #        r1=utility_basic["r1"], r2=utility_basic["r2"],
    #        h1=utility_basic["h1"], h2=utility_basic["h2"],
    #        c1=utility_basic["c1"], c2=utility_basic["c2"],
    #        c12=utility_basic["c12"], c21=utility_basic["c21"],
    #        C_R=utility_basic.get("C_R", 5.0)
    #)
    
    params = Params(lam=2.0, mu1=2.0, mu2=2.0,
        r1=7.0, r2=7.0,
        h1=1.0, h2=1.0,
        c1=2.0, c2=2.0,
        c12=3.0, c21=3.0,
        C_R=4.0
    )
    # requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=True, params=params)
    duration = 20 #0 #00  
    
    # Add a configuration flag for the new e-greedy policy
    use_e_greedy_policy = True  # Set to True to enable comparison 
    
    # Set intervals for dispatching queue states
    intervals = [3, 5, 7, 9]
    wasted_by_interval = []
    per_outcome_by_interval = []
    request_objs = []
    
    request_objs_opt = []
    request_objs_nonopt = []
    request_objs_queue_length = []
    #jockey_anchors = ["markov_model_service_rate", "markov_model_inter_change_time"]
    jockey_anchors = ["markov_model_service_rate", "markov_model_inter_change_time", "policy_queue_length_thresholds"]

    all_results = {anchor: [] for anchor in jockey_anchors}
            
    # Iterate through each interval in the intervals array
    all_reneging_rates = []
    all_jockeying_rates = []             
    
    
    def run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, policy_enabled, jockey_anchors):
        
        results = {
            interval: {
                "reneging_rates": {anchor: {} for anchor in jockey_anchors},
                "jockeying_rates": {anchor: {} for anchor in jockey_anchors},
                "service_rates": {anchor: {} for anchor in jockey_anchors}
            }
            for interval in intervals
        }
     
        # Optionally, collect histories for the last interval/anchor run
        all_histories = []  # To store (interval, anchor, history) for each run
        
        # Instantiate your policy solvers ONCE (outside the loop)
        
        
        combined_policy_solver = CombinedPolicySolver(params)
        
        
        '''
        for interval in intervals:
            for anchor in jockey_anchors:
                if policy_enabled:
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
                    
                    request_objs_opt.append(requestObj)
                    # Store history with anchor and interval
                    GLOBAL_SIMULATION_HISTORIES_POLICY.append({
                        "anchor": anchor,
                        "interval": interval,
                        "history": list(requestObj.history)   # Copy to avoid mutation
                    }) 
                    
                elif policy_enabled and "policy_queue_length_thresholds" in anchor:
                    # This is your custom queue-length-threshold policy
                    requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=True, combined_policy_solver=combined_policy_solver, anchor=anchor )
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
                    
                    request_objs_queue_length.append(requestObj)
                    # Store history with anchor and interval
                    GLOBAL_SIMULATION_HISTORIES_POLICY_QUEUE_LEN.append({
                        "anchor": anchor,
                        "interval": interval,
                        "history": list(requestObj.history)   # Copy to avoid mutation
                    })                                       
                    
                else:
                    requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=False)
                    requestObj.jockey_anchor = anchor
                    requestObj.run(duration, interval)
                    results[interval]["reneging_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["reneging_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["reneging_rate"]),
                    }
                    results[interval]["jockeying_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["jockeying_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["jockeying_rate"]),
                    }
                    
                    request_objs_nonopt.append(requestObj) 
                    GLOBAL_SIMULATION_HISTORIES_NOPOLICY.append({
                        "anchor": anchor,
                        "interval": interval,
                        "history": list(requestObj.history)
                    })                
                                                                           
                all_histories.append({
                    "interval": interval,
                    "anchor": anchor,
                    #"history": list(self.get_history())  # Make a copy
                    "history": list(requestObj.get_history())  # Make a copy
                })

                if hasattr(requestObj, "request_log"):
                    requestObj.request_log.clear()
                elif "request_log" in globals():
                    request_log.clear()
                    
            request_objs.append(requestObj)
        
        return results, GLOBAL_SIMULATION_HISTORIES_POLICY if policy_enabled else GLOBAL_SIMULATION_HISTORIES_NOPOLICY  
        
        ''' 
             
        for interval in intervals:
            for anchor in jockey_anchors:
                if policy_enabled and not ("policy_queue_length_thresholds" in anchor):
					
                    # Default (Markov or baseline) policy
                    requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=policy_enabled)
                    requestObj.jockey_anchor = anchor
                    requestObj.run(duration, interval)
                    results[interval]["reneging_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["reneging_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["reneging_rate"]),
                    }
                    
                    results[interval]["jockeying_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["jockeying_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["jockeying_rate"]),
                    }
                    
                    request_objs_opt.append(requestObj)
                    GLOBAL_SIMULATION_HISTORIES_POLICY.append({
                        "anchor": anchor,
                        "interval": interval,
                        "history": list(requestObj.history)
                    })
                    
                elif policy_enabled and "policy_queue_length_thresholds" in anchor:
                    # Your custom queue-length-thresholds policy
                    requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=True, combined_policy_solver=combined_policy_solver, anchor=anchor, params=params)
                    requestObj.jockey_anchor = anchor
                    requestObj.run(duration, interval)
                    results[interval]["reneging_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["reneging_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["reneging_rate"]),
                    }
                    results[interval]["jockeying_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["jockeying_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["jockeying_rate"]),
                    }
                    request_objs_queue_length.append(requestObj)
                    GLOBAL_SIMULATION_HISTORIES_POLICY_QUEUE_LEN.append({
                        "anchor": anchor,
                        "interval": interval,
                        "history": list(requestObj.history)
                    })                    
                else:
                    # No policy case
                    requestObj = RequestQueue(utility_basic, discount_coef, policy_enabled=False)
                    requestObj.jockey_anchor = anchor
                    requestObj.run(duration, interval)
                    results[interval]["reneging_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["reneging_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["reneging_rate"]),
                    }
                    results[interval]["jockeying_rates"][anchor] = {
                        "server_1": list(requestObj.dispatch_data[interval]["server_1"]["jockeying_rate"]),
                        "server_2": list(requestObj.dispatch_data[interval]["server_2"]["jockeying_rate"]),
                    }
                    request_objs_nonopt.append(requestObj)
                    GLOBAL_SIMULATION_HISTORIES_NOPOLICY.append({
                        "anchor": anchor,
                        "interval": interval,
                        "history": list(requestObj.history)
                    })

                all_histories.append({
                    "interval": interval,
                    "anchor": anchor,
                    "history": list(requestObj.get_history())
                })

                if hasattr(requestObj, "request_log"):
                    requestObj.request_log.clear()
                elif "request_log" in globals():
                    request_log.clear()

            request_objs.append(requestObj)

        # FINAL RETURN: choose which histories to return based on which branch matches
        if policy_enabled and any("policy_queue_length_thresholds" in anchor for anchor in jockey_anchors):
            return results, GLOBAL_SIMULATION_HISTORIES_POLICY_QUEUE_LEN
        elif policy_enabled:
            return results, GLOBAL_SIMULATION_HISTORIES_POLICY
        else:
            return results, GLOBAL_SIMULATION_HISTORIES_NOPOLICY
        
    # With policy-driven service rates
    results_policy , histories_policy = run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, policy_enabled=True,  jockey_anchors=jockey_anchors)  
    # results_policy, histories_policy, histories_egreedy = run_simulation_for_policy_mode( utility_basic, discount_coef, intervals, duration, policy_enabled=True, jockey_anchors=jockey_anchors, use_e_greedy_policy=True)
    # results_policy, (histories_policy, histories_nopolicy, histories_egreedy) = run_simulation_for_policy_mode( utility_basic, discount_coef, intervals, duration, policy_enabled=True, jockey_anchors=jockey_anchors, use_e_greedy_policy=True) 

    # With static/non-policy-driven service rates
    results_no_policy, histories_nopolicy = run_simulation_for_policy_mode(utility_basic, discount_coef, intervals, duration, policy_enabled=False, jockey_anchors=jockey_anchors) # , use_e_greedy_policy=False)              
    
    waiting_times, outcomes, time_stamps = extract_waiting_times_and_outcomes(requestObj)
           
    '''
     come back to the functions below
    '''

    ## plot_boxplot_waiting_times_by_outcome_2x2(histories_nopolicy, histories_policy)
    ##### plot_boxplot_waiting_times_by_outcome_by_interval_and_anchor(histories_nopolicy, histories_policy)
    
    # service rates and jockeying/reneging rates not smooth
    # requestObj.plot_rates_by_intervals()
    
    plot_six_panels(results_no_policy, intervals, jockey_anchors)    
    plot_six_panels(results_policy, intervals, jockey_anchors)
    # plot_six_panels_combo(results_policy, intervals, jockey_anchors, histories_egreedy=histories_egreedy)         
        
    surface_datas = build_surface_datas_for_comparison(request_objs_opt, request_objs_nonopt,  intervals, objective_func) # request_objs_egreedy,
    bird_surface_plot_comparison(surface_datas, intervals)
    
    ###### surface_plot_multi_interval(request_objs, intervals)

    ## plot_all_avg_waiting_time_by_anchor_interval(histories_policy, histories_nopolicy, window=20)
    
    ## =>plot_avg_wait_by_queue_length_grouped_by_anchor(histories_policy, histories_nopolicy, window=2)
    
    ## plot_avg_wait_by_queue_length(histories_policy, histories_nopolicy, window=2) # gives three plots for wait vs length served,rneged, jockeyed
    
    ## plot_avg_wait_by_queue_length_grouped(histories_policy, histories_nopolicy, window=2)  # has wait vs length grouped along reneged and jockeyed  
    
    # --- New: Plot policy and model histories ---
    #print("\n[Diagnostics] Plotting policy evolution...")
    #plot_policy_history(requestObj.policy.get_policy_history())

    #print("\n[Diagnostics] Plotting predictive model fit history...")
    #plot_predictive_model_history(requestObj.predictive_model.get_fit_history())         
	 
if __name__ == "__main__":
    main()

    #params = Params(lam=2.0, mu1=2.0, mu2=2.0,
    #                r1=7.0, r2=7.0,
    #                h1=1.0, h2=1.0,
    #                c1=2.0, c2=2.0,
    #                c12=3.0, c21=3.0,
    #                C_R=4.0)
    
    # jockeying-only policy
    #jockey_res = compute_optimal_jockeying_policy(params, N_x=20, N_y=20, verbose=True)
    #print("Jockeying policy a0 sample (small grid):")
    #print(jockey_res['policy']['a0'][:6,:6])

    # reneging-only policy (routing uses shortest-queue by default)
    #renege_res = compute_optimal_reneging_policy(params, N_x=20, N_y=20, verbose=True)
    #print("Reneging thresholds (q1,q2):", renege_res['thresholds'])


   
