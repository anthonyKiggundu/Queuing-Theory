"""
An Advantage Actor-Critic model of Information gaming
Decision to renege or jockey are guided by the experience from previous customers.
- A finished customer gets a reward or loss measured by the difference between the 
  expected prediction in latency at joining time versus the actual latency experienced.
  Experience is defined by the defined queue state -> length, wait at pose k,service rate+ admission rate
  
@author: RL-A2C extension by Anthony K.
"""

###############################################################################
from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from jockey import *
from RenegeJockey import Observations

import gymnasium as gym
###############################################################################

# Init actor-critic agent
# agent = A2C(gym.make('ImpatientTenantEnv-v1'), random_seed=SEED)

# Init optimizers
# actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
# critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

# Create the Impatient Customer Environment
'''
gym.envs.register(
     id='ImpatientTenantEnv-v1',
     entry_point='gym-examples.gym_examples.envs:ImpatientTenantEnv',
     max_episode_steps=300,
)

env = gym.make('ImpatientTenantEnv-v1') # , render_mode="human")

observation, info = env.reset(seed=42)

print("\n *****>>>>>> ", observation)
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(self, n_features: int, n_actions: int, device: torch.device, critic_lr: float,
                 actor_lr: float, n_envs: int,) -> None:
        
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        critic_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # estimate V(s)
        ]

        actor_layers = [
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(
                32, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
        # We add this due to a runtime error -> mat1 and mat2 must have the same dtype
        self.double()

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        # x = torch.Tensor(x).to(self.device)
        x = torch.tensor(x).to(self.device)
        
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        # print("What is tensor X? ", x, "State Values: ",state_values,"Actions Vectors: ",action_logits_vec)
        return (state_values, action_logits_vec)

    def select_action(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        
        # The args here we pick from the dict x -> the states we take here are
        # size of queue, intensity, serv_rate,wait
        #x_ = list(x.values())
        
        
        # print("\n Who is X ---> ", list(x), np.shape(x))
        state_values, action_logits = self.forward(x)
        action_pd = torch.distributions.Categorical(logits=action_logits)  # implicitly uses softmax
        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()
        
        return (actions, action_log_probs, state_values, entropy)

    def get_losses(self, rewards: torch.Tensor, action_log_probs: torch.Tensor,
                    value_preds: torch.Tensor, entropy: torch.Tensor, masks: torch.Tensor,
                    gamma: float, lam: float, ent_coef: float, device: torch.device,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (n_steps_per_update
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(self, critic_loss: torch.Tensor, actor_loss: torch.Tensor) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()



    '''
    Initialization: Initialize the policy parameters θθ(actor) and the value function parameters ϕϕ (critic).

    Interaction with the Environment: The agent interacts with the environment by taking actions according to the current policy and receiving observations and rewards in return.

    Advantage Computation: Compute the advantage function A(s,a) based on the current policy and value estimates. 

    Policy and Value Updates:   Simultaneously update the actor’s parameters(θ)(θ) using the policy gradient. 
                            The policy gradient is derived from the advantage function and guides the actor 
                            to increase the probabilities of actions that lead to higher advantages.
                            
                            Simultaneously update the critic’s parameters (ϕ)(ϕ)using a value-based method. 
                            This often involves minimizing the temporal difference (TD) error, which is the 
                            difference between the observed rewards and the predicted values.

    ''' 


    '''
        https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/
        Learning from the Advantage versus =>  A2C leverages the advantage function, incorporating the 
                                           difference between the action’s value and the average value of actions in that state. 
                                           This additional information refines the learning process further.
        Learning from the Average =>    The base Actor-Critic method uses the difference between the actual 
                                    reward and the estimated value (critic’s evaluation) to update the actor.
    '''

        
    # We go through the entire state or history and 
    # then extract only those states where the action(Jockey/Renege) == True    
    def states_of_interest(self, states):
        
        #filtered = filter(lambda x: x['Jockey'], states.values())
        
        jockeyed_entries_filtered = []
        renege_states_filtered  = []
        
        #for state in states:
        #    jockeyed_entries_filtered = [x for x in state.values() if state['Jockey']]
        #    renege_states_filtered = [x for x in state.values() if state['Renege']]
            
            #jockeyed_entries_filtered = filter(lambda x: x['Jockey'], state.values())
            #renege_states_filtered = filter(lambda x: x['Renege'], state.values())
        
        #print("\n ALL STATE ========>>>> ", states)
        for state in states: 
            if len(state) == 0:
                continue
            else:
                for key ,value in state.items():
                    if "Jockey" in key and value:
                        jockeyed_entries_filtered.append(list(state.values()))
                    elif "Renege" in key and value:
                        renege_states_filtered.append(list(state.values()))
        
        return (jockeyed_entries_filtered, renege_states_filtered)
    
    
    def change_listOfdict_listOflist(self, test_list):
        # Convert List of Dictionaries to List of Lists
        # Using list comprehension
        # print("\n Before: ", test_list)
        res = []
        for idx, sub in enumerate(test_list, start = 0):
            if idx == 0:
                res.append(list(sub.keys()))
                res.append(list(sub.values()))
            else:
                res.append(list(sub.values()))
                
        # Remove any empty lists
        list2 = [x for x in res if x != []]  
                      
        # print("\n ---------------------> ", res, " ----------- ", list2)
        return list2 # np.array(res)
    

    def train_agent(self, env, n_envs, n_updates, n_steps_per_update, agent):   
                # environment hyperparams
        #n_envs = 10
        #n_updates = 1000
        #n_steps_per_update = 128
        #randomize_domain = False
        
        # agent hyperparams
        gamma = 0.999
        lam = 0.95  # hyperparameter for GAE
        ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
        #actor_lr = 0.001
        #critic_lr = 0.005
        
        # Note: the actor has a slower learning rate so that the value targets become
        # more stationary and are theirfore easier to estimate for the critic
        
        # environment setup
        
        #envs = gym.vector.make("ImpatientTenantEnv-v1.0",critic_lr=critic_lr, actor_lr=actor_lr, random_seed=42,
        #                   num_envs=n_envs, max_episode_steps=600)
        
        # env_fns = [lambda: gym.make("ImpatientTenantEnv-v1.0", max_episode_steps=600)] * n_envs # gym.make("ImpatientTenantEnv-v1.0")
        
        # envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)
        
        
        #obs_shape = envs.single_observation_space.shape[0]
        #action_shape = envs.single_action_space.n
        
        # set the device
        #use_cuda = False
        #if use_cuda:
        #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #else:
        #    device = torch.device("cpu")
        
        # init the agent
        #agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)
             
        # create a wrapper environment to save episode returns and episode lengths
        envs_wrapper = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_envs * n_updates)
        
        critic_losses = []
        actor_losses = []
        entropies = []
        lst_new_state = []
        # curr_state = RequestQueues.get_obs()
        #curr_observ = Observations.get_obs()
        
        
        # use tqdm to get a progress bar for training
        for sample_phase in tqdm(range(n_updates)):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically
        
            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
            # print("\n --------------->>>>>> ", ep_value_preds, len(ep_value_preds))
            ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
            ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
            #print("\n ++++++++++++++++++++++++++++++ ", ep_value_preds, len(ep_value_preds[0]), ep_rewards, len(ep_rewards), 
            #      ep_action_log_probs, len(ep_action_log_probs))
            masks = torch.zeros(n_steps_per_update, n_envs, device=device)
        
            # at the start of training reset all envs to get an initial state
            if sample_phase == 0:
                states, info = envs_wrapper.reset(seed=42)
                #if len(states[0]) == 0:
                #    lst_new_state.append(states[1:])
                #else:
                #    lst_new_state.append(states[0:])
                # print("\n ****** The STATES:", states[1])
            
            #print("\n States Original: ",states, "\n ======================> ",self.change_listOfdict_listOflist(states))
            
            # jockey_states, renege_states = self.states_of_interest(states)
                
            #print("\n ****** The FILTERED STATES: ",jockey_states, renege_states)
            #if len(jockey_states) > 0:
            #    states_in_true = jockey_states # [jockey_states, renege_states]
            #else:
            #   states_in_true = renege_states
            
            new_states = self.change_listOfdict_listOflist(states)
            n_envs = new_states
            # lst_new_state = self.change_listOfdict_listOflist(lst_new_state)
            # play n steps in our parallel environments to collect data
            counter_ep_value = 0 
            counter_action_value = 0
            for step in range(n_steps_per_update):
                # select an action A_{t} using S_{t} as input for the agent
                actions, action_log_probs, state_value_preds, entropy = agent.select_action(new_states)
        
                #print("\n ..... Actions: ", ep_value_preds[step] , state_value_preds[:len(ep_value_preds[step])] ,
                #      len(state_value_preds), "\n ======== STATES ========== : ",new_states,len(new_states)) # "States: ",states
                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                # truncated, -> removed this from below because it was not  needed and 
                # throws an error from the base code in ../../wrappers/time_limits.py
                states, rewards, terminated, truncated, infos = envs_wrapper.step(actions.cpu().numpy())
                
        
                ep_value_preds[step] = torch.squeeze(state_value_preds[:len(ep_value_preds[step])]) # counter:len(ep_value_preds[step])] ...expand( len(ep_value_preds[step]), 1)) #
                ep_rewards[step] = torch.tensor(rewards, device=device)
                ep_action_log_probs[step] = action_log_probs[:len(ep_action_log_probs[step])]
        
                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor(terminated) #[not term for term in terminated])
                counter_ep_value = counter_ep_value + len(ep_value_preds[step])
                counter_action_value = counter_action_value + len(ep_action_log_probs[step])
                print("\n ********************* End iterations ****************** ", counter_ep_value, counter_action_value)
        
            # calculate the losses for actor and critic
            critic_loss, actor_loss = agent.get_losses(ep_rewards, ep_action_log_probs, ep_value_preds,
                entropy, masks, gamma, lam, ent_coef, device,)
        
            # update the actor and critic networks
            agent.update_parameters(critic_loss, actor_loss)
        
            # log the losses and entropy
            critic_losses.append(critic_loss.detach().cpu().numpy())
            actor_losses.append(actor_loss.detach().cpu().numpy())
            entropies.append(entropy.detach().mean().cpu().numpy())
            
            
    def run (self):
        
        """ play a couple of showcase episodes """
        
        n_showcase_episodes = 3
        
        for episode in range(n_showcase_episodes):
            print(f"starting episode {episode}...")
        
            # create a new sample environment to get new random parameters
            # Here we need new arrival rates and service rates for the different episodes
            
            #if randomize_domain:
            #    env = gym.make(
            #        "ImpatientTenantEnv-v1",
            #        render_mode="human",
            #        gravity=np.clip(np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99,
            #            a_max=-0.01),
            #            enable_wind=np.random.choice([True, False]
            #        ),
            #        wind_power=np.clip(
            #            np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
            #        ),
            
            #        turbulence_power=np.clip(
            #            np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
            #        ),
            #        max_episode_steps=500,
            #                )
            #else:
            
            env = gym.make("ImpatientTenantEnv-v1", render_mode="human", max_episode_steps=500)
        
            # get an initial state
            state, info = env.reset()
        
            # play one episode
            done = False
            while not done:
                # select an action A_{t} using S_{t} as input for the agent
                with torch.no_grad():
                    action, _, _, _ = agent.select_action(state[None, :])
        
                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                state, reward, terminated, truncated, info = env.step(action.item())
        
                # update if the environment is done
                done = terminated or truncated
        env.close()
        

