from ActorCritic import A2C as a2c
# from stable_baselines3.common.env_checker import check_env
from gymnasium.utils.env_checker import check_env
from RenegeJockey import *
import torch.optim as optim

#import gym
import gymnasium as gym
import sys
import gym_examples
import torch

'''
gym.envs.register(
     id='ImpatientTenantEnv-v1',
     entry_point='gym-examples.gym_examples.envs:ImpatientTenantEnv',
     max_episode_steps=300,
)

env = gym.make('ImpatientTenantEnv-v1') # , render_mode="human")
'''

class Queues(object):
    def __init__(self, num_of_queues=2):
        super().__init__()
        
        self.num_of_queues = num_of_queues
        self.arrival_rates = [3,5,7,9,11,13,15]
        rand_idx = random.randrange(len(self.arrival_rates))
        self.sampled_arr_rate = self.arrival_rates[rand_idx]
        
        self.dict_queues = self.generate_queues()
        self.dict_servers = self.queue_setup_manager()
        
        
    def queue_setup_manager(self):
                
        # deltalambda controls the difference between the service rate of either queues    
        deltaLambda=random.randint(1, 2)
        
        serv_rate_one=self.sampled_arr_rate + deltaLambda 
        serv_rate_two=self.sampled_arr_rate - deltaLambda

        _serv_rate_one=serv_rate_one / 2
        _serv_rate_two=serv_rate_two / 2
        
        self.dict_servers["Server1"] = _serv_rate_one
        self.dict_servers["Server2"] = _serv_rate_two
        
        print("\n Current Arrival Rate:", self.sampled_arr_rate, "Server1:", _serv_rate_one, "Server2:", _serv_rate_two) 
        
        return self.dict_servers        
    
    
    def generate_queues(self):
        
        for i in range(self.num_of_queues):
            code_string = "Server%01d" % (i+1)
            queue_object = np.array([])
            self.dict_queues.update({code_string: queue_object})

        return self.dict_queues


    def get_arrivals_rates(self):

        return self.sampled_arr_rate
    

# observation, info = env.reset(seed=42)

def main():
    LR = .01  # Learning rate
    SEED = None  # Random seed for reproducibility
    MAX_EPISODES = 10000  # Max number of episodes

    
    n_envs = 9 # 10
    n_updates = 1000
    n_steps_per_update = 128
    randomize_domain = False

    # agent hyperparams
    gamma = 0.999

    lam = 0.95  # hyperparameter for GAE
    ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
    actor_lr = 0.001
    critic_lr = 0.005

        # Note: the actor has a slower learning rate so that the value targets become
        # more stationary and are theirfore easier to estimate for the critic

        # environment setup
    if randomize_domain:
        env = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "ImpatientTenantEnv-v1",
                    gravity=np.clip(
                        np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                    ),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(
                        np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                    ),
                    turbulence_power=np.clip(
                        np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                    ),
                    max_episode_steps=600,
                )
                for i in range(n_envs)
            ]
        )

    else:
        gym.envs.register(
            id='ImpatientTenantEnv-v1.0',
            entry_point='gym-examples.gym_examples.envs:ImpatientTenantEnv',
            max_episode_steps=300,
        )
        
        env = gym.vector.make('ImpatientTenantEnv-v1.0', num_envs=n_envs)
        #env = gym.make_vec('ImpatientTenantEnv-v1.0', num_envs=n_envs)
        
        check_env(env, skip_render_check=True)
        
        observation, info = env.reset(seed=42)
        
        #print("\n *************** Making environment **************** ", observation)

    #    envs = gym.vector.make("ImpatientTenantEnv-v1",critic_lr=critic_lr, actor_lr=actor_lr, random_seed=SEED,
    #                                num_envs=n_envs, max_episode_steps=600)

    
    obs_shape = len(env.observation_space) #.shape[0]
    action_shape = len(env.action_space.n)

    # set the device
    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # init the agent
    agent = a2c(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

    # OrderedDict
    #RequestQueue.run()    
    
    # Run one episode to get
    # the model would need to be "run" in the environment.

    # Init actor-critic agent
    #agent = a2c() #gym.make('ImpatientTenantEnv-v1'), critic_lr=LR, actor_lr=LR, random_seed=SEED)
    #a2c.setup()
    agent.train_agent(env, n_envs, n_updates, n_steps_per_update, agent)
    agent.run()
    agent.update_parameters()

    # Init optimizers
    actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
    critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

    '''
    r = []  # Array containing total rewards
    avg_r = 0  # Value storing average reward over last 100 episodes
    
    '''
    
if __name__ == "__main__":
    main()
    #tenantEnv = ImpatientTenantEnv()
