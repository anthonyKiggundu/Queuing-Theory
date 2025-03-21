import gym
import A2C as a2c
import torch.optim as optim

def main():
    LR = .01  # Learning rate
    SEED = None  # Random seed for reproducibility
    MAX_EPISODES = 10000  # Max number of episodes

    # Init actor-critic agent
    agent = A2C(gym.make('UngeduldigenKunden'), critic_lr=LR, actor_lr=LR, random_seed=SEED)

    r = []  # Array containing total rewards
    avg_r = 0  # Value storing average reward over last 100 episodes
    
    
    
if __name__ == "__main__":
    tenantEnv = ImpatientTenantEnv()
