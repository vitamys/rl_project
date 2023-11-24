
import simglucose
import os
import gym
import tqdm
import numpy as np
import json
from tqdm import tqdm
from gym.wrappers import FlattenObservation
import datetime as dt
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
from collections import namedtuple


Observation = namedtuple('Observation', ['CGM'])

from copy import deepcopy


import torch as th
import torch.nn.functional as F

class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""
    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)
        
    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)

class Actor(th.nn.Module):

    def __init__(self, state_dimension, action_dimension, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.cnn = th.nn.Sequential(
            # ? signals channels
            th.nn.Conv1d(2, 4, 5, bias=True, padding='same'),
            th.nn.LeakyReLU(),
            th.nn.Conv1d(4, 8, 5, bias=True, padding='same'),
            th.nn.LeakyReLU(),
            th.nn.Conv1d(8, 16, 5, bias=True, padding='same'),
            th.nn.LeakyReLU(),
        )

        self.avgpool =  th.nn.AdaptiveAvgPool1d(1)

        self.classifier =  th.nn.Sequential(
            th.nn.Linear(16+state_dimension, 32),
            th.nn.LeakyReLU(),
            th.nn.Linear(32, 32),
            th.nn.LeakyReLU(),
            th.nn.Linear(32, action_dimension),
            th.nn.Sigmoid()
        )

    def forward(self, state):
        x = state.unflatten(2, (2,12))
        x = x.squeeze(1)
        x = self.cnn(x)
        x = self.avgpool(x)
        #x = th.reshape(x, (-1, 16)) # flatten after avgpool
        # batchsize , 1, 16
        # batchsize , 16, 1
        x = x.transpose(1, 2)
        x = th.cat([x, state], 2)
        return self.max_action * self.classifier(x)

class Critic(th.nn.Module):

    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()
        self.features = th.nn.Sequential(
            # ? signals channels
            th.nn.Conv1d(2, 4, 5, bias=True, padding='same'),
            th.nn.LeakyReLU(),
            th.nn.Conv1d(4, 8, 5, bias=True, padding='same'),
            th.nn.LeakyReLU(),
            th.nn.Conv1d(8, 16, 5, bias=True, padding='same'),
            th.nn.LeakyReLU(),
        )
        self.avgpool =  th.nn.AdaptiveAvgPool1d(1)

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        # 16 + action_dimension, output of the cnn
        self.classifier =  th.nn.Sequential(
            th.nn.Linear(16+action_dimension+state_dimension, 32),
            th.nn.LeakyReLU(),
            th.nn.Linear(32, 32),
            th.nn.LeakyReLU(),
            th.nn.Linear(32,1),
        )   

    def forward(self, state, action):
        x = state.unflatten(2, (2,12))
        x = x.squeeze(1)
        cnn = self.features(x)
        cnn = self.avgpool(cnn)
        cnn = cnn.transpose(1, 2)
        x = th.cat([cnn, action], 2)
        x  = th.cat([x, state], 2)
        return  self.classifier(x)

class ReplayBuffer(object):

    def __init__(self, state_dimension, action_dimension, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dimension))
        self.action = np.zeros((max_size, action_dimension))
        self.next_state = np.zeros((max_size, state_dimension))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.max_episode_duration = 0
        self.max_reached = False
        self.episode_durations = []

    def add(self, state, action, next_state, reward, done, episode_duration):
        if self.max_reached and episode_duration < 0.9 * np.mean(self.episode_durations):
            # keep only the longest episodes
            return
        if self.ptr +1 == self.max_size:
            self.max_reached = True
            print('max_reached')

        self.episode_durations.append(episode_duration)

        self.state[self.ptr] = np.array(state)
        self.action[self.ptr] = np.array(action)
        self.next_state[self.ptr] = np.array(next_state)
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            th.FloatTensor(self.state[ind]).to(self.device),
            th.FloatTensor(self.action[ind]).to(self.device),
            th.FloatTensor(self.next_state[ind]).to(self.device),
            th.FloatTensor(self.reward[ind]).to(self.device),
            th.FloatTensor(self.not_done[ind]).to(self.device)
        )

class DDPGAgent(object):

    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters())
        self.actor_optimizer.lr = 1e-5
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters())
        self.critic_optimizer.lr = 1e-5

    def select_action(self, state):
        state = th.FloatTensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, filename):
        th.save(self.critic.state_dict(), filename + '_critic')
        th.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
        th.save(self.actor.state_dict(), filename + '_actor')
        th.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')

    def load_checkpoint(self, filename):
        self.critic.load_state_dict(
            th.load(
                filename + "_critic",
                map_location=th.device('cpu')
            )
        )
        self.critic_optimizer.load_state_dict(
            th.load(
                filename + "_critic_optimizer",
                map_location=th.device('cpu')
            )
        )
        self.critic_target = deepcopy(self.critic)
        self.actor.load_state_dict(
            th.load(
                filename + "_actor",
                map_location=th.device('cpu')
            )
        )
        self.actor_optimizer.load_state_dict(
            th.load(
                filename + "_actor_optimizer",
                map_location=th.device('cpu')
            )
        )
        self.actor_target = deepcopy(self.actor)

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # reshape to (batch_size, 1, -1)
        next_state = next_state.reshape(batch_size, 1, -1)
        state = state.reshape(batch_size, 1, -1)
        action = action.reshape(batch_size, 1, -1)

        reward = reward.reshape(batch_size, 1, -1)
        not_done = not_done.reshape(batch_size, 1, -1)

        target_q = self.critic_target(next_state, self.actor_target(next_state))

        target_q = reward + (not_done * self.discount * target_q).detach()

        current_q = self.critic(state, action)

        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        DDPGAgent.soft_update(self.critic, self.critic_target, self.tau)
        DDPGAgent.soft_update(self.actor, self.actor_target, self.tau)

def evaluate_policy(policy, env_name, seed, eval_episodes=10, render=False, max_timesteps=1000):
    eval_env = gym.make(env_name)
    eval_env = FlattenObservation(eval_env)
    eval_env = FlattenAction(eval_env)

    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, info = eval_env.reset()
        done = False
        for _ in range(max_timesteps):
            if render:
                eval_env.render(mode='human')
            state = np.array(state)
            state = state[None,:]
            state = state[None,:]
            action = policy.select_action(state)
            print(action)
            state, reward, done, _, info = eval_env.step(action)
            avg_reward += reward
            if done:
                break
    avg_reward /= eval_episodes
    return avg_reward

class Trainer:

    def __init__(self, config_file, enable_logging):
        self.enable_logging = enable_logging
        self.config = Trainer.parse_config(config_file)
        self.env = gym.make(self.config['env_name'])
        self.env = FlattenObservation(self.env)
        self.env = FlattenAction(self.env)
        self.apply_seed()
        self.state_dimension = self.env.observation_space.shape[0]
        self.action_dimension = self.env.action_space.shape[0]
        print(f"State dimension: {self.state_dimension}")
        print(f"Action dimension: {self.action_dimension}")

        self.max_action = float(self.env.action_space.high[0])
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.agent = DDPGAgent(
            state_dim=self.state_dimension, action_dim=self.action_dimension,
            max_action=self.max_action, device=self.device,
            discount=self.config['discount'], tau=self.config['tau']
        )
        self.save_file_name = f"DDPG_{self.config['env_name']}_{self.config['seed']}"

        # Load weights
        #self.agent.load_checkpoint("models/"+self.save_file_name)

        self.memory = ReplayBuffer(self.state_dimension, self.action_dimension)
        if self.enable_logging:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter('./logs/' + self.config['env_name'] + '/')
        try:
            os.mkdir('./models')
        except Exception as e:
            pass

    @staticmethod
    def parse_config(json_file):
        with open(json_file, 'r') as f:
            configs = json.load(f)
        return configs

    def apply_seed(self):
        self.env.seed(self.config['seed'])
        th.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def train(self):
        ctrller = BBController()
        #self.env.render(mode="human")
        done = False
        episode_reward = 0
        evaluations = []
        episode_rewards = []
        episode_durations = []
        reward = 0
        max_episode_timesteps = 20 * 24 * 7
        expl_noise = self.config["expl_noise"]
        for ts in tqdm(range(1, int(self.config['num_episodes']) + 1)):
            state, info = self.env.reset()

            for episode_duration in range(max_episode_timesteps):

                if ts < self.config['start_time_step']:
                    #action = self.env.action_space.sample()
                    obs = Observation(state[11])

                    ctrl_action = ctrller.policy(obs, reward, done, **info)
                    action = [ctrl_action.basal + ctrl_action.bolus]
                else:
                    # create mini-batch
                    state_ = np.array(state)
                    state_ = state_[None, :]
                    state_ = state_[None, :]
                    action = (
                            self.agent.select_action(state_) + np.random.normal(
                        0, self.max_action * expl_noise,
                        size=self.action_dimension
                    )
                    ).clip(0, self.max_action)
                    
                next_state, reward, done, _, info = self.env.step(action)
                self.memory.add(state, action, next_state, reward, float(done), episode_duration)
                state = next_state
                episode_reward += reward
                if done:
                    break

            if expl_noise > 0.001:
                expl_noise *= self.config["expl_noise_decay"]
            episode_durations.append(episode_duration)
            episode_rewards.append(episode_reward)
            episode_reward = 0
            self.agent.train(self.memory, self.config['batch_size'])
            
            if ts > self.config['start_time_step'] and ts % self.config["evaluate_frequency"] == 0:
                self.agent.save_checkpoint(f"./models/{self.save_file_name}")
                mean_reward = np.mean(episode_rewards[-self.config["evaluate_frequency"]:])
                print(f"Episode: {ts} \t mean reward: {mean_reward}")
                print(f"Episode duration: {np.mean(episode_durations[-self.config['evaluate_frequency']:])}")
                #evaluations.append(evaluate_policy(self.agent, self.config['env_name'], self.config['seed']))

        return episode_rewards, evaluations

def paper_reward_function(BG_last_hour):
    G = BG_last_hour[-1]
    if G >= 90 and G <= 140:
        return 1
    if (G >= 70 and G < 90) or (G > 140 and G <= 180):
        return 0.1
    if G>= 180 and G <= 300:
        return -0.4-(G-180)/200
    if G >= 30 and G < 70:
        return -0.6-(70-G)/100
    else:
        return -1


def register_gym():

    np_random = np.random.default_rng(seed=42)
    scenarios = []

    for i in range(100):
        minute = np_random.integers(0, 19)*3
        hour = np_random.integers(0, 24) 
        start_time = dt.datetime(2018, 1, 1, hour, minute, 0) 
        scenarios.append(CustomScenario(start_time=start_time, scenario=[]))

    gym.envs.register(  
        id='simglucose-basal',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': ['adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
                                'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010'],
                'reward_fun': paper_reward_function,
                'custom_scenario': scenarios,
                'history_length': 12},
)
