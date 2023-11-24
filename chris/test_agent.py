from stuff import *

register_gym()

config = Trainer.parse_config("basal.json")
env = gym.make(config['env_name'])
env = FlattenObservation(env)
env = FlattenAction(env)
state_dimension = env.observation_space.shape[0]
action_dimension = env.action_space.shape[0]
print(f"State dimension: {state_dimension}")
print(f"Action dimension: {action_dimension}")

max_action = float(env.action_space.high[0])
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
agent = DDPGAgent(
    state_dim=state_dimension, action_dim=action_dimension,
    max_action=max_action, device=device,
    discount=config['discount'], tau=config['tau']
)

obs, info = env.reset()

agent.load_checkpoint("models/DDPG_simglucose-basal_0")

#print(obs)
#print(agent.select_action(np.array(obs)))

evaluate_policy(agent, "simglucose-basal", 0, render=True)