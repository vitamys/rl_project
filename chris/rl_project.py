from stuff import *


register_gym()

trainer = Trainer(config_file="basal.json", enable_logging=False)

episode_rewards, evaluations = trainer.train()

import csv

with open('episode_rewards.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(episode_rewards)

with open('evaluations.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(evaluations)
