import gymnasium as gym
from stable_baselines3 import SAC
import os


model_dir = ".output/models"
log_dir = ".output/logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env_name = 'Humanoid-v4'

gymenv = gym.make(env_name, render_mode=None)

model = SAC("MlpPolicy", gymenv, verbose=1, device='cuda', 
            tensorboard_log=log_dir)
model.learn(total_timesteps = 25000)
model.save(f"{model_dir}/sac_humanoid")

del model
model = SAC.load(f"{model_dir}/sac_humanoid")

gymenv = gym.make(env_name, render_mode='human')
obs = gymenv.reset()[0]
done = False
extra_steps = 500
while True:
    action, _ = model.predict(obs)
    obs, _, done, _, _ = gymenv.step(action)
    gymenv.render()
    if done:
            extra_steps -= 1

            if extra_steps < 0:
                break

