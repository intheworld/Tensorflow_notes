import argparse

import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import SAC
import os


model_dir = ".output/models"
log_dir = ".output/logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env: Env):
    model = SAC("MlpPolicy", env, verbose=1, device='cuda',
                tensorboard_log=log_dir)
    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/SAC_{TIMESTEPS*iters}")

def test(env: Env, path_to_model: str):
    model = SAC.load(path_to_model)
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        env.render()
        if done:
                extra_steps -= 1
                if extra_steps < 0:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        gym_env = gym.make(args.gymenv, render_mode=None)
        train(gym_env)

    if args.test:
        if os.path.isfile(args.test):
            gym_env = gym.make(args.gymenv, render_mode='human')
            test(gym_env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
