from pprint import pprint

import gym
import numpy as np
import argparse
from time import perf_counter

import torch
from sklearn.linear_model import Ridge
from torch.distributions import Bernoulli

from ccil.environments.hopper import HopperStateEncoder
# from ccil.environments.mountain_car import MountainCarStateEncoder
from ccil.utils.data import Trajectory
from ccil.utils.policy_runner import PolicyRunner, FixedMaskPolicyAgent, \
                                        run_fixed_mask
from ccil.utils.utils import data_root_path
from ccil.imitate import load_dataset

from pytorch_lightning import seed_everything


def sample(weights, temperature):

    return Bernoulli(logits=torch.from_numpy(weights) / temperature).sample().long().numpy()


def linear_regression(masks, rewards, alpha=1.0):
    model = Ridge(alpha).fit(masks, rewards)
    return model.coef_, model.intercept_


class SoftQAlgo:
    def __init__(
            self,
            num_dims,
            reward_fn,
            its,
            temperature=1.0,
            device=None,
            evals_per_it=1,
    ):
        self.num_dims = num_dims
        self.reward_fn = reward_fn
        self.its = its
        self.device = device
        self.temperature = lambda t: temperature
        self.evals_per_it = evals_per_it

    def run(self):
        t = self.temperature(0)
        weights = np.ones(self.num_dims)

        # trace = []
        masks = []
        rewards = []
        past_mean_reward = []
        best_reward = 0
        best_mask = np.ones(self.num_dims)
        i = 0

        for it in range(self.its):
            # print(f"--steps: {i}--")
            start = perf_counter()
            mask = sample(weights, t)
            reward = np.mean([self.reward_fn(mask) \
                for _ in range(self.evals_per_it)])
            masks.append(mask)
            rewards.append(reward)
            past_mean_reward.append(np.mean(rewards))
            if reward > best_reward:
                best_mask  = mask
                best_reward = reward

            weights, _ = linear_regression(masks, rewards, alpha=1.0)

            pprint({
                    "it": it,
                    "reward": reward,
                    "mask": mask,
                    "weights": weights,
                    "mode": (np.sign(weights).astype(np.int64) + 1) // 2,
                    "time": perf_counter() - start,
                    "past_mean_reward": past_mean_reward[-1],
                }
            )
            i += 1
            

        return best_mask, best_reward, rewards, past_mean_reward


def intervention_policy_execution(args):
    seed = args.seed
    seed_everything(seed)
    policy_save_dir = data_root_path / 'policies'
    if args.policy_name:
        policy_path = policy_save_dir / f"{args.policy_name}.pkl"
    else:
        policy_paths = policy_save_dir.glob('*.pkl')
        if not policy_paths:
            raise RuntimeError("No policy found")
        policy_path = next(iter(sorted(policy_paths, reverse=True)))
    # policy_model = torch.load(policy_path)
    policy_model = torch.load(policy_path, map_location=torch.device('cpu'))
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # policy_model = policy_model.to(device)
    # print(f"Loaded policy from {policy_path} to device: {device}\n")

    env = gym.make("Hopper-v2")
    env.reset(seed=seed)
    dataset, state_encoder = load_dataset(args.confounded, args.drop_dims, 
                                args.latent_dim, args.deconfounder)
    # state_encoder = HopperStateEncoder(random=False)

    def run_step(mask):
        env.reset(seed=seed)
        # seed_everything(24)
        trajectories = run_fixed_mask(env, policy_model, state_encoder, mask, 1)
        return Trajectory.reward_sum_mean(trajectories)

    input_dim = state_encoder.step(dataset[0].states[0, -1].numpy(), None).shape[-1]
    best_mask, best_reward, rewards, past_mean_reward = SoftQAlgo(input_dim, run_step, args.num_its, 
                                        temperature=args.temperature).run()

    # best_mask = trace[-1]['mode']

    # env.reset(seed=42)
    trajectories = run_fixed_mask(env, policy_model, state_encoder, best_mask, 1)
    print(f"Final mask {best_mask.tolist()}")
    print(f"Final reward {Trajectory.reward_sum_mean(trajectories)}")
    print(f"Best Reward:{best_reward}")

    return args.num_its, rewards, past_mean_reward


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--policy_name', help="Policy save filename")
    parser.add_argument('--drop_dims', nargs='+', type=int, default=[])
    parser.add_argument('--latent_dim', type=int, default=-1)
    parser.add_argument('--num_its', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=10)
    parser.add_argument('--deconfounder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=24)
    intervention_policy_execution(parser.parse_args())


if __name__ == '__main__':
    main()
