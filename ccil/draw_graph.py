from argparse import Namespace
from glob import glob
import re
import matplotlib.pyplot as plt
import os
from ccil.imitate import imitate
from ccil.intervention_policy_execution import intervention_policy_execution
from pathlib import Path
from datetime import datetime
import numpy as np
import pickle

from deconfounder.deconfounder import INPUT_PATH


def draw():
    print(os.getcwd())
    data_root_dir = "./data"
    policies_dir = Path(data_root_dir + "/policies")
    policies = [str(os.path.splitext(f)[0]) for f in os.listdir(policies_dir) if re.match(r"24_uniform_(vae|None|ppca|fm)_\[3\]", f)]
    print(policies)
    num_iters = 80
    drop_dims = [3]
    temperature = 10
    confounded = True

    plt.xlabel("num iterations")
    plt.ylabel("rewards")

    saved_d = {}
    
    for policy in policies:
        d = {"policy_name": policy, "num_its": num_iters, 
            "drop_dims": drop_dims, "temperature": temperature, 
            "confounded": confounded, "latent_dim": int(policy.split("_")[-1]), 
            "seed": 3106, "deconfounder": policy.split("_")[2]}

        ns = Namespace(**d)
        its, rewards, past_mean_reward = intervention_policy_execution(ns)
        print(f'Policy: {policy}, Initial Reward: {rewards[0]}')
        saved_d[d['deconfounder']] = rewards
        line_label = f"{d['deconfounder']}_drop_{d['drop_dims']}_latent_{d['latent_dim']}"
        # plt.plot([i for i in range(its)], past_mean_reward, label=line_label)
        plt.plot([i for i in range(its)], smooth(rewards, 20), label=line_label)
    
    # simples = [265.76] * its
    # plt.plot([i for i in range(its)], simples, label="bc with confounder")
            
    plt.legend()
    plt.savefig(f"./graphs/test-{num_iters}-seed-{d['seed']}-drop-{d['drop_dims']}-temp-{d['temperature']}-{datetime.now():%Y%m%d-%H%M%S}.png")

    OUTPUT_PATH = f"./graphs/{num_iters}-seed-{d['seed']}-drop-{d['drop_dims']}-temp-{d['temperature']}-{datetime.now():%Y%m%d-%H%M%S}.pkl"
    with open(OUTPUT_PATH, 'wb') as file:
        pickle.dump(saved_d, file, protocol=pickle.HIGHEST_PROTOCOL)

def smooth(x, w):
    # return np.convolve(x, np.ones(w), 'valid') / w
    return np.convolve(x, np.ones(w), 'same') / w


# def smooth(arr):
    # res = []
    # for i in range(len(arr)):
    #     if i == 0 or i == len(arr)-1:
    #         res.append(arr[i])
    #     else:
    #         res.append((arr[i-1]+arr[i]+arr[i+1])/3)

    # return res

def load_iters(iters):
    INPUT_PATH = "./graphs/150-seed-3106-drop-[3]-temp-10-20221203-222618.pkl"
    with open(INPUT_PATH, 'rb') as fin:
        obj = pickle.load(fin)
    # print(obj.keys())
    font = {'family': 'monospace', 'weight': 'bold', 'size': 10}
    plt.rc('font', **font)
    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations', size=10, weight='bold')
    ax.set_ylabel('Reward', size=10, weight='bold')
    for deconfounder, rewards in obj.items():
        if "fm" in deconfounder:
            line_label  = "PPCA"
        elif "vae" in deconfounder:
            line_label = "VAE"
        else:
            line_label = "No Deconfounder"

        # line_label = f"{deconfounder}_drop_{[3]}_latent_{1}"
        smoothed_rewards = smooth(rewards[:iters], 20)
        ax.plot([i for i in range(len(smoothed_rewards))], smoothed_rewards, label=line_label, lw=2.5, ms=2)
    
    plt.legend(prop = font)
    plt.savefig(f"./graphs/final-{iters}-seed-{3106}-drop-{[3]}-temp-{10}-{datetime.now():%Y%m%d-%H%M%S}.png", bbox_inches='tight')

def draw_bc():
    drop_dims = [3]
    d = {'confounded': True, 'drop_dims': drop_dims, 'latent_dim': -1, 'network': 'simple', 
         'deconfounder': None, "epochs": 10, "num_samples": 300, 'save': False,
         'data_seed': 24
         }

    cutoffs = [i for i in range(50, 300, 50)]
    deconfounders = {"fm": "PPCA", "vae": "VAE", "None": "No Deconfounder"}
    

    font = {'family': 'monospace', 'weight': 'bold', 'size': 10}
    plt.rc('font', **font)
    fig, ax = plt.subplots()
    ax.set_xlabel('Expert Dataset Size', size=10, weight='bold')
    ax.set_ylabel('Reward', size=10, weight='bold')
    for deconfounder in deconfounders:
        print(f"------draw deconfounder {deconfounder}")
        d['deconfounder'] = deconfounder
        rewards = []
        for cutoff in cutoffs:
            print(f"------draw cutoff {cutoff}")
            if "fm" in deconfounder:
                line_label  = "PPCA"
                d['latent_dim'] = 1
            elif "vae" in deconfounder:
                line_label = "VAE"
                d['latent_dim'] = 1
            else:
                line_label = "No Deconfounder"
                d['deconfounder'] = None
            d['cutoff'] = cutoff
            ns = Namespace(**d)
            reward = imitate(ns)
            rewards.append(reward)
            
        # smoothed_rewards = smooth(rewards, 10)
        ax.plot(cutoffs, rewards, label=line_label, lw=2.5, ms=2)
    
    plt.legend(prop = font)
    plt.savefig(f"./graphs/bc-{datetime.now():%Y%m%d-%H%M%S}.png", bbox_inches='tight')


if __name__ == '__main__':
    # draw()
    # load_iters(80)
    draw_bc()