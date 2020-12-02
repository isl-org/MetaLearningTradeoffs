import numpy as np
import os
import json
import csv
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import betainc

DIR = os.getcwd()+'/data'
OUTPUT = os.getcwd()+'/results'
if os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

algos = ['ppo', 'promp', 'trpo', 'trpomaml']
environment_mode_pairs = [['walker', 'params-interpolate'], ['walker', 'goal-interpolate'],
        ['cheetah', 'goal-interpolate'], ['hopper', 'params-interpolate'],
        ['metaworld', 'ml1-push'], ['metaworld', 'ml1-reach'],
        ['metaworld', 'ml10'], ['metaworld', 'ml45']]
seeds = [1, 2, 3, 4, 5]
checkpoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
updates = [0, 1, 2, 3, 4, 5]

reward_means = np.zeros((len(algos), len(environment_mode_pairs), len(seeds), len(checkpoints), len(updates)))
reward_stderrs = np.zeros((len(algos), len(environment_mode_pairs), len(seeds), len(checkpoints), len(updates)))

for a in range(len(algos)):
    alg = algos[a]
    alg_dir = os.path.join(DIR, alg)
    if os.path.isdir(alg_dir):
        for b in range(len(environment_mode_pairs)):
            pair = environment_mode_pairs[b]
            environment = pair[0]
            mode = pair[1]
            mode_dir = os.path.join(alg_dir, mode)
            env_dir = os.path.join(mode_dir, environment)
            if os.path.isdir(env_dir):
                for c in range(len(seeds)):
                    run = seeds[c]
                    run_dir = os.path.join(env_dir, 'run_'+str(run))
                    if os.path.isdir(run_dir):
                        hyperparam_file = os.path.join(run_dir, 'params.json')

                        # Extract hyperparams
                        if os.path.exists(hyperparam_file):
                            with open(hyperparam_file, 'r') as hyperparams:
                                try:
                                    hyperparam_dict = json.load(hyperparams)
                                    checkpoint_gap = hyperparam_dict['checkpoint_gap']
                                except (ValueError, json.decoder.JSONDecodeError) as e:
                                    print("WARNING: exception reading params.json at %s" % run_dir)
                                    pass
                        else:
                            print("WARNING: Missing hyperparameters file at %s, skipping" % run_dir)
                            continue

                        # Get evaluation stats
                        for d in range(len(checkpoints)):
                            checkpoint = checkpoints[d]
                            checkpoint_dir = os.path.join(run_dir, 'checkpoint_'+str(checkpoint))
                            if os.path.isdir(checkpoint_dir):
                                evaluation_file = os.path.join(checkpoint_dir, 'evaluation.json')
                                print("Reading %s" % evaluation_file)
                                if os.path.exists(evaluation_file):
                                    with open(evaluation_file, 'r') as evaluation:
                                        trials = []
                                        for line in evaluation:
                                            try:
                                                trials.append(json.loads(line))
                                            except (ValueError, json.decode.JSONDecodeError) as e:
                                                print("WARNING: exception reading line %s; \n%s" % (line, e))
                                                pass

                                    for f in range(len(updates)):
                                        reward_means[a,b,c,d,f] = float(np.mean(np.asarray([data['avg_update_reward'][f] for data in trials])))
                                        reward_stderrs[a,b,c,d,f] = float(np.std(np.asarray([data['avg_update_reward'][f] for data in trials])))/np.sqrt(len(trials))
                                else:
                                    print("WARNING: Evaluation file is missing at %s, skipping" % checkpoint_dir)
                            else:
                                print("WARNING: Unexpected file at %s" % checkpoint_dir)
                    else:
                        print("WARNING: Unexpected file at %s" % run_dir)
            else:
                print("WARNING: Unexpected file at %s" % env_dir)
    else:
        print("WARNING: Unexpected file at %s" % alg_dir)


# Average the results over seeds

print("Computing average test rewards over seeds")
avg_reward_means = np.zeros((len(algos), len(environment_mode_pairs), len(checkpoints), len(updates)))
for a in range(len(algos)):
    for b in range(len(environment_mode_pairs)):
        for d in range(len(checkpoints)):
            for f in range(len(updates)):
                avg_reward_means[a,b,d,f] = np.mean(reward_means[a,b,:,d,f])
            with open(os.path.join(OUTPUT, 'output.txt'), 'a') as result:
                result.write("Average rewards for {} environment {} mode {} at checkpoint {}".format(algos[a], environment_mode_pairs[b][0], environment_mode_pairs[b][1], checkpoints[d]))
                result.write("\n")
                result.write(str(avg_reward_means[a,b,d,:]))
                result.write("\n")

print("Computing standard errors of average test rewards over seeds")
avg_reward_stderrs = np.zeros((len(algos), len(environment_mode_pairs), len(checkpoints), len(updates)))
for a in range(len(algos)):
    for b in range(len(environment_mode_pairs)):
        for d in range(len(checkpoints)):
            for f in range(len(updates)):
                varest = np.var(reward_means[a,b,:,d,f])
                varest += np.mean(reward_stderrs[a,b,:,d,f]**2)
                avg_reward_stderrs[a,b,d,f] = np.sqrt(varest/len(seeds))

# Construct contour plots

print("Contour plots of whether MAML is better than DRS, as functions of the checkpoint and update")

def welch_test(target_mean, target_std, target_n, comp_mean, comp_std, comp_n):
    # Computes a Welch test to see if the comparitor sample is larger than the target
    # sample. This is a one-sided test.

    nu = ((target_std**2/target_n + comp_std**2/comp_n)**2/
            (target_std**4/target_n/(target_n-1)+ comp_std**4/comp_n/(comp_n-1)))
    t_stat = ((target_mean-comp_mean)
            /np.sqrt(target_std**2/target_n+comp_std**2/comp_n))

    return 0.5*betainc(nu/2,1/2,nu/(t_stat**2+nu))

def get_tradeoff_map(task_avg_reward_means, task_avg_reward_stderrs):

    num_checkpoints = task_avg_reward_means.shape[2]
    num_updates = task_avg_reward_means.shape[3]
    num_seeds = 5 # From above
    bin_table = np.zeros((num_checkpoints, num_updates))
    fin_table = np.zeros((num_checkpoints, num_updates))

    for c in range(num_checkpoints):
        for u in range(num_updates):
            fin_table[c,u] = welch_test(task_avg_reward_means[1,c,u], np.sqrt(num_seeds)*task_avg_reward_stderrs[1,c,u], num_seeds, task_avg_reward_means[0,c,u], np.sqrt(num_seeds)*task_avg_reward_stderrs[0,c,u], num_seeds)

            if avg_reward_means[0,task,c,u] > avg_reward_means[1,task,c,u]:
                bin_table[c,u] = 1
            else:
                bin_table[c,u] = -1
    return bin_table*(1-2*fin_table), bin_table

yint = range(0,21,2)
for b in range(len(environment_mode_pairs)):
    mm, bt = get_tradeoff_map(avg_reward_means[0:2,b,:,:], avg_reward_stderrs[0:2,b,:,:])
    plt.figure(figsize=(5,2.5))
    plt.contourf(np.array(checkpoints).astype(int), np.array(updates).astype(int),  mm.T, np.linspace(-1.01, 1.01, 50),vmin=-1,vmax=1, cmap='RdBu_r')
    plt.xticks(yint)
    plt.xlabel('Meta-Training Budget')
    plt.ylabel('Test Updates')
    plt.tight_layout()
    plt.savefig('ppo_promp_{}.pdf'.format('-'.join(environment_mode_pairs[b])))
    plt.close(fig)

    mm, bt = get_tradeoff_map(avg_reward_means[2:4,b,:,:], avg_reward_stderrs[2:4,b,:,:])
    plt.figure(figsize=(5,2.5))
    plt.contourf(np.array(checkpoints).astype(int), np.array(updates).astype(int),  mm.T, np.linspace(-1.01, 1.01, 50),vmin=-1,vmax=1, cmap='RdBu_r')
    plt.xticks(yint)
    plt.xlabel('Meta-Training Budget')
    plt.ylabel('Test Updates')
    plt.tight_layout()
    plt.savefig('trpo_trpomaml_{}.pdf'.format('-'.join(environment_mode_pairs[b])))
    plt.close(fig)

# Construct reward plots

print("Plotting rewards as function of checkpoint, after one update")
for b in range(len(environment_mode_pairs)):
    environment = environment_mode_pairs[b][0]
    mode = environment_mode_pairs[b][1]
    if environment == 'metaworld':
        gap = 1000*20*10*150 # Number of time steps between each checkpoint
    else:
        gap = 100*40*20*200

    fig = plt.figure()
    plt.plot(gap*np.asarray(checkpoints), avg_reward_means[0,b,:,1], 'r') # DRS+PPO
    plt.fill_between(gap*np.asarray(checkpoints), avg_reward_means[0,b,:,1]-1.96*avg_reward_stderrs[0,b,:,1], avg_reward_means[0,b,:,1]+1.96*avg_reward_stderrs[0,b,:,1], facecolor='r', alpha=0.25)
    plt.plot(gap*np.asarray(checkpoints), avg_reward_means[1,b,:,1], 'b') # ProMP
    plt.fill_between(gap*np.asarray(checkpoints), avg_reward_means[1,b,:,1]-1.96*avg_reward_stderrs[1,b,:,1], avg_reward_means[1,b,:,1]+1.96*avg_reward_stderrs[1,b,:,1], facecolor='b', alpha=0.25)
    plt.xlabel('Training Timesteps')
    plt.ylabel('Test Reward')
    plt.savefig(os.path.join(OUTPUT, 'ppo_promp_{}_{}_reward_update_1.png'.format(environment, mode)))
    plt.close(fig)

    fig = plt.figure()
    plt.plot(gap*np.asarray(checkpoints), avg_reward_means[2,b,:,1], 'r') # DRS+TRPO
    plt.fill_between(gap*np.asarray(checkpoints), avg_reward_means[2,b,:,1]-1.96*avg_reward_stderrs[2,b,:,1], avg_reward_means[2,b,:,1]+1.96*avg_reward_stderrs[2,b,:,1], facecolor='r', alpha=0.25)
    plt.plot(gap*np.asarray(checkpoints), avg_reward_means[3,b,:,1], 'b') # ProMP
    plt.fill_between(gap*np.asarray(checkpoints), avg_reward_means[3,b,:,1]-1.96*avg_reward_stderrs[3,b,:,1], avg_reward_means[3,b,:,1]+1.96*avg_reward_stderrs[3,b,:,1], facecolor='b', alpha=0.25)
    plt.xlabel('Training Timesteps')
    plt.ylabel('Test Reward')
    plt.savefig(os.path.join(OUTPUT, 'trpo_trpomaml_{}_{}_reward_update_1.png'.format(environment, mode)))
    plt.close(fig)

