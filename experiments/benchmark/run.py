import os

HEADLESS = 'xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" --'
python_cmd = 'python3'
LOG_DIR = os.getcwd()+'/logs/'
script_dir = os.getcwd()+'/scripts/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(script_dir):
    os.makedirs(script_dir)

## Hyperparameters used for the experiments ##

environment_mode_pairs = [['walker', 'params-interpolate'], ['walker', 'goal-interpolate'],
        ['cheetah', 'goal-interpolate'], ['hopper', 'params-interpolate'], ['metaworld', 'ml1-push'], 
        ['metaworld', 'ml1-reach'], ['metaworld', 'ml10'], ['metaworld', 'ml45']]

seeds = [1, 2, 3, 4, 5]

# Learning rates for DRS+PPO
ppo_lrs = dict()
ppo_lrs['walker-params-interpolate'] = 1e-3
ppo_lrs['walker-goal-interpolate'] = 1e-3
ppo_lrs['cheetah-goal-interpolate'] = 1e-2
ppo_lrs['hopper-params-interpolate'] = 1e-3
ppo_lrs['metaworld-ml1-push'] = 1e-4
ppo_lrs['metaworld-ml1-reach'] = 1e-4
ppo_lrs['metaworld-ml10'] = 1e-4
ppo_lrs['metaworld-ml45'] = 1e-4

# Learning rates for ProMP
promp_lrs = dict()
promp_lrs['walker-params-interpolate'] = 1e-3
promp_lrs['walker-goal-interpolate'] = 1e-3
promp_lrs['cheetah-goal-interpolate'] = 1e-4
promp_lrs['hopper-params-interpolate'] = 1e-3
promp_lrs['metaworld-ml1-push'] = 1e-4
promp_lrs['metaworld-ml1-reach'] = 1e-4
promp_lrs['metaworld-ml10'] = 1e-4
promp_lrs['metaworld-ml45'] = 1e-4

# Inner learning rates for ProMP
promp_inner_lrs = dict()
promp_inner_lrs['walker-params-interpolate'] = 1e-2
promp_inner_lrs['walker-goal-interpolate'] = 1e-2
promp_inner_lrs['cheetah-goal-interpolate'] = 1e-2
promp_inner_lrs['hopper-params-interpolate'] = 1e-2
promp_inner_lrs['metaworld-ml1-push'] = 1e-4
promp_inner_lrs['metaworld-ml1-reach'] = 1e-4
promp_inner_lrs['metaworld-ml10'] = 1e-3
promp_inner_lrs['metaworld-ml45'] = 1e-5

# Step sizes for DRS+TRPO
trpo_step_sizes['walker-params-interpolate'] = 0.1
trpo_step_sizes['walker-goal-interpolate'] = 0.01
trpo_step_sizes['cheetah-goal-interpolate'] = 0.01
trpo_step_sizes['hopper-params-interpolate'] = 0.1
trpo_step_sizes['metaworld-ml1-push'] = 0.1
trpo_step_sizes['metaworld-ml1-reach'] = 0.001
trpo_step_sizes['metaworld-ml10'] = 0.001
trpo_step_sizes['metaworld-ml45'] = 0.01

# Step sizes for TRPO-MAML
trpomaml_step_sizes = dict()
trpomaml_step_sizes['walker-params-interpolate'] = 0.1
trpomaml_step_sizes['walker-goal-interpolate'] = 0.1
trpomaml_step_sizes['cheetah-goal-interpolate'] = 0.001
trpomaml_step_sizes['hopper-params-interpolate'] = 0.1
trpomaml_step_sizes['metaworld-ml1-push'] = 0.1
trpomaml_step_sizes['metaworld-ml1-reach'] = 0.001
trpomaml_step_sizes['metaworld-ml10'] = 0.01
trpomaml_step_sizes['metaworld-ml45'] = 0.1

# Inner learning rates for TRPO-MAML
trpomaml_inner_lrs = dict()
trpomaml_inner_lrs['walker-params-interpolate'] = 0.01
trpomaml_inner_lrs['walker-goal-interpolate'] = 0.01
trpomaml_inner_lrs['cheetah-goal-interpolate'] = 0.01
trpomaml_inner_lrs['hopper-params-interpolate'] = 0.01
trpomaml_inner_lrs['metaworld-ml1-push'] = 0.0001
trpomaml_inner_lrs['metaworld-ml1-reach'] = 0.00001
trpomaml_inner_lrs['metaworld-ml10'] = 0.001
trpomaml_inner_lrs['metaworld-ml45'] = 0.001


## Example code for creating scripts to run the experiments, starting from repository base directory ##

# DRS+PPO
for pair in environment_mode_pairs:
    run = 0
    for seed in seeds:
        run += 1
        environment = pair[0]
        mode = pair[1]
        lr = ppo_lrs[environment+'-'+mode]
        print('Creating PPO script for environment {} mode {} learning rate {} seed {}'.format(environment, mode, lr, seed))

        # Create command
        target = 'experiments/benchmark/drs_ppo.py --env ' + environment + ' --mode ' + mode + ' --run ' + str(run) + ' --learning-rate ' + str(lr) + ' --seed ' + str(seed)
        output = LOG_DIR + 'ppo_' + environment + '_' + mode + '_' + str(run) + '.log'
        cmd = '{headless} {python_cmd} {target} &> {output}'.format(headless=HEADLESS, python_cmd=python_cmd, target=target, output=output)

        # Write to executable scripts
        name = 'scripts/ppo_'+environment+'_'+mode+'_'+str(run)+'.sh'
        with open(name, 'w') as sh:
            sh.write('#!/bin/sh')
            sh.write('\n')
            sh.write(cmd)

# ProMP
for pair in environment_mode_pairs:
    run = 0
    for seed in seeds:
        run += 1
        environment = pair[0]
        mode = pair[1]
        lr = promp_lrs[environment+'-'+mode]
        inner_lr = promp_inner_lrs[environment+'-'+mode]
        print('Creating ProMP script for environment {} mode {} learning rate {} inner lr {} seed {}'.format(environment, mode, lr, inner_lr, seed))

        # Create command
        target = 'experiments/benchmark/promp.py --env ' + environment + ' --mode ' + mode + ' --run ' + str(run) + ' --learning-rate ' + str(lr) + ' --inner-lr ' + str(inner_lr) + ' --seed ' + str(seed)
        output = LOG_DIR + 'promp_' + environment + '_' + mode + '_' + str(run) + '.log'
        cmd = '{headless} {python_cmd} {target} &> {output}'.format(headless=HEADLESS, python_cmd=python_cmd, target=target, output=output)

        # Write to executable scripts
        name = 'scripts/promp_'+environment+'_'+mode+'_'+str(run)+'.sh'
        with open(name, 'w') as sh:
            sh.write('#!/bin/sh')
            sh.write('\n')
            sh.write(cmd)

# DRS+TRPO
for pair in environment_mode_pairs:
    run = 0
    for seed in seeds:
        run += 1
        environment = pair[0]
        mode = pair[1]
        step_size = trpo_step_sizes[environment+'-'+mode]
        print('Creating TRPO script for environment {} mode {} step size {} seed {}'.format(environment, mode, step_size, seed))

        # Create command
        target = 'experiments/benchmark/drs_trpo.py --env ' + environment + ' --mode ' + mode + ' --run ' + str(run) + ' --step-size ' + str(step_size) + ' --seed ' + str(seed)
        output = LOG_DIR + 'trpo_' + environment + '_' + mode + '_' + str(run) + '.log'
        cmd = '{headless} {python_cmd} {target} &> {output}'.format(headless=HEADLESS, python_cmd=python_cmd, target=target, output=output)

        # Write to executable scripts
        name = 'scripts/trpo_'+environment+'_'+mode+'_'+str(run)+'.sh'
        with open(name, 'w') as sh:
            sh.write('#!/bin/sh')
            sh.write('\n')
            sh.write(cmd)

# TRPO-MAML
for pair in environment_mode_pairs:
    run = 0
    for seed in seeds:
        run += 1
        environment = pair[0]
        mode = pair[1]
        step_size = trpomaml_step_sizes[environment+'-'+mode]
        inner_lr = trpomaml_inner_lrs[environment+'-'+mode]
        print('Creating TRPO MAML script for environment {} mode {} step size {} inner lr {} seed {}'.format(environment, mode, step_size, inner_lr, seed))

        # Create command
        target = 'experiments/benchmark/trpo_maml.py --env ' + environment + ' --mode ' + mode + ' --run ' + str(run) + ' --step-size ' + str(step_size) + ' --inner-lr ' + str(inner_lr) + ' --seed ' + str(seed)
        output = LOG_DIR + 'trpomaml_' + environment + '_' + mode + '_' + str(run) + '.log'
        cmd = '{headless} {python_cmd} {target} &> {output}'.format(headless=HEADLESS, python_cmd=python_cmd, target=target, output=output)

        # Write to executable scripts
        name = 'scripts/trpomaml_'+environment+'_'+mode+'_'+str(run)+'.sh'
        with open(name, 'w') as sh:
            sh.write('#!/bin/sh')
            sh.write('\n')
            sh.write(cmd)

# Testing

algos = ['ppo', 'promp', 'trpo', 'trpomaml']
for alg in algos:
    for pair in environment_mode_pairs:
        for seed in seeds:
            for c in range(21):
                environment = pair[0]
                mode = pair[1]
                if alg == 'ppo':
                    lr = ppo_lrs[environment+'-'+mode]
                elif alg == 'promp':
                    lr = promp_inner_lrs[environment+'-'+mode]
                elif alg == 'trpo':
                    lr = 0.0
                elif alg == 'trpomaml':
                    lr = trpomaml_inner_lrs[environment+'-'+mode]
                print('Creating test script for algorithm {} environment {} mode {} learning rate {} seed {} checkpoint {}'.format(alg, environment, mode, lr, seed, c))

                # Create command
                target = 'experiments/benchmark/test_finetune.py --algorithm ' + alg + ' --env ' + environment + ' --mode ' + mode + ' --run ' + str(seed) + ' --learning-rate ' + str(lr) + ' --seed ' + str(seed) + ' --checkpoint ' + str(c)
                output = LOG_DIR + 'test_' + alg + '_' + environment + '_' + mode + '_' + str(seed) + '_' + str(c) + '.log'
                cmd = '{headless} {python_cmd} {target} &> {output}'.format(headless=HEADLESS, python_cmd=python_cmd, target=target, output=output)

                # Write to executable scripts
                name = 'scripts/test_'+alg+'_'+environment+'_'+mode+'_'+str(seed)+'_'+str(c)+'.sh'
                with open(name, 'w') as sh:
                    sh.write('#!/bin/sh')
                    sh.write('\n')
                    sh.write(cmd)
