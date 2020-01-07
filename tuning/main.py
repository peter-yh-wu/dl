'''Simple Hyperparameter optimization

Peter Wu
peterw1@andrew.cmu.edu
'''

import os
import json
import numpy as np

from itertools import chain, combinations


def powerset(xs):
    '''powerset([1,2,3]) --> [] [1,] [2,] [3,] [1,2] [1,3] [2,3] [1,2,3]

    Args:
        xs is a list
    
    Return:
        list of lists, each sublist is a list(non-empty subset of set(xs))
    '''
    chn = chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))
    chn_list = list(chn)
    list_list = [list(x) for x in chn_list]
    return list_list[1:]

def grid_search_recursion(exp_path, command_path, goal,
                            log_path, tag, params, verbose=True):
    exp_log_path = os.path.join(exp_path, 'exp.log')

    if len(params) == 0:
        trial_id = log_path
        log_path = log_path+'.log'
        log_path = os.path.join(exp_path, log_path)
        cmd = 'python3 %s %s -log_path %s' % (command_path, tag, log_path)
        result = os.system(cmd)
        if result > 0:
            print('error running %s' % cmd)
            return
        with open(log_path, 'r') as inf:
            lines = inf.readlines()
            last_line = '0.0 0.0'
            if len(lines) != 0:
                last_line = lines[-1].strip()
        with open(exp_log_path, 'a+') as ouf:
            ouf.write("%s %s\n" % (trial_id, last_line))
    else:
        curr_param = params[0]
        new_params = params[1:]
        param_type = curr_param['type']
        if param_type == "DISCRETE" or param_type == "CATEGORICAL":
            values = curr_param['values']
            for v in values:
                new_tag = tag+" -%s %s" % (curr_param['name'], v)
                new_log_path = log_path+"_%s_%s" % (curr_param['name'], v)
                grid_search_recursion(exp_path, command_path, goal, 
                    new_log_path, new_tag, new_params)
        elif param_type == "COMBO":
            values = curr_param['values']
            combos = powerset(values)
            for c in combos:
                c_str = " ".join(c)
                new_tag = tag+" -%s %s" % (curr_param['name'], c_str)
                c_log_str = "_".join(c)
                new_log_path = log_path+"_%s_%s" % (curr_param['name'], c_log_str)
                grid_search_recursion(exp_path, command_path, goal, 
                    new_log_path, new_tag, new_params)
        elif param_type == "INTEGER":
            min_value = curr_param['min']
            max_value = curr_param['max']
            step_size = 1
            if "step" in curr_param:
                step_size = curr_param["step"]
            for v in range(min_value, max_value+1, step_size):
                new_tag = tag+" -%s %s" % (curr_param['name'], v)
                new_log_path = log_path+"_%s_%s" % (curr_param['name'], v)
                grid_search_recursion(exp_path, command_path, goal,
                    new_log_path, new_tag, new_params)
        elif param_type == "DOUBLE":
            min_value = curr_param['min']
            max_value = curr_param['max']
            step_size = curr_param["step"]
            num_values = int((max_value-min_value)/step_size)+1
            for i in range(num_values):
                v = min_value+step_size*i
                new_tag = tag+" -%s %s" % (curr_param['name'], v)
                new_log_path = log_path+"-%s_%s" % (curr_param['name'], v)
                grid_search_recursion(exp_path, command_path, goal,
                    new_log_path, new_tag, new_params)

def main(verbose=True):
    config_files = os.listdir('configs')
    config_files = [f for f in config_files if f.endswith('.json')]
    config_file_paths = ['configs/'+f for f in config_files]

    for i, f in enumerate(config_file_paths):
        with open(f, 'r') as inf:
            config_json = json.load(inf)
        
        exp_name = config_files[i][:-5]
        if 'name' in config_json:
            exp_name = config_json['name']
        search_algo = config_json['search']
        task = config_json['task']
        exp = config_json['exp']
        shared_append_str = config_json['append']
        goal = config_json['goal']
        models = config_json['models']
        if verbose:
            print(exp_name)
            print(search_algo)
            print(task)
            print(shared_append_str)
            print(goal)
            print(models)

        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        command_path = os.path.join(parent_dir, 'tasks', task, exp, 'main.py')

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        exp_path = os.path.join(curr_dir, 'persistent', exp_name)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        exp_log_path = os.path.join(exp_path, 'exp.log')
        with open(exp_log_path, 'w+') as ouf:
            pass

        if search_algo == 'GRID':
            for model in models:
                model_name = model['name']
                model_append_str = shared_append_str+" "+model['append']+ \
                                    ' -model %s' % model_name
                params = model['params']
                init_log_path_list = model_append_str.split()
                init_log_path = goal
                for elem in init_log_path_list:
                    if elem.startswith('-'):
                        init_log_path = init_log_path+elem
                    else:
                        init_log_path = init_log_path+'_'+elem
                grid_search_recursion(exp_path, command_path, 
                    exp_path+'/', init_log_path, model_append_str, params)

            # record results
            with open(exp_log_path, 'r') as inf:
                lines = inf.readlines()
            lines = [l for l in lines if not l.startswith("#")]
            triples = [l.strip().split() for l in lines]
            expt_ids = [l[0] for l in triples]
            accs = [float(l[1]) for l in triples]
            mets = [float(l[2]) for l in triples]
            best_acc_index = np.argmax(np.array(accs))
            if goal == "MAX":
                best_met_index = np.argmax(np.array(mets))
            else: # goal = "MIN"
                best_met_index = np.argmin(np.array(mets))
            best_acc_expt = triples[best_acc_index][0]
            best_met_expt = triples[best_met_index][0]
            with open(exp_log_path, 'a+') as ouf:
                ouf.write("# best acc trial: %s\n" % best_acc_expt)
                ouf.write("# best met trial: %s\n" % best_met_expt)

            # remove results of non-optimal experiments
            all_files = os.listdir(exp_path)
            npy_files = [f for f in all_files if f.endswith('.npy')]
            ckpt_files = [f for f in all_files if f.endswith('.ckpt')]
            bad_npy_files = [f for f in npy_files if \
                not (f.startswith(best_acc_expt) or f.startswith(best_met_expt))]
            bad_ckpt_files = [f for f in ckpt_files if \
                not (f.startswith(best_acc_expt) or f.startswith(best_met_expt))]
            bad_npy_paths = [os.path.join(exp_path, f) for f in bad_npy_files]
            bad_ckpt_paths = [os.path.join(exp_path, f) for f in bad_ckpt_files]
            for p in bad_npy_paths:
                os.remove(p)
            for p in bad_ckpt_paths:
                os.remove(p)

        elif search_algo == 'RANDOM':
            pass

if __name__ == "__main__":
    main()