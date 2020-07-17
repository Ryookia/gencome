from functools import partial, wraps
import math
import random
import operator
import copy
import numpy as np
from scipy.stats import spearmanr
from deap import  gp
from timeit import default_timer as timer
import multiprocessing

import gencome.config

DOUBLE_COUNT = f"({gencome.config.COUNT_LABEL}, {gencome.config.COUNT_LABEL})"
DOUBLE_NOT_COUNT = f"({gencome.config.NOT_COUNT_LABEL}, {gencome.config.NOT_COUNT_LABEL})"

from gencome.utils import get_primitive_keyword, str_individual_with_real_feature_names

logger = gencome.config.logger

def has_invalid_leafs_ind_list(ind):
    for i, x in enumerate(ind):
        if i+1 < len(ind) and ind[i+1] == x:
            return True
    return False

def has_invalid_leafs(ind):
    str_ind = str(ind)
    return DOUBLE_COUNT in str_ind or DOUBLE_NOT_COUNT in str_ind

def is_too_deep_ind(ind, max_value):
    return  ind.height > max_value

def evaluation(data, individual, pset):
    max_tree_depth, x_features, y_count_result = data
    start = timer()
    if is_too_deep_ind(individual, max_tree_depth):
        end = timer()
        logger.debug(f"Evaluating ({multiprocessing.current_process().name}) {end-start:.2f}s, fitness=0.0, {str_individual_with_real_feature_names(individual)}")
        return 0,
    func = gp.compile(expr=individual, pset=pset)
    
    x_count_result = []
    for index in x_features:
        count = 0
        for entry in x_features[index]:
            if func(entry):
                count += 1
        x_count_result.append(count)

    if gencome.config.correlation == "Spearman":
        value = spearmanr(x_count_result, y_count_result)[0]
    elif gencome.config.correlation == "Pearson":
        value = np.corrcoef(x_count_result, y_count_result).min()
    if math.isnan(value):
        end = timer()
        logger.debug(f"Evaluating ({multiprocessing.current_process().name}) {end-start:.2f}s, fitness=0.0, {str_individual_with_real_feature_names(individual)}")
        return 0,
    end = timer()
    logger.debug(f"Evaluating ({multiprocessing.current_process().name}) {end-start:.2f}s, fitness={value:.4f}, {str_individual_with_real_feature_names(individual)}")
    return value,

def multiple_mutator(individual, pset):
    weight_sum = gencome.config.mut_uniform_weight + gencome.config.mut_replacement_weight \
        + gencome.config.mut_insert_weight + gencome.config.mut_shrink_weight
    cur_weight = gencome.config.mut_uniform_weight
    rng = random.random() * weight_sum
    if rng < cur_weight:
        return gp.mutUniform(individual, expr=gencome.config.toolbox.expr_mut, pset=pset)
    cur_weight += gencome.config.mut_replacement_weight
    if rng < cur_weight:
        return gp.mutNodeReplacement(individual, pset=pset)
    cur_weight += gencome.config.mut_insert_weight
    if rng < cur_weight:
        return gp.mutInsert(individual, pset=pset)
    return gp.mutShrink(individual)


def invalid_tree():
 
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_height =  gencome.config.max_tree_depth
            keep_inds = [copy.deepcopy(ind) for ind in args if not is_too_deep_ind(ind, max_height)]
            if len(keep_inds) == 0:
                new_ind = gencome.config.toolbox.individual()
                attempts = 0
                while is_too_deep_ind(new_ind, max_height) and attempts < gencome.config.MAX_ATTEMPTS_TO_GENERATE_VALID_IND:
                    new_ind = gencome.config.toolbox.individual()
                    attempts += 1
                keep_inds = [new_ind,]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if is_too_deep_ind(ind, max_height):
                    new_inds[i] = random.choice(keep_inds)
            return new_inds

        return wrapper

    return decorator

def gen_grow(pset, min_, max_, type_=None):
   
    def condition(height, depth):
        return depth == height or \
               (depth >= min_ and random.random() < pset.terminalRatio)

    ind = gp.generate(pset, min_, max_, condition, type_)
    attempts = 0
    while has_invalid_leafs_ind_list(ind):
        ind = gp.generate(pset, min_, max_, condition, type_)
        attempts += 1
        if attempts > gencome.config.MAX_ATTEMPTS_TO_GENERATE_VALID_IND:
            break
    return ind

def has_keyword(keyword, out1, out2, obj_features):
    if (int(obj_features.tolist()[gencome.config.features.index(keyword)])) > 0:
        return out1(obj_features)
    else:
        return out2(obj_features)

def count(_):
    return True


def no_count(_):
    return False


def primitive_feature(keyword):
    return partial(primitive_keyword, keyword)


def primitive_keyword(keyword, out1, out2):
    return partial(has_keyword, keyword, out1, out2)


