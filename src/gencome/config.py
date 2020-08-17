
import logging
logger = logging.getLogger(f'gencome')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

features = None
x_features = None
y_dicts = []
y = None
fitness_type = None
toolbox = None
correlation = None
min_tree_depth = None
max_tree_depth = None
min_init_tree_depth = None
max_init_tree_depth = None
tournament_size = None
population_size = None
crossover_prob = None
mutate_prob = None
generations =None
mut_uniform_weight = None
mut_replacement_weight = None
mut_insert_weight = None
mut_shrink_weight = None

BASE_FEATURE_NAME = "Feature"
NOT_PREFIX = "!"
COUNT_LABEL = "true"
NOT_COUNT_LABEL = "false"

MAX_ATTEMPTS_TO_GENERATE_VALID_IND = 1000