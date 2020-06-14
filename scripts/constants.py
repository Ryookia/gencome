# FEATURES_FILE = "/train-features.csv"
FEATURES_FILE = "/ress.csv"
CLASS_KEYWORD = "class"
TEST_KEYWORD = "test"
FILE_SEPARATOR = "_"

MAX_TREE_DEPTH = 5
MIN_TREE_DEPTH = 1

POPULATION_SIZE = 150

TOURNAMENT_SIZE = 5

CROSS_PROBABILITY = 0.3
MUTATE_PROBABILITY = 0.1
GENERATION_COUNT = 50

BASE_FEATURE_NAME = "Feature"

JAVA_KEYWORD_LIST = ['abstract', 'continue', 'for', 'new', 'switch',
                     'assert', 'default', 'goto', 'package', 'synchronized',
                     'boolean', 'do', 'if', 'private', 'this',
                     'break', 'double', 'implements', 'protected', 'throw',
                     'byte', 'else', 'import', 'public', 'throws',
                     'case', 'enum', 'instanceof', 'return', 'transient',
                     'catch', 'extends', 'int', 'short', 'try',
                     'char', 'final', 'interface', 'static', 'void',
                     'class', 'finally', 'long', 'strictfp', 'volatile',
                     'const', 'float', 'native', 'super', 'while']
JAVA_KEYWORD_LIST_KEY = 'java keywords'

JAVA_PRIMITIVES_LIST = ['boolean', 'double', 'byte', 'int', 'short', 'char', 'long', ' float']
JAVA_PRIMITIVES_LIST_KEY = 'java primitives'

JAVA_EXCEPTIONS_LIST = ['throw', 'catch', 'try', 'finally']
JAVA_EXCEPTIONS_LIST_KEY = 'exception keyword'

JAVA_ACCESS_MOD_LIST = ['private', 'public', 'protected']
JAVA_ACCESS_MOD_LIST_KEY = 'access modifier'

JAVA_CONDITION_LIST = ['if', 'else', 'switch', 'case', 'break']
JAVA_CONDITION_LIST_KEY = 'condition keywords'

JAVA_LOOP_LIST = ['for', 'while', 'do', 'continue']
JAVA_LOOP_LIST_KEY = 'loop keyword'

JAVA_REDUCED_KEYWORD_LIST = ['abstract', 'new',
                             'assert', 'default', 'goto', 'package', 'synchronized',
                             'this', 'implements', 'import', 'enum', 'instanceof',
                             'return', 'transient', 'extends', 'final', 'interface',
                             'static', 'void', 'class', 'strictfp', 'volatile',
                             'const', 'native', 'super']
JAVA_REDUCED_KEYWORD_LIST_KEY = 'java keywords reduced'

JAVA_COMMENT_LIST = ['header_comment', 'whole_line_comment', 'block_comment']
JAVA_COMMENT_LIST_KEY = 'comment line'

JAVA_SQUARE_BRACKETS = ['[', ']']
JAVA_SQUARE_BRACKETS_KEY = 'square bracket'
JAVA_ROUND_BRACKETS = ['(', ')']
JAVA_ROUND_BRACKETS_KEY = 'round bracket'
JAVA_CURLY_BRACKETS = ['{', '}']
JAVA_CURLY_BRACKETS_KEY = 'curly bracket'
JAVA_ANGLE_BRACKETS = ['<', '>']
JAVA_ANGLE_BRACKETS_KEY = 'angle bracket'

#Keywords that appeared below 20 times in whole database
JAVA_LOW_FREQUENCY_WORD_LIST = ['header_comment', '0A', '0_0', '0a', '0a0',
                                '0aA', 'A0', 'A0_A', 'A_0', 'A_0_0',
                                'Aa0Aa', 'Aa_a', '^', '_', '_A',
                                'a0a', 'aAa0Aa', 'a_A', 'a_a', 'assert',
                                'continue', 'finally', 'native', 'switch', 'synchronized',
                                'transient', 'block_comment']
JAVA_LOW_FREQUENCY_WORD_LIST_KEY = 'low frequency keyword'

JAVA_CONTENTS = ['contents']

JAVA_BRACKETS = JAVA_SQUARE_BRACKETS + JAVA_ROUND_BRACKETS + JAVA_CURLY_BRACKETS + JAVA_ANGLE_BRACKETS
JAVA_BRACKETS_KEY = "bracket"
