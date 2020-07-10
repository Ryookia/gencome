import csv
import os

import gencome.config
from gencome.utils import get_primitive_keyword, str_individual_with_real_feature_names

def save_rules(file, rules, number, individual):
    file.write(f"Definition #{number} (score: {', '.join([str(x) for x in individual.fitness.values])})\n")
    file.write(f"Tree: {str_individual_with_real_feature_names(individual)}\n")
    file.write(f"Rules:\n")
    for rule in rules:
        if rule[1] == gencome.config.COUNT_LABEL:
            # skup rules having the same keyword for true and false
            keywords_true = set()
            keywords_false = set()
            for i, keyword in enumerate(rule[0]):
                if keyword[0]:
                   keywords_true.add(keyword[1]) 
                else:
                    keywords_false.add(keyword[1]) 
            if len(keywords_true.intersection(keywords_false)) > 0:
                continue
            file.write("if ")
            for i, keyword in enumerate(rule[0]):
                line = ""
                if not keyword[0]:
                    line += gencome.config.NOT_PREFIX
                line += "'"+get_primitive_keyword(keyword[1], gencome.config.features)+"'"
                file.write(line)
                if i+1 < len(rule[0]):
                    file.write(" and ")
            file.write(" : ")
            file.write(rule[1])
            file.write("\n")
    file.write(f"otherwise : {gencome.config.NOT_COUNT_LABEL}\n")
    file.write("----\n")
