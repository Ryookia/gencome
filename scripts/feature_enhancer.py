import numpy as np
import sys
import csv

from scripts.constants import JAVA_PRIMITIVES_LIST, JAVA_EXCEPTIONS_LIST, JAVA_ACCESS_MOD_LIST, JAVA_CONDITION_LIST, \
    JAVA_LOOP_LIST, JAVA_COMMENT_LIST, JAVA_BRACKETS, JAVA_PRIMITIVES_LIST_KEY, JAVA_EXCEPTIONS_LIST_KEY, \
    JAVA_ACCESS_MOD_LIST_KEY, JAVA_CONDITION_LIST_KEY, JAVA_LOOP_LIST_KEY, JAVA_COMMENT_LIST_KEY, JAVA_BRACKETS_KEY, \
    JAVA_LOW_FREQUENCY_WORD_LIST, JAVA_CONTENTS

if len(sys.argv) == 1:
    print("No argument for csv file")
    exit()

FILE_SEPARATOR = "$"
data_file_path = sys.argv[1]

data = []

FEATURES_GROUP = [JAVA_PRIMITIVES_LIST, JAVA_EXCEPTIONS_LIST,
                  JAVA_ACCESS_MOD_LIST, JAVA_CONDITION_LIST,
                  JAVA_LOOP_LIST, JAVA_COMMENT_LIST,
                  JAVA_BRACKETS]
FEATURES_GROUP_KEY = [JAVA_PRIMITIVES_LIST_KEY, JAVA_EXCEPTIONS_LIST_KEY,
                      JAVA_ACCESS_MOD_LIST_KEY, JAVA_CONDITION_LIST_KEY,
                      JAVA_LOOP_LIST_KEY, JAVA_COMMENT_LIST_KEY,
                      JAVA_BRACKETS_KEY]
FEATURES_REMOVE = [JAVA_LOW_FREQUENCY_WORD_LIST, JAVA_CONTENTS]

with open(data_file_path) as csvFile:
    reader = csv.reader(csvFile, delimiter=FILE_SEPARATOR)
    features = next(reader)
    for line in reader:
        data.append(line)

    columns_to_group = []
    columns_to_remove = []
    for group in FEATURES_GROUP:
        tmp_group = []
        for feature in group:
            try:
                tmp_group.append(features.index(feature))
            except ValueError as exception:
                continue
        columns_to_remove += tmp_group
        columns_to_group.append(tmp_group)

    for group in FEATURES_REMOVE:
        tmp_remove = []
        for feature in group:
            try:
                tmp_remove.append(features.index(feature))
            except ValueError as exception:
                continue
        columns_to_remove += tmp_remove

    columns_to_remove = list(dict.fromkeys(columns_to_remove))
    columns_to_remove.sort()

    for line in data:
        for group in columns_to_group:
            amount = 0
            for feature_col in group:
                amount += int(line[feature_col])
            line.append(str(amount))

        offset = 0
        for column in columns_to_remove:
            line.pop(column - offset)
            offset += 1

    offset = 0
    for column in columns_to_remove:
        features.pop(column - offset)
        offset += 1
    for key in FEATURES_GROUP_KEY:
        features.append(key)

    # for line in data:
    #     for index, cell in enumerate(line):
    #         line[index] = '"' + str(cell) + '"'

    # for index, feature in enumerate(features):
    #     features[index] = '"' + feature

    with open('/home/Shodan/ress.csv', 'w+') as csvResult:
        writer = csv.writer(csvResult, delimiter=FILE_SEPARATOR, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(features)
        for line in data:
            writer.writerow(line)

