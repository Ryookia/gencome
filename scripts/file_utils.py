import csv
import os
from scripts.utils import get_primitive_keyword
from scripts.data_holder import DataHolder
from scripts.constants import FEATURES_FILE, CLASS_KEYWORD, TEST_KEYWORD, FILE_SEPARATOR, NOT_PREFIX


def load_data(data_file_path):

    data = DataHolder()

    with open(data_file_path + FEATURES_FILE) as csvFile:
        reader = csv.reader(csvFile, delimiter='$')
        data.features = next(reader)[1:-1]
        current_file = None
        for line in reader:
            file_name = line[0].split(":")[0]
            file_line = line[0].split(":")[1]
            if current_file != file_name:
                data.features_data[file_name] = {}
                current_file = file_name
            data.features_data[file_name][int(file_line)] = line[1:]
    sorted_data = sorted(data.features_data.items())
    data.features_data = dict()
    for entry in sorted_data:
        data.features_data[entry[0]] = entry[1]
    file_names = os.listdir(data_file_path)

    for file_name in file_names:
        current_file_data = file_name.split(FILE_SEPARATOR)
        if len(current_file_data) == 1:
            continue
        if current_file_data[0] != CLASS_KEYWORD and current_file_data[0] != TEST_KEYWORD:
            continue
        value_type = current_file_data[1].split(".")[0]
        data.value_types.append(value_type)
        if current_file_data[0] == CLASS_KEYWORD:
            data.class_data[value_type] = {}
            with open(data_file_path + "/" + file_name) as csvFile:
                reader = csv.reader(csvFile, delimiter='$')
                next(reader)
                for line in reader:
                    data.class_data[value_type][line[1]] = line[2]
        if current_file_data[0] == TEST_KEYWORD:
            data.test_data[value_type] = {}
            with open(data_file_path + "/" + file_name) as csvFile:
                reader = csv.reader(csvFile, delimiter='$')
                next(reader)
                for line in reader:
                    data.test_data[value_type][line[1]] = line[2]

    for type_key in data.test_data.keys():
        if type_key not in data.test_data_values:
            data.test_data_values[type_key] = []
            for key in data.test_data[type_key]:
                data.test_data_values[type_key].append(int(data.test_data[type_key][key]))

    return data


def save_rules(file, rules, data_holder):
    for rule in rules:
        for keyword in rule[0]:
            line = ""
            if not keyword[0]:
                line += NOT_PREFIX
            line += get_primitive_keyword(keyword[1], data_holder.features)
            file.write(line)
            file.write(', ')
        file.write(rule[1])
        file.write("\n")
    file.write("----\n")
