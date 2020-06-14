import numpy as np
import sys
import csv

if len(sys.argv) == 1:
    print("No argument for csv file")
    exit()

FILE_SEPARATOR = ","
data_file_path = sys.argv[1]

class_data = []
test_data = []

with open(data_file_path) as csvFile:
    reader = csv.reader(csvFile, delimiter=FILE_SEPARATOR)
    features = next(reader)
    for line in reader:
        class_data.append(int(line[0]))
        test_data.append(int(line[1]))


print(np.corrcoef(class_data, test_data).min())
