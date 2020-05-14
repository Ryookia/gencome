import sys
import csv
import os

if len(sys.argv) == 1:
    print("No argument for csv file")
    exit()

FILE_SEPARATOR = "$"

csvFilePath = sys.argv[1]
resultClassFiles = [["/class_blank.csv", 1], ["/class_comment.csv", 2], ["/class_code.csv", 3]]
resultTestFiles = [["/test_blank.csv", 1], ["/test_comment.csv", 2], ["/test_code.csv", 3]]

with open(csvFilePath) as csvFile:
    data = csv.reader(csvFile, delimiter=',')
    dataSorted = sorted(data, key=lambda x: x[0], reverse=False)
    id = 0
    idSet = {}
    for resultClassFileData in resultClassFiles:
        currentFile = open(os.getcwd() + resultClassFileData[0], "w+")
        currentFile.write("id$name$value\n")
        for row in dataSorted:
            if row[0][:6] == "./main" and row[0][-4:] == "java":
                fileName = row[0][7:]
                if fileName in idSet.keys():
                    id = idSet[fileName]
                else:
                    idSet[fileName] = id
                    id += 1
                currentFile.write(str(idSet[fileName]) + FILE_SEPARATOR +
                                  fileName + FILE_SEPARATOR + row[resultClassFileData[1]] + "\n")

    for resultTestFileData in resultTestFiles:
        currentFile = open(os.getcwd() + resultTestFileData[0], "w+")
        currentFile.write("id$name$value\n")
        for row in dataSorted:
            if row[0][:6] == "./test" and row[0][-4:] == "java":
                fileName = row[0][7:]
                if not (fileName.split(".")[0][:-5] + ".java" in idSet.keys()):
                    exit("Test file with no class file")
                else:
                    id = idSet[fileName.split(".")[0][:-5] + ".java"]
                currentFile.write(str(id) + FILE_SEPARATOR
                                  + fileName + FILE_SEPARATOR + row[resultTestFileData[1]] + "\n")






