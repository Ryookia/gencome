import os
import sys
import shutil


def copy_files(dir_path, file_names):
    for fileName in file_names:
        fileSplitted = fileName.split(".")
        if len(fileSplitted) == 2 and fileSplitted[1] == "java":
            testFilePath = testPath + relativePath + "/" + fileSplitted[0] + "Tests." + fileSplitted[1]
            if os.path.exists(testFilePath):
                print(dir_path + fileName)
                print(testFilePath)
                shutil.copy(dir_path + "/" + fileName, resultPathDir + "main")
                shutil.copy(testFilePath, resultPathDir + "test")


if len(sys.argv) == 1:
    print("No argument for base path")
    exit()
if len(sys.argv) == 2:
    print("No argument for module name")

basePath = sys.argv[1]
moduleName = sys.argv[2]
resultPathDir = os.getcwd()

if len(sys.argv) == 4:
    resultPathDir = sys.argv[3]

classPath = basePath + moduleName + "/src/main"
testPath = basePath + moduleName + "/src/test"

classPathLen = len(classPath)

if not os.path.exists(resultPathDir + "main"):
    os.makedirs(resultPathDir + "main")

if not os.path.exists(resultPathDir + "test"):
    os.makedirs(resultPathDir + "test")

for (dirPath, dirNames, fileNames) in os.walk(classPath):
    relativePath = dirPath[classPathLen:]
    if len(fileNames) > 0:
        copy_files(dirPath, fileNames)
    print(dirPath)



