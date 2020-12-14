import numpy as np
from os import walk, path, replace
from xml.dom.minidom import parse
import cv2, sys, os

f = []
root = './data2'

for (dirpath, dirnames, filenames) in walk(root + '/Annotations/'):
    f.append(filenames)
data = f[0]
np.random.shuffle(data)

lenData = len(data)

totalTrain = round(0.80*lenData)
totalVal = round(0.19*lenData)

train = data[:totalTrain]
val = data[totalTrain:totalTrain + totalVal]
test = data[totalTrain + totalVal:]

train_txt = open('train.txt', "w+")
val_txt = open('val.txt', "w+")
test_txt = open('test.txt', "w+")
trainval_txt = open('trainval.txt', "w+")

def getFileNameWithoutExt(filename):
    return filename.split('.')[0]

def lprint(str):
    print(str.ljust(os.get_terminal_size().columns - 1), end="\r")

def genDataset(dataset, file):
    for i in range(len(dataset)):
        xml_data = parse(root + '/Annotations/' + dataset[i])
        filename = xml_data.getElementsByTagName('filename')[0].firstChild.data
        lprint("Checking {}/JPEGImages/{}".format(root, filename))
        assert path.exists(root + "/JPEGImages/" + filename) == True
        file.write(getFileNameWithoutExt(filename) + "\n")
        trainval_txt.write(getFileNameWithoutExt(filename) + "\n")
        img = cv2.imread((root + "/JPEGImages/" + filename))
        assert img.shape == (240, 320, 3)

print("\nChecking dataset...\n")

genDataset(train, train_txt)
genDataset(val, val_txt)
genDataset(test, test_txt)

lprint("DATA LOOK OK!")


print("\n\nGenerated {} train, {} val and {} test".format(len(train), len(val), len(test)))
train_txt.close()
val_txt.close()
trainval_txt.close()

print("\n")

target = root + '/ImageSets/Main/train.txt'
print("Copy train.txt to {}".format(target))
replace("train.txt", target)
target = root + '/ImageSets/Main/val.txt'
print("Copy val.txt to {}".format(target))
replace("val.txt", target)
target = root + '/ImageSets/Main/trainval.txt'
print("Copy trainval.txt to {}".format(target))
replace("trainval.txt", target)
target = root + '/ImageSets/Main/test.txt'
print("Copy test.txt to {}".format(target))
replace("test.txt", target)
print("\nDone!\n")