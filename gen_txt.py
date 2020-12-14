import numpy as np
from os import walk, path, replace
from xml.dom.minidom import parse
import cv2

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


def genDataset(dataset, file):
    for i in range(len(dataset)):
        xml_data = parse(root + '/Annotations/' + dataset[i])
        filename = xml_data.getElementsByTagName('filename')[0].firstChild.data    
        print("Checking " + root + "/JPEGImages/" + filename)    
        assert path.exists(root + "/JPEGImages/" + filename) == True
        file.write(getFileNameWithoutExt(filename) + "\n")
        trainval_txt.write(getFileNameWithoutExt(filename) + "\n")
        img = cv2.imread((root + "/JPEGImages/" + filename))
        assert img.shape == (240, 320, 3)

print("Generating dataset...")

genDataset(train, train_txt)
genDataset(val, val_txt)
genDataset(test, test_txt)

print("Generated {} train, {} val and {} test".format(len(train), len(val), len(test)))
print("Copying file...")
train_txt.close()
val_txt.close()
trainval_txt.close()
replace("train.txt", root + '/ImageSets/Main/train.txt')
replace("val.txt", root + '/ImageSets/Main/val.txt')
replace("trainval.txt", root + '/ImageSets/Main/trainval.txt')
replace("test.txt", root + '/ImageSets/Main/test.txt')
print("Done")