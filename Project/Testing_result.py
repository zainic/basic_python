from matplotlib import image as img
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import pack as pk

# declare resolution
height = 24
width = 24
# Import data and target
Raw_Data = pd.read_csv("test.csv")
print(Raw_Data)
test = Raw_Data['image'].to_numpy()
target = Raw_Data['t'].to_numpy()
print("Data :\n", test)
print("Target :\n", target)
bkelas = 10
bfeat = width * height
# Import image for testing
mtest = []
for i in test:
    str = 'Test\img%d.png' % i
    data = img.imread(str)
    array = np.asarray(data)
    gray_array = pk.gray(array)
    max_array = pk.maximize(gray_array)
    mtest.append(max_array)
main_test = []
for i in test:
    TwoD = pk.compressed3Dto2D(mtest[i-1])
    main_test.append(pk.TwoD_to_OneD(TwoD))
Raw_Weight = pd.read_csv("weight.csv")
W = Raw_Weight.drop("1", axis=1).to_numpy()
print(W)
Akurasi = 0
for i in range(len(main_test)):
    jarak = np.zeros(bkelas)
    for j in range(bkelas):
        jarak[j] = pk.norm(main_test[i] - W[j])
    Jmin = np.amin(jarak)
    J = np.argmin(jarak)
    if target[i] == J:
        Akurasi = Akurasi + 1
    print("Img", i+1, ".png dipetakan ke ", J, "Seharusnya", target[i])
print("Akurasi = ", Akurasi / len(main_test) * 100)