from matplotlib import image as img
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import pack as pk

# Declare dimension of image for training
np.random.seed(42)
height = 24
width = 24
# Import data and target
Raw_Data = pd.read_csv("train.csv")
print(Raw_Data)
train = Raw_Data['image'].to_numpy()
target = Raw_Data['t'].to_numpy()
print("Data :\n", train)
print("Target :\n", target)
bkelas = 10
bfeat = width * height
# Import image based on data
mdata = []
for i in train:
    str = 'Training\img%d.png' % i
    data = img.imread(str)
    array = np.asarray(data)
    gray_array = pk.gray(array)
    max_array = pk.maximize(gray_array)
    mdata.append(max_array)
# Create Weight
W = []
for i in train:
    X = []
    for j in range(height):
        for k in range(width):
            Temp = np.random.rand()
            X.append(Temp)
    W.append(X)
#ilustration of first 25 image after getting imported
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(mdata[i])
plt.show()







# Compressed every image into 1D array
main_data = []
for i in train:
    TwoD = pk.compressed3Dto2D(mdata[i-1])
    main_data.append(pk.TwoD_to_OneD(TwoD))
M = pd.DataFrame(main_data)
M.to_csv('main.csv')

# Training Process
Alfa = 0.02
alfa = Alfa
Decalfa = 0.99
MaxIt = 100
Num = 0
while Num < MaxIt:
    for i in range(len(train)):
        jarak = np.zeros(bkelas)
        for j in range(bkelas):
            jarak[j] = pk.norm(main_data[i]-W[j])
        Jmin = np.amin(jarak)
        J = np.argmin(jarak)
        if J == target[i]:
            for k in range(bfeat):
                W[J][k] = W[J][k] + alfa*(main_data[i][k]-W[J][k])
        else:
            for k in range(bfeat):
                W[J][k] = W[J][k] - alfa*(main_data[i][k]-W[J][k])
    alfa = alfa*Decalfa
    Num = Num + 1
DataT = main_data
bdataT = len(train)
Akurasi = 0
for i in range(bdataT):
    jarak = np.zeros(bkelas)
    for j in range(bkelas):
        jarak[j] = pk.norm(DataT[i] - W[j])
    Jmin = np.amin(jarak)
    J = np.argmin(jarak)
    if target[i] == J:
        Akurasi = Akurasi + 1
    print("Img", i+1, ".png dipetakan ke ", J, "Seharusnya", target[i])
print("Akurasi untuk alfa (", Alfa, ") dan Decalfa (", Decalfa, ") = ", Akurasi / bdataT * 100)
# Exporting current weight
weight = pd.DataFrame(W)
weight.to_csv('weight.csv')
# What program see into our scratch
ImgW = []
for k in range(len(W)):
    n = 0
    P = []
    for i in range(height):
        H = []
        for j in range(width):
            if W[k][n] >= 1:
                S = 0.9999
            else:
                S = W[k][n]
            H.append([S, S, S])
            n += 1
        P.append(H)
    ImgW.append(P)
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(ImgW[i])
plt.show()
