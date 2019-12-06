import os
import pickle
import numpy as np
np.seterr(all='raise') 

batch_size = 100
input_length = 30
time_interval = 15    # 20min 一个interval
pred_length = 30
root_dir = "/data/wuning/traffic-data/raw"
files = os.listdir(root_dir)
data = []
for i in range(500): #选择五百条路
  f = open(root_dir + "/" + files[i], "r")    
#  print(root_dir + "/" + files[i])
  lines = f.readlines()
  line = lines[0]
  temp = [float(item) for item in line.split()]
  temp = np.array(temp)
  try:
    temp = (temp - temp.min()) / (temp.max() - temp.min())
  except Exception as err:
    print(err)
    print("----")
    continue
  data.append(temp)

slices = []
slices_time = []
slices_pred = []

# slices  n * input_length   slices_time   n * pred_length

train_seq2seq = [] #[roads, batch_num, batch_size, 2, length]

#train_direct = []  #

for da in data:
  road = []  
  slices = []
  slices_time = []
  for i in range(0, len(da), input_length + pred_length):
    if i + input_length + pred_length > len(da):
      break          
    slices.append(da[i : i + input_length + pred_length])
#    slices_pred.append(da[i + input_length : i + input_length + pred_length])
    slices_time.append([((j % 1440) / 15) for j in range(i, i + input_length + pred_length)])
  #  print(len(slices_pred))
  for i in range(0, len(slices), batch_size):    
#    batch = np.array(slices[i : i + batch_size])[:, np.newaxis, :]
    if(i + batch_size > len(slices)):
      break 
    num = 0
    batch = np.concatenate((np.array(slices[i : i + batch_size])[:, np.newaxis, :], np.array(slices_time[i : i + batch_size])[:, np.newaxis, :]), axis = 1)
    road.append(batch)
  train_seq2seq.append(road)    
pickle.dump(train_seq2seq, open("/data/wuning/traffic-data/beijing/gru/train_v1", "wb"))

