import os
import pickle
import numpy as np
import json
np.seterr(all='raise') 

batch_size = 100
input_length = 30
time_interval = 15    # 20min 一个interval
pred_length = 30
root_dir = "/data/wuning/traffic-data/raw"
adj_dir = "/data/wuning/traffic-data/beijing/map/2ring.json"   #RTICLinksTriple.json
road_adj = json.load(open(adj_dir, "r"))
files = os.listdir(root_dir)
data = {}

#def convert2file(id):
#  return id[:6] + "_" + id[6] + id[7:].zfill(4)
road2road = {}

def load_two_ring_data():   #加载二环数据  2ring.json
  for item in road_adj:
    print(road_adj[item])
    if "o" in road_adj[item]: 
      for ori in road_adj[item]["o"]:
        if item in road2road:  
          road2road[item].append(ori)
        else:
          road2road[item] = [ori]
    if "d" in road_adj[item]:      
      for des in road_adj[item]["d"]:
        if item in road2road:  
          road2road[item].append(des)
        else:
          road2road[item] = [des]
      
def load_all_ring_data():    #加载全部数据   RTICLinksTriple.json
  for item in road_adj:
    if convert2file(item[0]) in road2road:  
      road2road[convert2file(item[0])].append(convert2file(item[1]))
    else:
      road2road[convert2file(item[0])] = [convert2file(item[1])]

load_two_ring_data()

waiting_set = []
queue = ["565972_20181"]   #595662_10083 is start link of all ring data.
while len(waiting_set) < 2500:
  waiting_set.append(queue[0])    
  current = queue[0]
  print(current)  
  queue.pop(0)
  if current in road2road:
    for item in road2road[current]:
      queue.append(item)
  else:
    continue         
for i in range(len(waiting_set)): 
  try:
    f = open(root_dir + "/" + waiting_set[i], "r")  
    
  except Exception as err:
    continue 
#  print(root_dir + "/" + files[i])
  lines = f.readlines()
  line = lines[0]
  temp = [float(item) for item in line.split()]
  temp = np.array(temp)
  try:
#    pass
    temp = temp / 100.0
#    temp = (temp - temp.min()) / (temp.max() - temp.min())
  except Exception as err:
    print(err)
    print("----")
    continue
  data[waiting_set[i]] = temp

slices = []
slices_time = []
slices_pred = []

# slices  n * input_length   slices_time   n * pred_length

train_seq2seq = [] #[roads, batch_num, batch_size, 2, length]

#train_direct = []  #

def get_batch_data(da, input_lengh, pred_length): 
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
  return road
print(len(waiting_set))
for road_id in waiting_set[:1000]:
  if road_id in data and road_id in road2road: 
    road = get_batch_data(data[road_id], input_length, pred_length)
  else:
    continue       
  neighs = []
#  print(road_id, road2road)
  for next_hop in road2road[road_id]:
      if next_hop in data:
        temp = get_batch_data(data[next_hop], input_length, pred_length)
        if np.array(temp).shape == np.array(road).shape:
          neighs.append(temp)
  if len(neighs) == 0:
    continue
  train_seq2seq.append([road, neighs])    
pickle.dump(train_seq2seq, open("/data/wuning/traffic-data/beijing/gru_gcn/train_v3_no_scale", "wb"))

