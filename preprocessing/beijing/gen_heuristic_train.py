import json
import random
import numpy as np
import pickle
import networkx as nx
import copy
import os
# 生成query以及query对应的Optimal path和optimal time
id2road = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2road", "rb"))
node2id = pickle.load(open("/data/wuning/traffic-data/beijing/map/node2id", "rb"))
id2old_road = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2old_road", "rb"))
id2length = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2length", "rb"))

id2node = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2node", "rb"))
road2id = pickle.load(open("/data/wuning/traffic-data/beijing/map/road2id", "rb"))

def probe(a):
 while True:
  print(a)   

def euclid_distance(a, b):
 return np.sqrt(np.sum(((np.array(a) - np.array(b)) * np.array([0.667 * 111000, 111000]))**2))

def load_future_data(start_time, end_time):
  transfer_time_graph = []
  c_data = {}
 
  for t in range(end_time - start_time):
    transfer_time_graph.append(nx.DiGraph())  
   
  for key in range(len(id2road)):
    ID = id2road[key] #snode_enode
    Flag = False
    try:
      sf = open("/data/wuning/traffic-data/raw/" + id2old_road[key][:6] + "_" + id2old_road[key][6] + id2old_road[key][7:].zfill(4), "rb")
      speeds = sf.readlines()
    except Exception as err:
      continue
      Flag = True
    speeds = speeds[0].split()
    for t, g in zip(range(start_time, end_time), transfer_time_graph):
      lineCount = t        
      if Flag:
        travel_time = 10000   
      else:
        speed = float(speeds[lineCount])
        if speed == 0:
         speed = 1     
        travel_time = float(id2length[key]) / speed * 60
  
      from_node = int(ID.split("_")[0])
      if from_node < 0:
        pass
      else:
        from_node = str(from_node)

      to_node = int(ID.split("_")[1])  
      if to_node < 0:
        pass
      else:
        to_node = str(to_node)
#      print("from_node:", from_node, "to_node:", to_node, ID)
      g.add_edge(node2id[from_node], node2id[to_node], weight = travel_time)
  return transfer_time_graph

def get_time_on_road(G):
 residual = 1
 cost = 0
 while True:
  one_step_part = 5 / ((G[int(cost)]))
  if residual - one_step_part < 0:
   break  
  residual -= one_step_part
  cost += 1
 cost += residual * ((G[int(cost)]))  
 return cost

def insert(item, the_list, f_score):  #插入 保持从小到大的顺序
 if(len(the_list) == 0):
   return [item]
 for i in range(len(the_list)):
  if(f_score[the_list[i]] > f_score[item]):
   the_list.insert(i, item)
   break
  if i == len(the_list) - 1:
   the_list.append(item)
 return the_list

def move(item, the_list, f_score):
 for it in the_list:
  if(f_score[it] == f_score[item]):
   the_list.remove(it)
   break
 return insert(item, the_list, f_score)


def time_dependent_path(startTime, start, end, G):
  openSet = [start]
  closedSet = []
  pathFounded = {start: [start]}
  gScore = {}
  fScore = {}
  gScore[start] = 0
  fScore[start] = 0
  bestScore = 0
  bestTra = 0
  count = 0
  while len(openSet) > 0:
    count += 1
    current = openSet[0]
    openSet.remove(current)
    closedSet.append(current)
#    print(current)
    if current == end:
      bestTra = copy.deepcopy(pathFounded[current]) 
      bestTra.append(end)
      bestScore = gScore[current]
      break
    time = gScore[current]

    if time > 10000:
      continue    
    for neigh in G[int(time)][current]:
      if neigh in closedSet:
        continue
      if G[int(time)][current][neigh]["weight"] == 10000:
        continue
#      print("time:", time)  
      one_step_value = get_time_on_road([G[int(t)][current][neigh]["weight"] for t in range(int(time), int(time) + 1000)])

      g_score = one_step_value + gScore[current]
      temp = copy.deepcopy(pathFounded[current])
      temp.append(neigh)
      import random
      h_score = 0 #random.uniform(0, 20)   #不使用启发值 相当于dijkstra
      if neigh in gScore and g_score > gScore[neigh]:
        continue
      gScore[neigh] = g_score
      fScore[neigh] = gScore[neigh] + h_score
      if neigh not in openSet:
        openSet = insert(neigh, openSet, fScore)
      else:
        openSet = move(neigh, openSet, fScore)
      pathFounded[neigh] = temp  
  return bestTra, bestScore, count  

def node_list_to_link_list(route):
  #require: node的id序列
  #return: node的list以及link的list
  o_route = []
  node_list = []
  for i in range(len(route) - 1):
    node_list.append(str(id2node[route[i]]))
    o_route.append(str(id2node[route[i]]) + "_" + str(id2node[route[i + 1]]))
  f_route = []
  for item in o_route[:-1]: 
    f_route.append(road2id[item])
  return node_list, f_route

files = os.listdir("/data/wuning/traffic-data/raw")
roads_with_data = []
for filename in files:
  roads_with_data.append(filename)
roads = json.load(open("/data/wuning/traffic-data/beijing/map/RTICLinksNew.json","r"))
G = load_future_data(0, 2440)

def gen_batch(msg):
 print("msg:", msg)  
 query_num = 3000
 query = []
 for i in range(query_num):
  if i % 100 == 0:
   print("msg:", msg, "num:", i)  
  ind1 = random.randint(0, len(roads["features"]) - 1)
  start = roads["features"][ind1]["properties"]["SNodeLong"]
  start_cor = roads["features"][ind1]["geometry"]["coordinates"][0]
  ind2 = random.randint(0, len(roads["features"]) - 1)
  end = roads["features"][ind2]["properties"]["SNodeLong"]
  end_cor = roads["features"][ind2]["geometry"]["coordinates"][0]
  if not (roads["features"][ind1]["properties"]["MapID"] + "_" + roads["features"][ind1]["properties"]["Kind"] + roads["features"][ind1]["properties"]["ID"].zfill(4) in roads_with_data and roads["features"][ind2]["properties"]["MapID"] + "_" + roads["features"][ind2]["properties"]["Kind"] + roads["features"][ind2]["properties"]["ID"].zfill(4) in roads_with_data):
   continue  
 
  if euclid_distance(start_cor, end_cor) < 5000: 
   query.append([start, end])
 
  if len(query) > 101:
    break

 batch_size = 100

 
 
 time_list = [6 * 60, 12 * 60, 18 * 60]

 times = []
 paths = []
 labels = []
 count = 0
 for q in query:
  print("msg:", msg, "count:", count)
  count += 1 
  for time in time_list:
   optimal_path, optimal_value, search_count = time_dependent_path(time, node2id[q[0]], node2id[q[1]], G[time:])        
   times.append(time)
  # print(optimal_path, optimal_value)
   if not isinstance(optimal_path, list):
    continue 
   paths.append(node_list_to_link_list(optimal_path))
   labels.append(optimal_value)

 time_batches = []
 path_batches = []
 label_batches = []

 train_batches = []

 for i in range(0, len(times), 100):
  time_batch = times[i: i + 100]
  path_batch = paths[i: i + 100]
  label_batch = labels[i: i + 100]
  train_batches.append([time_batch, path_batch, label_batch])    

# [time_batch, path_batch, label_batch]  

 pickle.dump(train_batches, open("/data/wuning/traffic-data/beijing/heuristic/heuristic_train_" + str(msg), "wb"))

for i in range(1000, 5000):
 gen_batch(i)
#import multiprocessing

#if __name__ == "__main__":
# pool = multiprocessing.Pool(processes=4) 
# for i in range(4):        
#  pool.apply_async(gen_batch, (i, ))
 
# pool.apply_async(func2, (2, ))#1,2,3,4指的是分配的CPU的核的编号,虽然指定了,但还是会根据资源来切换
# pool.apply_async(func3, (3, ))
# pool.apply_async(func4, (4, ))
# pool.close()
# pool.join()
