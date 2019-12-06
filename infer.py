import pickle
import sys
import torch
from model.gru_gcn import *
import networkx as nx
import argparse
from utils import *
import numpy as np
import copy
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='Choose GPU 0,1,2 or 3')
parser.add_argument('--model_type', type=str, default='gru', help='type of model, gru or gat')
parser.add_argument('--model_para', type=int, default=99, help='parameter of model, epoch')
parser.add_argument('--dataset', type=str, default="beijing", help='choose dataset')
args = parser.parse_args()

global road_graph
global search_graph
global id2road
global id2node
global road2id
global node2id
global id2length
global id2old_road
global current_data
global min_thres
global max_thres
global future_data

time = [340 + 500 - 420 + 1440 * 20, 400 + 500 - 420 + 1440 * 20, 560 + 500 - 420 + 1440 * 20]   #75

predict_model = None
current_data = {} # 一行为一条路 一列为一分钟
future_data = {}
min_thres = {}
max_thres = {}
 
 
 
def load_model():
  return (torch.load("/data/wuning/traffic-data/model/"  + args.dataset + "/" + args.model_type + "/" + str(args.model_para) + ".e"),
  torch.load("/data/wuning/traffic-data/model/"  + args.dataset + "/" + args.model_type + "/" + str(args.model_para) + ".d"),
  torch.load("/data/wuning/traffic-data/model/"  + args.dataset + "/" + args.model_type + "/" + str(args.model_para) + ".g"),
  torch.load("/data/wuning/traffic-data/model/"  + args.dataset + "/gru_gcn_tcn/28.h"))
   
def prepare_traffic_data(start_time, current_time, end_time):
   
  transfer_time_graph = []
  c_data = {}
  for t in range(end_time - current_time + 1):
    transfer_time_graph.append(nx.DiGraph())  

  for key in range(len(id2road)):
    ID = id2road[key] #snode_enode
#    print(key, ID)
    Flag = False
    global id2old_road
    try:
      sf = open("/data/wuning/traffic-data/raw/" + id2old_road[key][:6] + "_" + id2old_road[key][6] + id2old_road[key][7:].zfill(4), "rb")
      speeds = sf.readlines()
    except Exception as err:
      print(err)
      continue
      Flag = True
    speeds = speeds[0].split()
    c_data[key] = [float(speed) / 100 for speed in speeds[start_time : current_time]]  

    his_data = [float(speed) / 100 for speed in speeds[: current_time]]

#    print(his_data)
    min_thre = np.min(his_data)
    max_thre = np.max(his_data)
    global min_thres
    global max_thres
    min_thres[key] = min_thre
    max_thres[key] = max_thre
#    print(node2id)
    for t, g in zip(range(current_time - 1 , end_time), transfer_time_graph):
      lineCount = t        
      if Flag:
        travel_time = 10000   
      else:
        speed = float(speeds[lineCount])
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

#  print("c_data:", c_data)    
  return c_data, transfer_time_graph, max_thres, min_thres
def load_data(start_time, current_time, end_time):
  global search_graph
  global road_graph
  global id2road
  global id2node
  global id2length
  global node2id
  global road2id
  global id2old_road
  road_graph = pickle.load(open("/data/wuning/traffic-data/" + args.dataset + "/map/road_graph", "rb"))   # road为node
  search_graph = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/search_graph", "rb"))  # raw map
  id2road = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2road", "rb")) 
  id2node = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2node", "rb"))
  id2length = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2length", "rb"))
  node2id = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/node2id", "rb"))
  road2id = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/road2id", "rb"))
  id2old_road = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2old_road", "rb"))
  
#  print(road_graph)
#  for i in range(len(id2road)):
#    road2id[id2road[i]] = i
#  for i in range(len(id2node)):
#    node2id[id2node[i]] = i
  global current_data
  global future_data
  global min_thres
  global max_thres
  traffic_condition_data_path = "/data/wuning/traffic-data/" + args.dataset + "/" + str(start_time) + "_" + str(current_time) + "_" + str(end_time)  
  if os.path.exists(traffic_condition_data_path):
    temp = pickle.load(open(traffic_condition_data_path, "rb"))
    current_data = temp[0]
    future_data = temp[1]    
    max_thres = temp[2]
    min_thres = temp[3]
  else:
    current_data, future_data, max_thres, min_thres = prepare_traffic_data(start_time, current_time, end_time) 
    pickle.dump([current_data, future_data, max_thres, min_thres], open(traffic_condition_data_path, "wb"))
#  print(current_data)
def eval(input_tensor, 
     neighs_tensor, 
     neighs_time_tensor, 
     input_time_tensor, 
     output_time_tensor, 
     encoder, 
     decoder, 
     graph_attention):

  encoder_hidden = encoder.initHidden()
  encoder_hidden = encoder_hidden[:, 0, :].unsqueeze(0)
  try:
    input_speed = torch.tensor(input_tensor, dtype=torch.float, device=args.device)
    input_time = torch.tensor(input_time_tensor, dtype=torch.long, device=args.device)
    output_time = torch.tensor(output_time_tensor, dtype=torch.long, device=args.device)
    neighs_speed = torch.tensor(neighs_tensor, dtype=torch.float, device=args.device)
    neighs_time = torch.tensor(neighs_time_tensor, dtype=torch.long, device=args.device)
  except Exception as err:
    print(err)
    return 0, 0, 0
  input_length = input_speed.size(1)

  target_length = output_time.size(1)
#  print(input_length, input_speed.shape, input_time.shape)
  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_speed[:, ei], input_time[:, ei], encoder_hidden)
  if args.model_type == "gru_gcn" and len(neighs_tensor) > 0:
    main_hidden = encoder_hidden  
    neighs_hidden = []
    for neigh_speed, neigh_time in zip(neighs_speed, neighs_time):      
      encoder_hidden = encoder.initHidden()
      encoder_hidden = encoder_hidden[:, 0, :].unsqueeze(0)
      neigh_time = neigh_time.unsqueeze(0)
      for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(neigh_speed[:, ei], neigh_time[:, ei], encoder_hidden)
      neighs_hidden.append(encoder_hidden.squeeze().unsqueeze(0)) 
    encoder_hidden = graph_attention(main_hidden, neighs_hidden)
  else:
    pass

  decoder_input = torch.tensor([[2.0 for i in range(1)]], device=args.device) # start id 被设置成了 180.0

  decoder_hidden = encoder_hidden

  pred = []
  real = []
  decoder_output = None
  predict_list = []
  for di in range(target_length):
    decoder_output, decoder_hidden = decoder(decoder_input, output_time[:, di], decoder_hidden)
    predict_list.append(decoder_output[0, 0])
    decoder_input = decoder_output
#  print("decoder_output:", decoder_output)
  return predict_list

def predict_heuristic(time_tensor, path_tensor, heuristic_network):
  max_length = np.max([len(item) for item in path_tensor])
  padded_path_tensor = [] 
  seq_lengths = [] 
  i = 0

#  while i < len(path_tensor):
#   if len(path_tensor[i]) < 5:
#    path_tensor.pop(i)    
#    i -= 1
#   i += 1
 
  for i in range(len(path_tensor)):
   for j in range(i + 1, len(path_tensor)):
    if len(path_tensor[i]) < len(path_tensor[j]):
     temp = path_tensor[j]
     path_tensor[j] = path_tensor[i]
     path_tensor[i] = temp 
    
  for path in path_tensor: 
   seq_lengths.append(len(path)) 
   while len(path) < max_length: 
    path.append(0)   
   padded_path_tensor.append(path)  
  
  length_tensor = []
  for path in padded_path_tensor:
   path_length = []
   for link in path:
    path_length.append(int(id2length[link] / 0.05))    
   length_tensor.append(path_length)
  
  padded_path_tensor = torch.tensor(padded_path_tensor, dtype=torch.long, device=args.device) 
  input_path = torch.nn.utils.rnn.pack_padded_sequence(padded_path_tensor, seq_lengths, batch_first=True)
  try:
    input_length = torch.tensor(seq_lengths, dtype=torch.float, device=args.device)
    input_time = torch.tensor(time_tensor, dtype=torch.long, device=args.device)
    input_road_length = torch.tensor(length_tensor, dtype=torch.long, device=args.device)
  except Exception as err:
    print(err)
#    print("--------")
    return 0, 0, 0
  heuristic_hidden = heuristic_network.initHidden(len(path_tensor))
  pred = heuristic_network(input_time, padded_path_tensor, input_length, input_road_length, heuristic_hidden)
  return pred

def predict(current_time, departure_time, current, neigh, rG, encoder, decoder, graph_attention):
#  print("current:", current, "neigh:", neigh)
  road_id = road2id[str(current) + "_" + str(neigh)]
  global current_data 
  global min_thres
  global max_thres
  input_data = [np.array(current_data[road_id])]# - min_thres[road_id] / (max_thres[road_id] - min_thres[road_id])]
#  print("current_time:", current_time, departure_time, input_data, len(input_data[0]))
  if int(current_time) == int(departure_time):
   return id2length[road_id] / (input_data[0][-1] * 100) * 60

  input_time = [[]]
  steps = len(current_data[road_id])
  for i in range(departure_time - steps, departure_time):# 已知出发前的路况
    input_time[0].append((i % 1440) / 15)    
  output_time = [[]]
#  print("d-c time", departure_time, current_time)
  for i in range(int(departure_time), int(current_time) + 30):
    output_time[0].append((i % 1440) / 15)  
  neighs = rG[road_id]  
  neighs_data = [] 
  neighs_time_data = [] 
  for neigh in neighs:
   neighs_data.append([current_data[neigh]])# - min_thres[neigh] / (max_thres[neigh] - min_thres[neigh])]) 
   neigh_time = [] 
   for i in range(departure_time - steps, departure_time):# 已知出发前的路况
    neigh_time.append((i % 1440) / 15)   
   neighs_time_data.append(neigh_time)
  weights = eval(input_data, neighs_data, neighs_time_data, input_time, output_time, encoder, decoder, graph_attention)  
#  weights = [1 if weight == 0 for weight in weights]
#  print("weights:", weights)
  weights = np.array(weights) * 100#(max_thres[road_id] - min_thres[road_id]) + min_thres[road_id]
#  if weight == 0:
#    return 10  
#  print("prediction speed:", weight, "latest value", current_data[road_id][0] * 100)
  return id2length[road_id] / weights * 60

def get_time_on_road(G):
 residual = 1
 cost = 0
 while True:
  one_step_part = 1 / ((G[int(cost)]))
  if residual - one_step_part < 0:
   break  
  residual -= one_step_part
  cost += 1
 cost += residual * ((G[int(cost)]))  
 return cost


def simulate_time(path, start_time, G):
 cost = 0
 T = 0
 for i in range(len(path) - 2):
#  print(path[i], path[i + 1])
  cost = get_time_on_road([G[int(t)][path[i]][path[i + 1]]["weight"] for t in range(int(T), int(T) + 30)])
  T += cost
 return T 

#  residual = 1
#  while True:
#   one_step_part = 1 / ((G[int(T)][path[i]][path[i + 1]]["weight"]) * 60)
#   if residual - one_step_part < 0:
#    break  
#   residual -= one_step_part
#   T += 1 
#  T += residual * ((G[int(T)][path[i]][path[i + 1]]["weight"]) * 60)    

#  T += G[int(T)][path[i]][path[i + 1]]["weight"] * 60

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

def search(start_time, start, end, G, rG, lG, encoder, decoder, graph_attention, heuristic_network):
  results = []
  search_count = 0
  departure_time = start_time  
  global node2id
  start = node2id[start]
  end = node2id[end]
  closed_road_set = []
  closedSet = []
  openSet = [start]
  pathFounded = {start: [start]}
  gScore = {}
  fScore = {}
  gScore[start] = 0
  fScore[start] = 0
  bestScore = 0
  bestTra = []
  time = departure_time
  while len(openSet) > 0:
    search_count += 1
    current = openSet[0]
    openSet.remove(current)
    closedSet.append(current)
    if current == end:
      bestTra = copy.deepcopy(pathFounded[current])
      bestTra.append(end)
      bestScore = gScore[current]
      break
    time = departure_time + gScore[current]   
    for neigh in G[current]:
      if (neigh in closedSet):
        continue
      closed_road_set.append(id2old_road[road2id[str(id2node[current]) + "_" + str(id2node[neigh])]])  
      multi_step_values = predict(time, 
                  departure_time, 
                  id2node[current], 
                  id2node[neigh],
                  rG, 
                  encoder, 
                  decoder, 
                  graph_attention) # st_value[0][-1][waiting]
      if not isinstance(multi_step_values, list):
        multi_step_values = [future_data[int(t - start_time)][current][neigh]["weight"] for t in range(int(time), int(time) + 30)]            
      one_step_value = get_time_on_road(multi_step_values)  
      g_score = one_step_value + gScore[current]
      temp = copy.deepcopy(pathFounded[current])
      temp.append(neigh)
#      h_score = 0.0

      try:
       shortest_path_length = nx.shortest_path_length(lG, source=neigh, target=end, weight="weight")
       h_score = shortest_path_length 
       shortest_path = nx.shortest_path(lG, source=neigh, target=end, weight="weight")
       _, shortest_path = node_list_to_link_id_list(shortest_path)
       h_score = predict_heuristic([0], [shortest_path, shortest_path], heuristic_network)
       h_score = h_score[0] / 2
      except Exception as err:
       print("err:", err)
       h_score = 1
       
      if (neigh in gScore) and (g_score < gScore[neigh]):
        continue
      gScore[neigh] = g_score
      fScore[neigh] = gScore[neigh] + h_score
      if neigh not in openSet:
        openSet = insert(neigh, openSet, fScore)
      else:
        openSet = move(neigh, openSet, fScore)
      pathFounded[neigh] =  temp
  return bestTra, bestScore, search_count, closed_road_set

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
  while len(openSet) > 0:
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
#    time = 0
#    print(current)
    if time > 10000:
      continue    
#    print(time, G[int(time)][current], len(pathFounded.keys()))
    for neigh in G[int(time)][current]:
      if neigh in closedSet:
        continue
      if G[int(time)][current][neigh]["weight"] == 10000:
        continue
#      print("time:", time)  
      one_step_value = get_time_on_road([G[int(t)][current][neigh]["weight"] for t in range(int(time), int(time) + 30)])
#G[0][current][neigh]["weight"]

#      one_step_value = G[int(time)][current][neigh]["weight"] * 60
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
  return bestTra, bestScore  

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
    f_route.append(id2old_road[road2id[item]])
  return node_list, f_route

def node_list_to_link_id_list(route):
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

def generate_path(start_node, end_node):

  load_data(time[0], time[1], time[2])

  encoder, decoder, graph_attention, heuristic_network = load_model()
  adj = np.matrix(search_graph)
  length_adj = copy.deepcopy(search_graph)
  for i in range(len(length_adj)):
   for j in range(len(length_adj[i])):
    if length_adj[i][j] == 1: 
     length_adj[i][j] = id2length[road2id[str(id2node[i]) + "_" + str(id2node[j])]]   
  length_adj = np.matrix(length_adj) 

#  print(str(start_node) + ":", [id2node[n] for n in future_data[0].neighbors(node2id[start_node])])

  r_adj = np.matrix(road_graph)

  G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
  rG = nx.from_numpy_matrix(r_adj, create_using=nx.DiGraph())
  l_G = nx.from_numpy_matrix(length_adj, create_using=nx.DiGraph())
#  print("2-------")
#  print(time[1], node2id[start_node], node2id[end_node])
  real_route, real_time = time_dependent_path(time[1], node2id[start_node], node2id[end_node], future_data[1:])
#  print("compare:", real_route, time[1])
#  print("label time:", node_list_to_link_list(real_route), time[1], simulate_time(real_route[:-1], time[1], future_data[1:]))
#  print(future_data[1][node2id[start_node]], future_data[1][node2id[end_node]])
#  print(time[1], future_data[1])
  o_real_route = []
  for item in real_route:
    o_real_route.append(id2node[item])
#  print(o_real_route, real_time, node_list_to_link_list(real_route))

  route, costs, search_count, closed_set = search(time[1], start_node, end_node, G, rG, l_G, encoder, decoder, graph_attention, heuristic_network)
#  print("search_count:", search_count)
#  print(route) 
  print(closed_set)
  node_list, f_route = node_list_to_link_list(route)
#  print("zzzzz")  
  real_costs = simulate_time(route[:-1], time[1], future_data[1:])
#  print("-------", f_route, costs, real_costs)

  base_route = nx.shortest_path(future_data[1], source=node2id[start_node], target=node2id[end_node], weight="weight")
  base_route_length = nx.shortest_path_length(future_data[1], source=node2id[start_node], target=node2id[end_node], weight="weight")
#  print(base_route, "base_route_length:", base_route_length)
#  print("baseline:", base_route, [id2node[item] for item in base_route], node_list_to_link_list(base_route), simulate_time(base_route, time[1], future_data[1:]))
  return search_count, simulate_time(real_route[:-1], time[1], future_data[1:]), real_costs, simulate_time(base_route, time[1], future_data[1:])
 
def load_query(type):
  queries = pickle.load(open("/data/wuning/traffic-data/beijing/query/" + type + "_query", "rb"))
  return queries
 
def load_baseline(name, type):
  baseline_result = pickle.load(open("/data/jinyang/baseline/" + name + "/" + type + "_query_result", "rb"))
  return baseline_result
 
def main():
    
  #short [[-947, -902], [-1052, -1713], [-1074, -1648], [-1531, -1534], [-266, -826], [-1573, -673], [-377, -370], [-746, -615], [-1694, -1091], [-2801, -727]]
  #medium  [[-1170, -1726], [-1898, -1450], [-1418, -2179], [-931, -608], [-654, -1478], [-770, -1577], [-815, -1351], [-1582, -2774], [-1300, -702], [-1709, -2936]]
  #long [[-1025, '595661162'], [-2252, '5956631426'], [-1938, -718], [-1691, -361], [-1521, -375], [-1747, -679], [-1729, -360], [-2130, -1131], [-2156, -826], [-1666, -804]]
#  queries = [[-1025, '595661162'], [-2252, '5956631426'], [-1938, -718], [-1691, -361], [-1521, -375], [-1747, -679], [-1729, -360], [-2130, -1131], [-2156, -826], [-1666, -804]]
  # long query for beijing on FR1 FR2 
  #[[-1489, -1064], [-1833, -943], [-1489, -1064], [-309, -895], [-888, '5956721176'], [-1756, -1557], [-986, -1351]]

  queries = load_query("short")
#  queries = [[-1489, -1064], [-1833, -943], [-1489, -1064], [-309, -895], [-888, '5956721176'], [-1756, -1557], [-986, -1351]]
  baseline_result = load_baseline("T-drive", "long")
  search_total = 0
  query_count = 0
  for query in queries[:100]:
   search_count, optimal_time, real_cost, static_cost = generate_path(query[0], query[1])
   print("-----------")
   print("optimal_time:", optimal_time)
   print("our_time:", real_cost)
   print("static_time:", static_cost)
   
   try:
    baseline_route = [node2id[item] for item in baseline_result[(query[0], query[1], time[1] - 1440 * 20)]] 
   except Exception as err:
    continue    
   print("baseline_route:", baseline_route, time[1])
   baseline_time = simulate_time(baseline_route, time[1], future_data[1:])
   print(baseline_route, time[1])
   print("baseline_time:", baseline_time)
   search_total += search_count
   query_count += 1
  print("ave_roads:", search_total / query_count) 
if __name__ == '__main__':
  main()

