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
parser.add_argument('--dataset', type=str, default="baidu", help='choose dataset')
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
global hash2id
day = 0
time = [0 + 96 * day, 30 + 96 * day, 190 + 96 * day]   #75

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
    old_ID = id2old_road[key]
    try:
      sf = open("/data/wuning/traffic-data/baidu/TrafficData/" + hash2id[old_ID], "rb")
      speeds = sf.readlines()
    except Exception as err:
      print(err)
      continue
      Flag = True
#    print("speeds:", speeds)  
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
  global hash2id
  road_graph = pickle.load(open("/data/wuning/traffic-data/" + args.dataset + "/map/road_graph", "rb"))   # road为node
  search_graph = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/search_graph", "rb"))  # raw map
  id2road = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2road", "rb")) 
  id2node = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2node", "rb"))
  id2length = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2length", "rb"))
  node2id = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/node2id", "rb"))
  road2id = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/road2id", "rb"))
  id2old_road = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/id2old_road", "rb"))
  hash2id = pickle.load(open("/data/wuning/traffic-data/" + args.dataset +"/map/hash2id", "rb"))
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
  one_step_part = 15 / ((G[int(cost / 15)]))
  if residual - one_step_part < 0:
   break  
  residual -= one_step_part
  cost += 15
 cost += residual * ((G[int(cost / 15)]))  
# print("cost:", G[int(cost / 15)])
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
      h_score = 0.0

#      try:
#       shortest_path_length = nx.shortest_path_length(lG, source=neigh, target=end, weight="weight")
#       h_score = shortest_path_length 
#       shortest_path = nx.shortest_path(lG, source=neigh, target=end, weight="weight")
#       _, shortest_path = node_list_to_link_id_list(shortest_path)
#       h_score = predict_heuristic([0], [shortest_path, shortest_path], heuristic_network)
#       h_score = h_score[0] / 2
#      except Exception as err:
#       print("err:", err)
#       h_score = 1
       
      if (neigh in gScore) and (g_score < gScore[neigh]):
        continue
      gScore[neigh] = g_score
      fScore[neigh] = gScore[neigh] + h_score
      if neigh not in openSet:
        openSet = insert(neigh, openSet, fScore)
      else:
        openSet = move(neigh, openSet, fScore)
      pathFounded[neigh] =  temp
  return bestTra, bestScore, search_count

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
      one_step_value = get_time_on_road([G[int(t)][current][neigh]["weight"] for t in range(int(time), int(time + 30))])
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
  if(real_route == 0):
   return -1, -1, -1, -1  
  print(real_route, real_time, node2id[start_node], node2id[end_node])
  print("label time:", node_list_to_link_list(real_route), time[1], simulate_time(real_route[:-1], time[1], future_data[1:]))
#  print(future_data[1][node2id[start_node]], future_data[1][node2id[end_node]])
#  print(time[1], future_data[1])
  o_real_route = []
  for item in real_route:
    o_real_route.append(id2node[item])
#  print(o_real_route, real_time, node_list_to_link_list(real_route))

#  route, costs, search_count = search(time[1], start_node, end_node, G, rG, l_G, encoder, decoder, graph_attention, heuristic_network)
  route, costs, search_count = (-1, -1, -1)

#  print("search_count:", search_count)
#  print(route) 
#  node_list, f_route = node_list_to_link_list(route)
#  print("zzzzz")  
#  real_costs = simulate_time(route[:-1], time[1], future_data[1:])
  real_costs = -1
#  print("-------", f_route, costs, real_costs)

  base_route = nx.shortest_path(future_data[1], source=node2id[start_node], target=node2id[end_node], weight="weight")
  base_route_length = nx.shortest_path_length(future_data[1], source=node2id[start_node], target=node2id[end_node], weight="weight")
#  print(base_route, "base_route_length:", base_route_length)
#  print("baseline:", base_route, [id2node[item] for item in base_route], node_list_to_link_list(base_route), simulate_time(base_route, time[1], future_data[1:]))
  return search_count, simulate_time(real_route[:-1], time[1], future_data[1:]), real_costs, simulate_time(base_route, time[1], future_data[1:])
 
def load_query(type):
  queries = pickle.load(open("/data/wuning/traffic-data/baidu/query/" + type + "_query", "rb"))
  return queries
 
def load_baseline(name, type):
  baseline_result = pickle.load(open("/data/jinyang/baseline/" + name + "/" + type + "_query_result", "rb"))
  return baseline_result
 
def main():
    
  #short [[-947, -902], [-1052, -1713], [-1074, -1648], [-1531, -1534], [-266, -826], [-1573, -673], [-377, -370], [-746, -615], [-1694, -1091], [-2801, -727]]
  #medium  [[-1170, -1726], [-1898, -1450], [-1418, -2179], [-931, -608], [-654, -1478], [-770, -1577], [-815, -1351], [-1582, -2774], [-1300, -702], [-1709, -2936]]
  #long [[-1025, '595661162'], [-2252, '5956631426'], [-1938, -718], [-1691, -361], [-1521, -375], [-1747, -679], [-1729, -360], [-2130, -1131], [-2156, -826], [-1666, -804]]
#  queries = [[-1025, '595661162'], [-2252, '5956631426'], [-1938, -718], [-1691, -361], [-1521, -375], [-1747, -679], [-1729, -360], [-2130, -1131], [-2156, -826], [-1666, -804]]
 
#  queries = load_query("short")
#  queries = [['1520409228', '1520418272'], ['1530610281', '1520414942'], ['1520405646', '1553397849'], ['1520412496', '1520403967'], ['1520402656', '1520404199'], ['1553397881', '1520419765'], ['1531267467', '1520412667'], ['1520407636', '1549771442'], ['1554430289', '1552895290'], ['1520407809', '1520404840'], ['1520413547', '1520405770'], ['1520407934', '1530612376'], ['1520411639', '1520418795'], ['1520408372', '1520405852'], ['1520403791', '1520402676'], ['1550584812', '1520419469'], ['1520403872', '1554152985'], ['1520409228', '1554152450'], ['1550585127', '1530611729'], ['1531457463', '1520414025'], ['1520413233', '1530990509'], ['1520409002', '1520417878'], ['1520408876', '1553398064'], ['1530829587', '1530760694'], ['1530050935', '1549771765'], ['1520412808', '1549446982'], ['1520408046', '1520418970'], ['1520407537', '1520417892'], ['1532872213', '1520403101'], ['1520402908', '1520421045'], ['1520417736', '1520409725'], ['1520402763', '1520405589'], ['1553398016', '1549446923'], ['1553533650', '1531235046'], ['1529841460', '1520408828'], ['1520408250', '1520408879'], ['1520410361', '1520413066'], ['1520418851', '1553398464'], ['1529648022', '1520403779'], ['1520405594', '1520402831'], ['1520413861', '1520409701'], ['1549771314', '1520412703'], ['1520403779', '1520403586'], ['1520404542', '1520417933'], ['1532871140', '1554152452'], ['1520417997', '1520408691'], ['1520414267', '1531223782'], ['1549315548', '1532874049'], ['1531375273', '1531353723'], ['1520417673', '1520405065'], ['1520420310', '1520412888'], ['1549772174', '1530611720'], ['1520417684', '1520409516'], ['1520402926', '1520420002'], ['1549338526', '1520402646'], ['1554151833', '1532871096'], ['1520415652', '1520413065'], ['1520414950', '1520402576'], ['1520419950', '1520407490'], ['1520413079', '1530826512'], ['1530610241', '1520409708'], ['1554152808', '1520411992'], ['1520422901', '1520409516'], ['1530715815', '1530772183'], ['1520414175', '1553533891'], ['1520419430', '1553533471'], ['1520408576', '1520413074'], ['1520417878', '1530841043'], ['1520414513', '1520410776'], ['1520418824', '1520412960'], ['1520417673', '1554151551'], ['1555291057', '1554152448'], ['1552895415', '1520412902'], ['1520403777', '1520403329'], ['1520407615', '1554430283'], ['1520413130', '1530610239'], ['1520412904', '1520408799'], ['1520408814', '1530825478'], ['1520421924', '1520419765'], ['1530697391', '1520410800'], ['1520408712', '1531075871'], ['1520408799', '1520418051'], ['1531074053', '1530611750'], ['1520419021', '1520412909'], ['1549446923', '1530610338'], ['1532873342', '1520418359'], ['1520408828', '1520407815'], ['1520404106', '1531363479'], ['1520407784', '1520408706'], ['1520403957', '1554151517'], ['1520412966', '1520419117'], ['1554430310', '1520412934'], ['1520413502', '1549316610'], ['1520410539', '1530610241'], ['1520415076', '1530860966'], ['1520419704', '1520413706'], ['1520404020', '1553533897'], ['1520407495', '1530810213'], ['1520422915', '1532871312'], ['1520417897', '1520405621']]
#  queries = [['1520407398', '1520411633'], ['1520422864', '1520416832'], ['1520416840', '1520417939'], ['1520416838', '1530875471'], ['1553533477', '1532872213'], ['1554152964', '1520422031'], ['1553398020', '1520422027'], ['1530610353', '1520416721'], ['1532871298', '1520412934'], ['1520407625', '1520416830'], ['1553533871', '1520416365'], ['1520417997', '1531362570'], ['1520421570', '1520412535'], ['1520407773', '1520416831'], ['1520411096', '1520412478'], ['1520411633', '1531363479'], ['1520416830', '1520407390'], ['1530867432', '1520416827'], ['1520411629', '1520412703'], ['1530728989', '1520412478'], ['1520406152', '1520407432'], ['1554152999', '1520422027'], ['1520417684', '1520416347'], ['1520418023', '1554153081'], ['1520407523', '1520406514'], ['1520407543', '1520416831'], ['1520407415', '1520422031'], ['1531363479', '1520406311'], ['1520412655', '1520421771'], ['1520422005', '1520412848'], ['1520412523', '1530728989'], ['1520416831', '1520412855'], ['1532871298', '1520417833'], ['1520416745', '1530990509'], ['1520416687', '1530990509'], ['1520412640', '1520406539'], ['1520417717', '1520416125'], ['1520407490', '1520416831'], ['1520406514', '1520402701'], ['1520402549', '1529655102'], ['1549315911', '1520422937'], ['1520406540', '1520412851'], ['1520416365', '1553533529'], ['1520406523', '1530948611'], ['1520407539', '1520421771'], ['1549339042', '1520406540'], ['1549315704', '1520416828'], ['1520407400', '1520411633'], ['1520412929', '1530728989'], ['1529655098', '1532871317'], ['1553533871', '1532872268'], ['1520406539', '1549772328'], ['1520406474', '1520402548'], ['1520411633', '1554153575'], ['1520412929', '1520406540'], ['1520422005', '1520417916'], ['1520407792', '1549707769'], ['1520402676', '1520421924'], ['1520416828', '1520407680'], ['1554153081', '1530752783'], ['1529655102', '1520422864'], ['1520407523', '1520416836'], ['1520417727', '1555291064'], ['1520421771', '1553398251'], ['1520422024', '1520412478'], ['1520416714', '1520417673'], ['1520417692', '1520406311'], ['1520422873', '1520416831'], ['1520416840', '1520417706'], ['1531100140', '1520422031'], ['1530611527', '1532871298'], ['1520406474', '1520412675'], ['1529655098', '1520407415'], ['1520407369', '1520416365'], ['1554153081', '1520412768'], ['1553398020', '1530791165'], ['1520406523', '1520417939'], ['1554152964', '1520416831']]

#  queries = [['1520406474','1520412675'],['1520416840','1520417706'],['1554153081','1530752783'],['1520411633','1554153575'],['1520406539','1549772328'],['1520407490','1520416831'],['1529655098','1520407415']]
  queries = [['1531076465', '1554158488'], ['1530637484', '1520485599'], ['1549278882', '1520486720'], ['1520482970', '1549341280'], ['1520485394', '1530684189'], ['1520497221', '1520491832'], ['1520489590', '1520500518'], ['1549759487', '1520488220'], ['1534688573', '1520492474'], ['1520480562', '1554156728'], ['1520485094', '1520492865'], ['1520485325', '1552896142'], ['1531464197', '1531141191'], ['1530862021', '1520481049'], ['1549277954', '1520486093'], ['1520482931', '1520492614'], ['1520485690', '1529638970'], ['1553536758', '1520487717'], ['1520488879', '1534688286'], ['1520481404', '1549758596'], ['1553398504', '1520489256'], ['1554157077', '1520484744'], ['1549759302', '1520500181'], ['1520485723', '1520498734'], ['1520490203', '1520478756'], ['1520486199', '1520490560'], ['1520486276', '1520493994'], ['1554158080', '1530739898'], ['1520488417', '1520500154'], ['1549276226', '1549758700'], ['1531044882', '1520489540'], ['1520481923', '1553536619'], ['1554157116', '1530684947'], ['1520487667', '1520482057'], ['1530638721', '1520479955']]
  baseline_result = load_baseline("T-drive", "long")
  search_total = 0
  query_count = 0
  for query in queries[0:100]:
   search_count, optimal_time, real_cost, static_cost = generate_path(query[0], query[1])
   print("-----------")
   print("optimal_time:", optimal_time, query[0], query[1])
   print("our_time:", real_cost)
   print("static_time:", static_cost)

   try:
    baseline_route = [node2id[item] for item in baseline_result[(query[0], query[1], time[1] - 1440 * 0 + 300)]] 
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

