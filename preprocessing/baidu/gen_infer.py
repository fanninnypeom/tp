import os
import pickle
import numpy as np
import json
np.seterr(all='raise') 

#road_graph = pickle.load(open("/data/wuning/traffic-data/" + args.dataset + "/map/road_graph", "rb"))   # road为node
#search_graph = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/search_graph", "rb"))  # raw map
#id2road = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/id2road", "rb")) 
#id2node = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/id2node", "rb"))
#id2length = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/id2length", "rb"))

link_cor_file = open("/data/wuning/traffic-data/baidu/map/link_gps_max_graph")
link_info_file = open("/data/wuning/traffic-data/baidu/map/road_network_sub-dataset_max_graph_no_direct")
link_hash_file = open("/data/wuning/traffic-data/baidu/map/link_id_hash_map.txt")

link_cor_lines = link_cor_file.readlines()
link_info_lines = link_info_file.readlines()
link_hash_lines = link_hash_file.readlines()

hash2id = {}

#------------
for line in link_hash_lines:
 hash2id[line.split()[1]] = line.split()[0]
#------------

data = {}
nodeSet = set()
id2road = []
id2node = []
id2length = []  
road2id = {}
node2id = {}

r2id = {}
n2id = {}

road_graph = None
search_graph = None


sub_graph = []


for line in link_info_lines[1:]:
 road = {}
 road["SNodeLong"] = line.split()[3]
 road["ENodeLong"] = line.split()[4]
 road["old_ID"] = line.split()[0]
 road["Length"] = float(line.split()[5])
 sub_graph.append(road)


for road in sub_graph:
 if not road["SNodeLong"] in id2node:
  id2node.append(road["SNodeLong"])      
 if not road["ENodeLong"] in id2node:
  id2node.append(road["ENodeLong"])      
 sid = id2node.index(road["SNodeLong"])
 eid = id2node.index(road["ENodeLong"])  
 node2id[road["SNodeLong"]] = sid 
 node2id[road["ENodeLong"]] = eid
 r2id[road["old_ID"]] = len(id2road) 
 road2id[str(road["SNodeLong"]) + "_" + str(road["ENodeLong"])] = len(id2road)
 id2road.append(str(road["SNodeLong"]) + "_" + str(road["ENodeLong"])) 
 
 id2length.append(float(road["Length"])) 

#---------------------

roads_adj = []
snode_list = []
enode_list = []
road_list = []
for road in sub_graph:
 snode_list.append(road["SNodeLong"])
 enode_list.append(road["ENodeLong"])
 road_list.append(road["old_ID"])

for item, s, e in zip(road_list, snode_list, enode_list):
 if e in snode_list:
  next_list = [i for i, x in enumerate(snode_list) if x == e]   
  for ne in next_list:
   roads_adj.append([item, road_list[ne]])        


#---------------------

road_graph = [[0 for i in range(len(id2road))] for i in range(len(id2road))]    
node_graph = [[0 for i in range(len(id2node))] for i in range(len(id2node))] 

for road in sub_graph:
 sid = node2id[road["SNodeLong"]] 
 eid = node2id[road["ENodeLong"]]
 node_graph[sid][eid] = 1

for road in roads_adj:
 try: 
  road_graph[r2id[road[0]]][r2id[road[1]]] = 1
 except Exception as err:
  pass  
#  road_graph = pickle.load(open("/data/wuning/traffic-data/" + args.dataset + "/map/road_graph", "rb"))   # road为node
#  search_graph = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/search_graph", "rb"))  # raw map
#  id2road = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/id2road", "rb")) 
#  id2node = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/id2node", "rb"))
#  id2length = pickle.load(open("/data/wuning/traffic-data/beijing/" + args.dataset +"/map/id2length", "rb"))

pickle.dump(road_graph, open("/data/wuning/traffic-data/baidu/map/road_graph", "wb"))
pickle.dump(node_graph, open("/data/wuning/traffic-data/baidu/map/search_graph", "wb"))
pickle.dump(id2road, open("/data/wuning/traffic-data/baidu/map/id2road", "wb"))
pickle.dump(id2node, open("/data/wuning/traffic-data/baidu/map/id2node", "wb"))
pickle.dump(id2length, open("/data/wuning/traffic-data/baidu/map/id2length", "wb"))
pickle.dump(road2id, open("/data/wuning/traffic-data/baidu/map/road2id", "wb"))
pickle.dump(node2id, open("/data/wuning/traffic-data/baidu/map/node2id", "wb"))

id2old_road = {}
for k in r2id:
  id2old_road[r2id[k]] = k

pickle.dump(id2old_road, open("/data/wuning/traffic-data/baidu/map/id2old_road", "wb"))

pickle.dump(hash2id, open("/data/wuning/traffic-data/baidu/map/hash2id", "wb"))







