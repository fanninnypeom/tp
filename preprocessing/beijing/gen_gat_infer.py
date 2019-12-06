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



batch_size = 100
input_length = 30
time_interval = 15    # 20min 一个interval
pred_length = 30
root_dir = "/data/wuning/traffic-data/raw"
adj_dir = "/data/wuning/traffic-data/beijing/map/RTICLinksNew.json"   #RTICLinksTriple.json
road_adj_dir = "/data/wuning/traffic-data/beijing/map/RTICLinksTriple.json"   #RTICLinksTriple.json


roads = json.load(open(adj_dir, "r"))
roads_adj = json.load(open(road_adj_dir, "r"))


files = os.listdir(root_dir)
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

roads_with_data = []
for filename in files:
  roads_with_data.append(filename)
sub_graph = []

for road in roads["features"]:
  if road["properties"]["MapID"] + "_" + road["properties"]["Kind"] + road["properties"]["ID"].zfill(4) in roads_with_data:
    sub_graph.append(road)  

for road in sub_graph:
 if not road["properties"]["SNodeLong"] in id2node:
  id2node.append(road["properties"]["SNodeLong"])      
 if not road["properties"]["ENodeLong"] in id2node:
  id2node.append(road["properties"]["ENodeLong"])      
 sid = id2node.index(road["properties"]["SNodeLong"])
 eid = id2node.index(road["properties"]["ENodeLong"])  
 node2id[road["properties"]["SNodeLong"]] = sid 
 node2id[road["properties"]["ENodeLong"]] = eid
  
 r2id[road["properties"]["MapID"] + road["properties"]["Kind"] + road["properties"]["ID"]] = len(id2road) 
 road2id[str(road["properties"]["SNodeLong"]) + "_" + str(road["properties"]["ENodeLong"])] = len(id2road)
 id2road.append(str(road["properties"]["SNodeLong"]) + "_" + str(road["properties"]["ENodeLong"])) 
 
 id2length.append(float(road["properties"]["Length"])) 

road_graph = [[0 for i in range(len(id2road))] for i in range(len(id2road))]    
node_graph = [[0 for i in range(len(id2node))] for i in range(len(id2node))] 

for road in sub_graph:
 sid = node2id[road["properties"]["SNodeLong"]] 
 eid = node2id[road["properties"]["ENodeLong"]]
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

pickle.dump(road_graph, open("/data/wuning/traffic-data/beijing/map/road_graph", "wb"))
pickle.dump(node_graph, open("/data/wuning/traffic-data/beijing/map/search_graph", "wb"))
pickle.dump(id2road, open("/data/wuning/traffic-data/beijing/map/id2road", "wb"))
pickle.dump(id2node, open("/data/wuning/traffic-data/beijing/map/id2node", "wb"))
pickle.dump(id2length, open("/data/wuning/traffic-data/beijing/map/id2length", "wb"))
pickle.dump(road2id, open("/data/wuning/traffic-data/beijing/map/road2id", "wb"))
pickle.dump(node2id, open("/data/wuning/traffic-data/beijing/map/node2id", "wb"))

id2old_road = {}
for k in r2id:
  id2old_road[r2id[k]] = k
pickle.dump(id2old_road, open("/data/wuning/traffic-data/beijing/map/id2old_road", "wb"))








