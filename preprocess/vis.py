import pickle
import networkx as nx
road_graph = pickle.load(open("/data/wuning/traffic-data/beijing/map/road_graph", "rb"))   # road为node
search_graph = pickle.load(open("/data/wuning/traffic-data/beijing/map/search_graph", "rb"))  # raw map
id2road = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2road", "rb")) 
id2node = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2node", "rb"))
id2length = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2length", "rb"))
node2id = pickle.load(open("/data/wuning/traffic-data/beijing/map/node2id", "rb"))
road2id = pickle.load(open("/data/wuning/traffic-data/beijing/map/road2id", "rb"))
id2old_road = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2old_road", "rb"))
  

def prepare_traffic_data(start_time, current_time, end_time):
     
  transfer_time_graph = []
  c_data = {}
  for t in range(end_time - current_time + 1):
    transfer_time_graph.append(nx.DiGraph())  
  road_speeds = []
  roads = []
  for key in range(len(id2road)): 
    ID = id2road[key] #snode_enode
#    print(key, ID)
    Flag = False
    try:
      sf = open("/data/wuning/traffic-data/raw/" + id2old_road[key][:6] + "_" + id2old_road[key][6] + id2old_road[key][7:].zfill(4), "rb")
      speeds = sf.readlines()
    except Exception as err:
      print(err)
      continue
      Flag = True
    speeds = speeds[0].split()
    speeds = [float(item) for item in speeds]
    roads.append(id2old_road[key])
    road_speeds.append(speeds[start_time: end_time])

#  print("c_data:", c_data)    
  return roads, road_speeds

road_IDs, road_data = prepare_traffic_data(360, 420, 480)  

pickle.dump(road_IDs, open("/data/wuning/traffic-data/beijing/vis_links", "wb"))
pickle.dump(road_data, open("/data/wuning/traffic-data/beijing/vis_data", "wb"))
