import json
import random
import numpy as np
import pickle
import os

def euclid_distance(a, b):
 return np.sqrt(np.sum(((np.array(a) - np.array(b)) * np.array([111000, 0.667 * 111000]))**2))

link_net = json.load(open("/data/wuning/traffic-data/pems/new_road_net_pems_v1.json", "r"))

sub_graph = []
count = 0
for k in link_net.keys():
 sub_graph.append(k) 
 count += 1
print("count", count)

link_info_file = open("/data/wuning/traffic-data/pems/pems_data_process/pems_data_process/pems", "r")

link_info_lines = link_info_file.readlines()

link2cor = {}
link2node = {}
links = []
for line in link_info_lines[1:]:
 if not line.split(",")[0] in sub_graph: 
  continue  
 temp = line.split(",")  
 links.append(temp[0])   
 if len(temp[8]) == 0 or len(temp[9]) == 0:
  continue     
 link2cor[temp[0]] = [float(temp[8]), float(temp[9])]




query_num = 50000
short_query = []
medium_query = []
long_query = []
       
  
for i in range(query_num):
 ind1 = random.randint(0, len(links) - 1)
 start = links[ind1]
 if not links[ind1] in link2cor:
  continue    

 start_cor = link2cor[links[ind1]]
 
 ind2 = random.randint(0, len(links) - 1)

 if not links[ind2] in link2cor:
  continue    

 end = links[ind2]
 end_cor = link2cor[links[ind2]] 
# print(euclid_distance(start_cor, end_cor)) 
 if euclid_distance(start_cor, end_cor) < 20000:
   short_query.append([start, end])
 if euclid_distance(start_cor, end_cor) > 20000 and euclid_distance(start_cor, end_cor) < 50000:    
   medium_query.append([start, end])     
 if euclid_distance(start_cor, end_cor) > 50000:    
   long_query.append([start, end])     
 if len(short_query) > 1000 and len(medium_query) > 1000 and len(long_query) > 1000:
   short_query = short_query[:1000]     
   medium_query = medium_query[:1000]    
   long_query = long_query[:1000]      
   break

#print("short", short_query, "medium", medium_query, "long", long_query)  

pickle.dump(short_query, open("/data/wuning/traffic-data/pems/query/short_query", "wb"))
pickle.dump(medium_query, open("/data/wuning/traffic-data/pems/query/medium_query", "wb"))
pickle.dump(long_query, open("/data/wuning/traffic-data/pems/query/long_query", "wb"))
