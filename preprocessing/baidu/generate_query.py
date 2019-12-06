import json
import random
import numpy as np
import pickle
import os

def euclid_distance(a, b):
 return np.sqrt(np.sum(((np.array(a) - np.array(b)) * np.array([111000, 0.667 * 111000]))**2))

link_cor_file = open("/data/wuning/traffic-data/baidu/map/link_gps")
link_info_file = open("/data/wuning/traffic-data/baidu/map/road_network_sub-dataset_max_graph_no_direct")

link_cor_lines = link_cor_file.readlines()

link_info_lines = link_info_file.readlines()

link2cor = {}
link2node = {}
links = []

links_list = []

for line in link_info_lines[1:]:
 links_list.append(line.split()[0]) 


for line in link_cor_lines:
 if not line.split()[0] in links_list:
  continue   
 temp = line.split()  
 links.append(temp[0])   
 link2cor[temp[0]] = [float(temp[1]), float(temp[2])]

#print(link_info_lines[1:], len(link_info_lines))
count = 0
for line in link_info_lines[1:]:
 count += 1 
 temp = line.split()  
# print("temp:", temp[0], temp[1], temp[2])
 link2node[temp[0]] = temp[3]



query_num = 50000000
short_query = []
medium_query = []
long_query = []
       
  
for i in range(query_num):
 ind1 = random.randint(0, len(links) - 1)
 start = link2node[links[ind1]]
 start_cor = link2cor[links[ind1]]
 
 ind2 = random.randint(0, len(links) - 1)
 end = link2node[links[ind2]]
 end_cor = link2cor[links[ind2]] 
# print(euclid_distance(start_cor, end_cor)) 
 if euclid_distance(start_cor, end_cor) < 10000:
   short_query.append([start, end])
 if euclid_distance(start_cor, end_cor) > 10000 and euclid_distance(start_cor, end_cor) < 15000:    
   medium_query.append([start, end])     
 if euclid_distance(start_cor, end_cor) > 15000:    
   long_query.append([start, end])     
 print(len(long_query))  
 if len(short_query) > 1000 and len(medium_query) > 1000 and len(long_query) > 1000:
   short_query = short_query[:1000]     
   medium_query = medium_query[:1000]    
   long_query = long_query[:1000]      
   break

#print("short", short_query, "medium", medium_query, "long", long_query)  

pickle.dump(short_query, open("/data/wuning/traffic-data/baidu/query/short_query_large", "wb"))
pickle.dump(medium_query, open("/data/wuning/traffic-data/baidu/query/medium_query_large", "wb"))
pickle.dump(long_query, open("/data/wuning/traffic-data/baidu/query/long_query_large", "wb"))
