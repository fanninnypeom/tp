import json
import random
import numpy as np
import pickle
import os

def euclid_distance(a, b):
 return np.sqrt(np.sum(((np.array(start_cor) - np.array(end_cor)) * np.array([0.667 * 111000, 111000]))**2))
roads = json.load(open("/data/wuning/traffic-data/beijing/map/RTICLinksNew.json","r"))
query_num = 5000
short_query = []
medium_query = []
long_query = []

files = os.listdir("/data/wuning/traffic-data/raw")
roads_with_data = []
for filename in files:
  roads_with_data.append(filename)


for i in range(query_num):
 ind1 = random.randint(0, len(roads["features"]) - 1)
 start = roads["features"][ind1]["properties"]["SNodeLong"]
 start_cor = roads["features"][ind1]["geometry"]["coordinates"][0]
 ind2 = random.randint(0, len(roads["features"]) - 1)
 end = roads["features"][ind2]["properties"]["SNodeLong"]
 end_cor = roads["features"][ind2]["geometry"]["coordinates"][0]

 if not (roads["features"][ind1]["properties"]["MapID"] + "_" + roads["features"][ind1]["properties"]["Kind"] + roads["features"][ind1]["properties"]["ID"].zfill(4) in roads_with_data and roads["features"][ind2]["properties"]["MapID"] + "_" + roads["features"][ind2]["properties"]["Kind"] + roads["features"][ind2]["properties"]["ID"].zfill(4) in roads_with_data):
  continue  

 if euclid_distance(start_cor, end_cor) > 4500 and euclid_distance(start_cor, end_cor) < 5500 :
   short_query.append([start, end])
 if euclid_distance(start_cor, end_cor) > 9000 and euclid_distance(start_cor, end_cor) < 11000:    
   medium_query.append([start, end])     
 if euclid_distance(start_cor, end_cor) > 19000 and euclid_distance(start_cor, end_cor) < 21000:    
   long_query.append([start, end])     
 if len(short_query) > 10 and len(medium_query) > 10 and len(long_query) > 10:
   short_query = short_query[:10]     
   medium_query = medium_query[:10]    
   long_query = long_query[:10]      
   break
print("short", short_query, "medium", medium_query, "long", long_query)  
# print(len(short_query), len(medium_query), len(long_query))
#pickle.dump(short_query, open("/data/wuning/traffic-data/beijing/query/short_query", "wb"))
#pickle.dump(medium_query, open("/data/wuning/traffic-data/beijing/query/medium_query", "wb"))
#pickle.dump(long_query, open("/data/wuning/traffic-data/beijing/query/long_query", "wb"))
