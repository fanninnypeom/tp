import pickle
import torch
import copy
import os
def get_train_data(dataset, model_name):
  f = open("/data/wuning/traffic-data/" + dataset + "/" + model_name + "/train", "rb")  #train_norm_5000 train_v2 train_v3_no_scale
  train_data = pickle.load(f)
  return train_data
def generate_query(start_time):
  waiting_list = [[-1126, -866]]  #[-492, -799 -1126,-866], 
  for waiting in waiting_list:
    yield waiting[0], waiting[1], start_time  

def get_heuristic_data(dataset):
  root_dir = "/data/wuning/traffic-data/" + dataset + "/heuristic"
  files = os.listdir(root_dir)
  roads_with_data = []
  heuristic_train_batch = []
  for filename in files:
   data = pickle.load(open(root_dir + "/" + filename, "rb"))
   heuristic_train_batch.extend(data[:-1])
  return heuristic_train_batch
