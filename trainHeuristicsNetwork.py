from model.heuristics_network import *
from model.gru_gcn import *
from model.tcn import TCN
from utils import *
from torch import optim
import numpy as np
import time
import random
import sys
import pickle
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

input_length = 300
tcn_length = 300
pred_length = 30
#print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "beijing"        # baidu  beijing  P7 

model_name = "gru_gcn_tcn"         # gru  gru_gcn gru_gcn_tcn

model_type = "gru_gcn_tcn"

n_channels = [50] * 3

kernel_size = 5

tcn_dropout = 0.0


steps = 25

teacher_forcing_ratio = 0.0
test_teacher_forcing_ratio = 0

id2length = pickle.load(open("/data/wuning/traffic-data/beijing/map/id2length", "rb"))



def load_model():
  return 0, 0, 0
#  return (torch.load("/data/wuning/traffic-data/model/"  + dataset + "/" + model_name + "/70.e"),
#  torch.load("/data/wuning/traffic-data/model/"  + dataset + "/" + model_name + "/70.d"),
#  torch.load("/data/wuning/traffic-data/model/"  + dataset + "/" + model_name + "/70.g"))
 
 
def train(time_tensor,
          path_tensor, 
          label_tensor,  
          encoder, 
          decoder, 
          graph_attention,
          heuristic, 
          heuristic_optimizer, 
          criterion, 
          max_length=30):
  
  
  heuristic_optimizer.zero_grad()
  max_length = np.max([len(item[1]) for item in path_tensor])
  padded_path_tensor = [] 
  seq_lengths = [] 
  i = 0
  while i < len(path_tensor):
   if len(path_tensor[i][1]) < 5:
    path_tensor.pop(i)    
    label_tensor.pop(i)
    i -= 1
   i += 1


  for i in range(len(path_tensor)):
   for j in range(i + 1, len(path_tensor)):
    if len(path_tensor[i][1]) < len(path_tensor[j][1]):
     temp = path_tensor[j]
     path_tensor[j] = path_tensor[i]
     path_tensor[i] = temp 
     temp_label = label_tensor[j]
     label_tensor[j] = label_tensor[i]
     label_tensor[i] = temp_label 
    
  for path in path_tensor: 
   seq_lengths.append(len(path[1])) 
   while len(path[1]) < max_length: 
    path[1].append(0)   
   padded_path_tensor.append(path[1])  
  
  
  length_tensor = []
  for path in padded_path_tensor:
   path_length = []
   for link in path:
    path_length.append(int(id2length[link] / 0.05))    
   length_tensor.append(path_length)
  
  
  padded_path_tensor = torch.tensor(padded_path_tensor, dtype=torch.long, device=device) 
  input_path = torch.nn.utils.rnn.pack_padded_sequence(padded_path_tensor, seq_lengths, batch_first=True)
  try:
    input_length = torch.tensor(seq_lengths, dtype=torch.float, device=device)
    input_time = torch.tensor(time_tensor, dtype=torch.long, device=device)
    input_road_length = torch.tensor(length_tensor, dtype=torch.long, device=device)
    output_label = torch.tensor(label_tensor, dtype=torch.float, device=device)
  except Exception as err:
    print(err)
#    print("--------")
    return 0, 0, 0
  heuristic_hidden = heuristic.initHidden(len(path_tensor))
  pred = heuristic(input_time, padded_path_tensor, input_length, input_road_length, heuristic_hidden)
#  print(pred, output_label)
#  output_label = output_label.unsqueeze(1)
  loss = criterion(pred, output_label)
#  print("pred:", pred, "label:", output_label)
  loss.backward()
 
  heuristic_optimizer.step()
#  print("raw_loss:", loss)
  return loss.item()

def test(time_tensor, 
         path_tensor, 
         label_tensor,  
         encoder, 
         decoder, 
         graph_attention, 
         heuristic,
         criterion, 
         max_length=30):
 

  max_length = np.max([len(item[1]) for item in path_tensor])
  padded_path_tensor = [] 
  seq_lengths = [] 
  i = 0
  while i < len(path_tensor):
   if len(path_tensor[i][1]) < 5:
    path_tensor.pop(i)    
    label_tensor.pop(i)
    i -= 1
   i += 1
 
 
  for i in range(len(path_tensor)):
   for j in range(i + 1, len(path_tensor)):
    if len(path_tensor[i][1]) < len(path_tensor[j][1]):
     temp = path_tensor[j]
     path_tensor[j] = path_tensor[i]
     path_tensor[i] = temp 
     temp_label = label_tensor[j]
     label_tensor[j] = label_tensor[i]
     label_tensor[i] = temp_label 
    
  for path in path_tensor: 
   seq_lengths.append(len(path[1])) 
   while len(path[1]) < max_length: 
    path[1].append(0)   
   padded_path_tensor.append(path[1])  
  
  
  length_tensor = []
  for path in padded_path_tensor:
   path_length = []
   for link in path:
    path_length.append(int(id2length[link] / 0.05))    
   length_tensor.append(path_length)
  
  
  padded_path_tensor = torch.tensor(padded_path_tensor, dtype=torch.long, device=device) 
  input_path = torch.nn.utils.rnn.pack_padded_sequence(padded_path_tensor, seq_lengths, batch_first=True)
  try:
    input_length = torch.tensor(seq_lengths, dtype=torch.float, device=device)
    input_time = torch.tensor(time_tensor, dtype=torch.long, device=device)
    input_road_length = torch.tensor(length_tensor, dtype=torch.long, device=device)
    output_label = torch.tensor(label_tensor, dtype=torch.float, device=device)
  except Exception as err:
    print(err)
#    print("--------")
    return 0, 0, 0
  heuristic_hidden = heuristic.initHidden(len(path_tensor))
  pred = heuristic(input_time, padded_path_tensor, input_length, input_road_length, heuristic_hidden)
#  print(pred, output_label)
#  output_label = output_label.unsqueeze(1)
  loss = criterion(pred, output_label)


  return loss.item()


def trainIters(encoder, 
        decoder, 
        graph_attention,
        heuristic, 
        batch_number,
        epoch, 
        batch, 
        print_every=1000, 
        plot_every=100, 
        learning_rate=0.0005):
  loss_plot = []
  loss_total = 0  # Reset every print_every
  loss_count = 0
  train_loss = 0
  heuristic_optimizer = optim.Adam(heuristic.parameters(), lr=learning_rate)

#  training_sets = np.random.permutation(train_sets)
  
  criterion = nn.MSELoss()
  time = batch[0]
  path = batch[1]
  label = batch[2]

  if batch_number < 2500:
    loss = train(time,
            path,
            label,  
            encoder, 
            decoder, 
            graph_attention,
            heuristic,
            heuristic_optimizer, 
            criterion)    
    train_loss += loss        
  else:
    loss = test(time,
            path,
            label,
            encoder, 
            decoder,
            graph_attention,
            heuristic, 
            criterion) 
#    print("test loss:", loss)                           
    loss_total += loss

  return loss_total, train_loss# / loss_count

epoches = 100


train_sets = get_heuristic_data(dataset)
print("batches:", np.array(train_sets).shape)
road_loss = 0
count = 0

#在此处初始化模型  为所有的路训练一个模型
encoder, decoder, graph_attention = load_model()
long_temporal = TCN(1, 128, n_channels, kernel_size).to(device)
heuristic = HN(device).to(device)
for e in range(epoches): 
  batch_num = 0
  loss_total = 0 
  loss_count = 1
  train_loss_total = 0
  train_loss_count = 1
  for road in train_sets:
    loss, train_loss = trainIters(encoder, decoder, graph_attention, heuristic, batch_num, e, road)
    if loss > 0:
      loss_total += loss 
      loss_count += 1 
    else:
      train_loss_total += train_loss 
      train_loss_count += 1 
      
    batch_num += 1
    
  print("epoch:", e, "ave_test_loss:", loss_total / loss_count, "ave_test_loss:", train_loss_total / train_loss_count)
  torch.save(heuristic, "/data/wuning/traffic-data/model/" + dataset + "/" + model_name + "/" + str(e) + ".h")
 
  road_loss += loss
  count += 1
  print(count, road_loss / count)




   