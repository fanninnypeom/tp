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

input_length = 30
pred_length = 30
#print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "beijing"        # baidu  beijing  P7 

model_name = "gru_gcn"         # gru  gru_gcn gru_gcn_tcn

model_type = "gru_gcn"

n_channels = [50] * 6

kernel_size = 5

tcn_dropout = 0.25


steps = 25

teacher_forcing_ratio = 0.0
test_teacher_forcing_ratio = 0


def train(input_tensor,
          output_tensor, 
          input_time_tensor, 
          output_time_tensor, 
          neighs_input_tensor, 
          neighs_time_tensor, 
          encoder, 
          decoder, 
          graph_attention, 
          encoder_optimizer, 
          decoder_optimizer, 
          criterion, 
          max_length=30):

  encoder_hidden = encoder.initHidden()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
#  print(np.array(input_tensor).shape)
#  print(np.array(output_tensor).shape)
#  print(np.array(time_tensor).shape)
  try:
    input_speed = torch.tensor(input_tensor, dtype=torch.float, device=device)
    input_time = torch.tensor(input_time_tensor, dtype=torch.long, device=device)
    output_time = torch.tensor(output_time_tensor, dtype=torch.long, device=device)
    target_tensor = torch.tensor(output_tensor, dtype=torch.float, device=device)
    neighs_speed_tensor = torch.tensor(neighs_input_tensor, dtype=torch.float, device=device)
    neighs_time_tensor = torch.tensor(neighs_time_tensor, dtype=torch.long, device=device)
  except Exception as err:
    print(err)
#    print("--------")
    return 0, 0, 0
#  print(input_speed)  
  input_length = input_speed.size(1)
#  print("target_tensor:", target_tensor.size())
#  target_tensor = target_tensor[0]

  target_length = max_length#target_tensor.size(1)

  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

  loss = 0

  for ei in range(input_length):
#    print("main:", input_speed[:, ei])  
    encoder_output, encoder_hidden = encoder(input_speed[:, ei], input_time[:, ei], encoder_hidden)

  if model_type == "gru_gcn":
    main_hidden = encoder_hidden  
    neighs_hidden = []
    for neigh_speed_tensor, neigh_time_tensor in zip(neighs_speed_tensor, neighs_time_tensor):      
      encoder_hidden = encoder.initHidden()
#      print("neighs:" , neigh_speed_tensor[:, 0])
      for ei in range(input_length):        
        encoder_output, encoder_hidden = encoder(neigh_speed_tensor[:, ei], neigh_time_tensor[:, ei], encoder_hidden)
      neighs_hidden.append(encoder_hidden.squeeze())   
    encoder_hidden = graph_attention(main_hidden, neighs_hidden)
#    encoder_hidden = graph_attention.relu(graph_attention.fuse(encoder_hidden))
  if model_type == "gru_gcn_tcn":
    main_hidden = encoder_hidden  
    neighs_hidden = []
    for neigh_speed_tensor, neigh_time_tensor in zip(neighs_speed_tensor, neighs_time_tensor):      
      encoder_hidden = encoder.initHidden()
#      print("neighs:" , neigh_speed_tensor[:, 0])
      for ei in range(input_length):        
        encoder_output, encoder_hidden = encoder(neigh_speed_tensor[:, ei], neigh_time_tensor[:, ei], encoder_hidden)
      neighs_hidden.append(encoder_hidden.squeeze())   
    encoder_hidden = graph_attention(main_hidden, neighs_hidden)
    long_temporal_hideen = TCN(1, 128, n_channels, kernel_size, dropout=tcn_dropout)
    encoder_hidden = encoder_hidden + long_temporal_hideen 

  decoder_input = torch.tensor([[2.0 for i in range(100)]], device=device) # start id 被设置成了 180.0

  decoder_hidden = encoder_hidden


  for di in range(target_length):
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    decoder_output, decoder_hidden = decoder(decoder_input, output_time[:, di], decoder_hidden)
    loss += criterion(decoder_output[:, 0], target_tensor[:, di])
    if use_teacher_forcing:
# Teacher forcing: Feed the target as the next input
      decoder_input = target_tensor[:, di]  # Teacher forcing
    else:
      decoder_input = decoder_output

  loss.backward()

  encoder_optimizer.step()
  decoder_optimizer.step()
#  print("raw_loss:", loss)
  return loss.item() / target_length, decoder_output[:, 0], target_tensor[:, target_length - 1]

def test(input_tensor, 
         output_tensor, 
         input_time_tensor, 
         output_time_tensor, 
         neighs_input_tensor, 
         neighs_time_tensor, 
         encoder, 
         decoder, 
         graph_attention, 
         criterion, 
         max_length=30):

  encoder_hidden = encoder.initHidden()
  try:
    input_speed = torch.tensor(input_tensor, dtype=torch.float, device=device)
    input_time = torch.tensor(input_time_tensor, dtype=torch.long, device=device)
    output_time = torch.tensor(output_time_tensor, dtype=torch.long, device=device)
    target_tensor = torch.tensor(output_tensor, dtype=torch.float, device=device)
    neighs_speed_tensor = torch.tensor(neighs_input_tensor, dtype=torch.float, device=device)
    neighs_time_tensor = torch.tensor(neighs_time_tensor, dtype=torch.long, device=device)
  except Exception as err:
    print(err)
    return 0, 0, 0
  input_length = input_speed.size(1)

  target_length = max_length#target_tensor.size(1)

  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_speed[:, ei], input_time[:, ei], encoder_hidden)

  if model_type == "gru_gcn":
    main_hidden = encoder_hidden  
    neighs_hidden = []
    for neigh_speed_tensor, neigh_time_tensor in zip(neighs_speed_tensor, neighs_time_tensor):      
      encoder_hidden = encoder.initHidden()
      for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(neigh_speed_tensor[:, ei], neigh_time_tensor[:, ei], encoder_hidden)
      neighs_hidden.append(encoder_hidden.squeeze())   
    encoder_hidden = graph_attention(main_hidden, neighs_hidden)
#    encoder_hidden = graph_attention.relu(graph_attention.fuse(encoder_hidden))

  decoder_input = torch.tensor([[2.0 for i in range(100)]], device=device) # start id 被设置成了 180.0

  decoder_hidden = encoder_hidden

  pred = []
  real = []

  use_teacher_forcing = True if random.random() < test_teacher_forcing_ratio else False
  if use_teacher_forcing:
# Teacher forcing: Feed the target as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden = decoder(decoder_input, output_time[:, di], decoder_hidden)
      if di == steps - 1:
        loss += criterion(decoder_output[:, 0], target_tensor[:, di])
      pred = decoder_output[:, 0]
      real = target_tensor[:, di]
      decoder_input = target_tensor[:, di]  # Teacher forcing
  else:
# Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
      decoder_output, decoder_hidden = decoder(decoder_input, output_time[:, di], decoder_hidden)
#      topv, topi = decoder_output.topk(1)
#      decoder_input = topi.squeeze().detach()  # detach from history as input
      decoder_input = decoder_output
      if di == steps - 1:
        loss += criterion(decoder_output[:, 0], target_tensor[:, di])
#      if decoder_input.item() == EOS_token:
#        break

  return loss.item(), decoder_output[:, 0], target_tensor[:, target_length - 1]


def trainIters(encoder, 
               decoder, 
               graph_attention, 
               epoch, 
               train_sets, 
               print_every=1000, 
               plot_every=100, 
               learning_rate=0.001):
  loss_plot = []
  loss_total = 0  # Reset every print_every
  loss_count = 0
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
#  training_sets = np.random.permutation(train_sets)
  
  criterion = nn.MSELoss()
 # print(len(train_sets))
  for iter in range(1, len(train_sets[0]) + 1):
#    if iter == 5:
#      sys.exit()
    training_pair = np.array(train_sets[0][iter - 1])
#    print(len(train_sets), len(train_sets[1]), len(train_sets[1][0]), iter)
#    try:
    neighs = np.array(train_sets[1])[:, iter-1] # N * batch_size * 2 * length
#    except Exception as err:
#      print(np.array(train_sets[1]).shape)    
    neighs_input_tensor = []
    neighs_time_tensor = []
    for neigh in neighs:
      neighs_input_tensor.append(neigh[:, 0, :])
      neighs_time_tensor.append(neigh[:, 1, :])
    neighs_input_tensor = np.array(neighs_input_tensor)  
    neighs_time_tensor = np.array(neighs_time_tensor)

    input_tensor = training_pair[:, 0, :]  #batch_size * length
    time_tensor = training_pair[:, 1, :]   #batch_size * length
#    time_tensor = training_pair[2]
#    for item in target_tensor[0]:
#      print(len(item))  
#    print(np.array(input_tensor).shape)
#    print(np.array(time_tensor).shape)
    if iter < len(train_sets) * 0.7:
      loss, pred, real = train(input_tensor[:, :input_length], 
                               input_tensor[:, input_length : input_length + pred_length], 
                               time_tensor[:, :input_length], 
                               time_tensor[:, input_length : input_length + pred_length],
                               neighs_input_tensor[:, :, :input_length],
                               neighs_time_tensor[:, :, :input_length],
                               encoder, 
                               decoder, 
                               graph_attention,
                               encoder_optimizer, 
                               decoder_optimizer, 
                               criterion)
    else:
      loss, pred, real = test(input_tensor[:, :input_length], 
                              input_tensor[:, input_length : input_length + pred_length], 
                              time_tensor[:, :input_length], 
                              time_tensor[:, input_length : input_length + pred_length], 
                              neighs_input_tensor[:, :, :input_length],
                              neighs_time_tensor[:, :, :input_length],
                              encoder, 
                              decoder,
                              graph_attention, 
                              criterion)
      loss_total += math.sqrt(loss)
      loss_count += 1
    if loss == 0:
      continue    
    loss_plot = []
    if iter >= len(train_sets) * 0.7:
#      print("epoch:", epoch, "batch", iter, ", loss:" ,loss, "map:", math.sqrt(loss))
      loss_plot.append(loss_total / 100)
#      print(loss_plot)
    else:
      pass  
#      print("epoch:", epoch, "batch", iter, ", train_loss:" ,loss, "map:", math.sqrt(loss))

  return loss_total / loss_count

epoches = 100


train_sets = get_train_data(dataset, model_name)
print(np.array(train_sets).shape)
road_loss = 0
count = 0

#在此处初始化模型  为所有的路训练一个模型
encoder = EncoderGRU(device).to(device)
decoder = DecoderGRU(device).to(device)   
graph_attention = GAT(device).to(device)

for road in train_sets:
  for e in range(epoches):
    loss = trainIters(encoder, decoder, graph_attention, e, road)
    print("epoch:", e, "ave_test_loss:", loss)
  torch.save(encoder, "/data/wuning/traffic-data/model/" + dataset + "/" + model_name + "/" + str(e) + ".e")   
  torch.save(decoder, "/data/wuning/traffic-data/model/" + dataset + "/" + model_name + "/" + str(e) + ".d")   
  torch.save(graph_attention, "/data/wuning/traffic-data/model/" + dataset + "/" + model_name + "/" + str(e) + ".g")   

  road_loss += loss
  count += 1
  print(count, road_loss / count)




   