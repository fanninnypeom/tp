from model.gru import *
from model.gru_directly import *
from utils import *
from torch import optim
import numpy as np
import time
import random
import sys
import pickle
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
input_length = 30
pred_length = 30
#print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "beijing"        # baidu  beijing  P7

model_name = "gru"         # gru  gru_gcn

step = 25



def train(input_tensor, output_tensor, time_tensor, encoder, encoder_optimizer, criterion, max_length=30):

  encoder_hidden = encoder.initHidden()
  encoder_optimizer.zero_grad()
  try:
    input_speed = torch.tensor(input_tensor, dtype=torch.float, device=device)
    input_time = torch.tensor(time_tensor, dtype=torch.long, device=device)
#    print(target_tensor.shape)
    target_tensor = torch.tensor(output_tensor, dtype=torch.float, device=device)
    
  except Exception as err:
    print(err)
#    print("--------")
    return 0, 0, 0
#  print(input_speed)  
  input_length = input_speed.size(1)
#  print("target_tensor:", target_tensor.size())
#  target_tensor = target_tensor[0]

  target_length = target_tensor.size(1)

  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_speed[:, ei], input_time[:, ei], encoder_hidden)
    loss += criterion(encoder_output[:, 0], target_tensor[:, ei])

#      if decoder_input.item() == EOS_token:
#        break

  loss.backward()

  encoder_optimizer.step()
#  print("raw_loss:", loss)
  return loss.item() / target_length, encoder_output[:, 0], target_tensor[:, target_length - 1]

def test(input_tensor, output_tensor, time_tensor, encoder, criterion, max_length=30):

  encoder_hidden = encoder.initHidden()
  try:
    input_speed = torch.tensor(input_tensor, dtype=torch.float, device=device)
    input_time = torch.tensor(time_tensor, dtype=torch.long, device=device)
#    print(target_tensor.shape)
    target_tensor = torch.tensor(output_tensor, dtype=torch.float, device=device)
    
  except Exception as err:
    print(err)
    return 0, 0, 0
  input_length = input_speed.size(1)

  target_length = target_tensor.size(1)

  encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

  loss = 0

  for ei in range(input_length):
    encoder_output, encoder_hidden = encoder(input_speed[:, ei], input_time[:, ei], encoder_hidden)
    loss += criterion(encoder_output[:, 0],target_tensor[:, ei])

  return loss.item() / target_length, encoder_output[:, 0], target_tensor[:, target_length - 1]


def trainIters(encoder, epoch, train_sets, print_every=1000, plot_every=100, learning_rate=0.001):
  start = time.time()
  loss_plot = []
  loss_total = 0  # Reset every print_every
  loss_count = 0 
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#  training_sets = np.random.permutation(train_sets)
  
  criterion = nn.MSELoss()
 # print(len(train_sets))
  for iter in range(1, len(train_sets) + 1):
#    if iter == 5:
#      sys.exit()
  
    training_pair = np.array(train_sets[iter - 1])
    input_tensor = training_pair[:, 0, :]
    time_tensor = training_pair[:, 1, :]
#    time_tensor = training_pair[2]
#    for item in target_tensor[0]:
#      print(len(item))  
#    print(np.array(input_tensor).shape)
#    print(np.array(time_tensor).shape)
    if iter < len(train_sets) * 0.7:
      loss, pred, real = train(input_tensor[:, :input_length], input_tensor[:, step: input_length + step],  time_tensor[:, : input_length ], encoder, encoder_optimizer, criterion)
    else:
      loss, pred, real = test(input_tensor[:, :input_length], input_tensor[:, step : input_length + step],  time_tensor[:, : input_length], encoder, criterion)
      loss_total += math.sqrt(loss)
      loss_count += 1
    if loss == 0:
      continue    
#    print("loss:", loss)
    loss_plot = []
    if iter >= len(train_sets) * 0.7:
#      print("epoch:", epoch, "batch", iter, ", loss:" ,loss, "map:", math.sqrt(loss))
      loss_plot.append(loss_total / 100)
#      print(loss_plot)
    else:
      pass  
#      print("epoch:", epoch, "batch", iter, ", train_loss:" ,loss, "map:", math.sqrt(loss))

  return loss_total / loss_count

epoches = 50


train_sets = get_train_data(dataset, model_name)
print(np.array(train_sets).shape)
road_loss = 0
count = 0
for road in train_sets:
  encoder = GRUdirectly(device).to(device)
  for e in range(epoches):
    loss = trainIters(encoder, e, road)
    print("epoch:", e, "ave_test_loss:", loss)
  print("loss:", loss)  
  road_loss += loss
  count += 1
  print(count, road_loss / count)




