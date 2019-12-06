import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HN(nn.Module):
  def __init__(self, device, hidden_size = 128, embedding_size = 512, length_embedding_size = 128, output_size = 1, batch_size = 100):
    super(HN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.length_embedding_size = length_embedding_size
    self.length_embedding = nn.Embedding(500, length_embedding_size)
    self.embedding_size = embedding_size
    self.embedding = nn.Embedding(10000, embedding_size)   # 每天分为96个时间片 
    self.gru = nn.GRU(embedding_size + length_embedding_size, embedding_size, bidirectional=True)
    self.out = nn.Linear(embedding_size * 2, output_size)
    self.relu = nn.ReLU()
                      
  def forward(self, departure_time, input_tensor, input_length, road_length, hidden):
    input = self.embedding(input_tensor)#.view(1, -1, self.embedding_size)
    road_length_embedding = self.length_embedding(road_length) 

    input = input.permute(1, 0, 2) 
    road_length_embedding = road_length_embedding.permute(1, 0, 2)

    input = torch.cat((input, road_length_embedding), 2)
    output, hidden = self.gru(input, hidden) 
#    output = output.permute((1, 0, 2))
#    print("hidden, hidden:", output, hidden)    
#    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
    
#    summary = torch.sum(output, 0) / input_length.unsqueeze(1)
#    pred = self.relu(self.out(summary))

#----------------
#    pred = self.relu(self.out(output)).squeeze()
#    result = torch.zeros(output.shape[1], device=self.device)
#    count = 0
#    pred = pred.transpose(1, 0)
#    for route, le in zip(pred, input_length):
#     route = route[:le.int()]  
#     result[count] = torch.sum(route, 0)
#     count += 1
#----------------
    output = output.permute(1, 0, 2)
    result = torch.zeros(output.shape[0], output.shape[2], device=self.device)
    count = 0   
    for route, le in zip(output, input_length):
     route = route[:le.int()]   
     result[count] = torch.mean(route, 0)
     count += 1

    pred = self.relu(self.out(result)).squeeze()  #86 27 1
  
    return pred
    
  def initHidden(self, batch_size):
    return torch.zeros(2, batch_size, self.embedding_size, device=self.device)

