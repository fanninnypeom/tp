import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GAT(nn.Module):
  def __init__(self, device, hidden_size = 128, embedding_size = 64, output_size = 1, batch_size = 100):
    super(GAT, self).__init__()
    #      self.hidden_size = hidden_size
    self.device = device
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.embedding = nn.Embedding(196, embedding_size)   # 每天分为96个时间片  
    self.gru = nn.GRU(1, hidden_size) #65 = 64 + 1
    self.out = nn.Linear(hidden_size + embedding_size, output_size)
    self.attn = nn.Linear(2 * hidden_size, 1)
    self.fuse = nn.Linear(2 * hidden_size, hidden_size)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, hidden, neighs):
    # neighs  n * batch_size * dims    hidden 1* batch_size * dims
    N = len(neighs)   
    neighs = torch.stack(neighs)
    output_neigh_cat = torch.cat((hidden.repeat(N, 1, 1), neighs), 2)
    attn_weight = F.softmax(self.attn(output_neigh_cat), 0)  # n * batch_size * 1
#    print(attn_weight.shape, attn_weight.transpose(1, 0)[0, :, 0])
    output = torch.bmm(attn_weight.permute(1, 2, 0), neighs.permute(1, 0, 2)).squeeze()

#    output = torch.mean(neighs, 0)
#    return output.unsqueeze(0)

#    output = torch.cat((hidden, output.unsqueeze(0)), 2)
    output = (hidden + output.unsqueeze(0)) 
    return output


  def initHidden(self):
    return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

class EncoderGRU(nn.Module):
  def __init__(self, device, hidden_size = 128, embedding_size = 64, output_size = 1, batch_size = 100):
    super(EncoderGRU, self).__init__()
    self.device = device
#    self.gru_size = gru_size              # input_speed: batch_size * length  float
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.embedding = nn.Embedding(196, embedding_size)
    self.gru = nn.GRU(1, hidden_size)
    self.out = nn.Linear(hidden_size + embedding_size, output_size)
    self.attn = nn.Linear(2 * hidden_size, 1)

#    self.real_speed = label                     # time: batch_size * length  int
#    self.steps = steps                          #  预测的步数  也就是 decoder的长度
  def forward(self, input_speed, time, hidden):
    time_emb = self.embedding(time).view(1, -1, self.embedding_size)

#    time_emb = self.embedding(input_speed).view(1, 1, -1)
#    output = input_speed 
#    print("input_speed:", input_speed)
#    print("hidden:", hidden)
    input_speed = input_speed.view(1, -1, 1)
    output, hidden = self.gru(input_speed, hidden)
    output = torch.cat((time_emb, output), 2)
#    print("hidden, hidden:", output, hidden)
    output = self.out(output[0])
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)

class DecoderGRU(nn.Module):   
    def __init__(self, device, hidden_size = 128, embedding_size = 64, output_size = 1, batch_size = 100):
      super(DecoderGRU, self).__init__()
#      self.hidden_size = hidden_size
      self.device = device
      self.hidden_size = hidden_size
      self.batch_size = batch_size
      self.embedding_size = embedding_size
      self.embedding = nn.Embedding(196, embedding_size)   # 每天分为96个时间片  
      self.gru = nn.GRU(1, hidden_size) #65 = 64 + 1
      self.out = nn.Linear(hidden_size + embedding_size, output_size)
      self.sigmoid = nn.ReLU()

    def forward(self, input_speed, time, hidden):
    # neighs  n * batch_size * dims    hidden batch_size * dims
      time_emb = self.embedding(time).view(1, -1, self.embedding_size)
      input_speed = input_speed.view(1, -1, 1)
      output, hidden = self.gru(input_speed, hidden)
      output = torch.cat((time_emb, output), 2)
      output = self.out(output[0])
      output = self.sigmoid(output)
      return output, hidden

    def initHidden(self):
      return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


