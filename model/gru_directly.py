import torch
import torch.nn as nn
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GRUdirectly(nn.Module):
  def __init__(self, device, hidden_size = 128, embedding_size = 64, output_size = 1, batch_size = 100):
    super(GRUdirectly, self).__init__()
    self.device = device
#    self.gru_size = gru_size              # input_speed: batch_size * length  float
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.embedding = nn.Embedding(196, embedding_size)
    self.gru = nn.GRU(1, hidden_size)
    self.out = nn.Linear(hidden_size + embedding_size, output_size)
    self.sigmoid = nn.Sigmoid()
#    self.real_speed = label                     # time: batch_size * length  int
#    self.steps = steps                          #  预测的步数  也就是 decoder的长度
  def forward(self, input_speed, time, hidden):
    time_emb = self.embedding(time).view(1, -1, self.embedding_size)
   
#    output = input_speed 
#    print("input_speed:", input_speed)
#    print("hidden:", hidden)
    input_speed = input_speed.view(1, -1, 1)
    output, hidden = self.gru(input_speed, hidden)
    output = torch.cat((time_emb, output), 2)
#    print("hidden, hidden:", output, hidden)
    output = self.out(output[0])
    output = self.sigmoid(output)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)



