import torch
import numpy as np
import math
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class embedding(torch.nn.Module):
  def __init__(self,vocab_size, d_embedding,window_length):
    super().__init__()
    self.embedding_table = torch.nn.Embedding(vocab_size,d_embedding)
    self.window_length = window_length
    def positional_encoding(window_length, d_embedding):
      """
      Returns a (d_embedding) tensor with values to be added to input embeddings as positional encoding
      """
      t = torch.zeros((window_length, d_embedding))
      for position_index in range(window_length):
        for dimension_index in range(d_embedding):
          t[position_index,dimension_index] = math.sin(position_index/(10000**(2*dimension_index/d_embedding)))
      return t
    self.encoding_tensor = positional_encoding(window_length,d_embedding).to(device)
  def forward(self,x):
    x = self.embedding_table(x)
    #print('enc: ',self.encoding_tensor.shape)
    for i in range(x.shape[0]):
      x[i] = self.encoding_tensor+x[i]
    return x
    return torch.nn.functional.normalize(x,dim = -1)
  
class unembedding(torch.nn.Module):
  def __init__(self,vocab_size,d_embedding):
    super().__init__()
    self.vocab_size = vocab_size
    self.unembedding = torch.nn.Linear(d_embedding,vocab_size)
  def forward(self,x):
    logits = (self.unembedding(x))
    return (logits)

class ff_block(torch.nn.Module):
  def __init__(self,dim_in, n_layers):
    super().__init__()
    self.dim_in = dim_in
    self.n_layers = n_layers
    self.layers = []
    self.layers.append(torch.nn.Linear(dim_in,4*dim_in))
    self.layers.append(torch.nn.LeakyReLU())
    self.layers.append(torch.nn.Dropout(p=0.2))
    for i in range(n_layers):
      self.layers.append(torch.nn.Linear(4*dim_in,4*dim_in))
      self.layers.append(torch.nn.LeakyReLU())
      self.layers.append(torch.nn.Dropout(p=0.2))
    self.layers.append(torch.nn.Linear(4*dim_in,dim_in))
    self.layers[-1].weight.data*=0
    self.ff = torch.nn.Sequential(*self.layers)
    
    self.lnorm = (torch.nn.LayerNorm([dim_in]))
  def forward(self,X):
    X = (X+self.ff(X))
    #return X
    return self.lnorm(X)

class self_attention_head(torch.nn.Module):
  """
  d_emb = dimension of token embedding
  d_h = dimension of key
  """
  def __init__(self, d_emb, d_h,window_length):
    super().__init__()
    self.d_emb = d_emb
    self.d_h=d_h
    self.Wq = torch.nn.Linear(d_emb,d_h)
    self.Wk = torch.nn.Linear(d_emb,d_h)
    (self.Wk.weight.data)*=0
    self.Wv = torch.nn.Linear(d_emb,d_h)
    self.M = torch.ones((window_length,window_length))
    self.M = torch.tril(self.M).to(device)
    #print(self.M)
  def forward(self,x):
    """
    Accepts a B,T,C tensor and outputs the same tensor with the result of an attentional calculation added 
    """
    Q = self.Wq(x)
    K = self.Wk(x)
    V = self.Wv(x)
    alpha = Q@(torch.transpose(K,-2,-1))
    alpha = alpha.masked_fill_(self.M==0,float('-inf'))
    alpha = torch.nn.functional.softmax(alpha, dim=-1)
    alpha = alpha*(self.d_h**0.5)
    return (alpha@V)

class multihead_attention(torch.nn.Module):
  def __init__(self,d_emb,d_h,window_length,n_heads):
    super().__init__()
    self.heads = torch.nn.ModuleList([self_attention_head(d_emb,d_h,window_length) for i in range(n_heads)])
    self.weights = torch.nn.Linear(d_h*n_heads,d_emb)
    self.lnorm = torch.nn.LayerNorm([d_emb])
  def forward(self,x):
    all_heads = torch.cat([head(x) for head in self.heads],dim=-1)
    return self.lnorm(self.weights(all_heads)+x)

  
class Transformer(torch.nn.Module):
  def __init__(self,num_blocks, vocab_size,d_embedding,d_head=16,window_length=8,n_heads = 4):
    super().__init__()
    self.num_blocks = num_blocks
    self.n_heads = n_heads
    self.window_length = window_length
    self.vocab_size = vocab_size
    self.d_embedding = d_embedding
    self.d_head = d_head
    self.loss = torch.nn.CrossEntropyLoss()
    self.layers = [
      embedding(vocab_size,d_embedding,window_length)
    ]
    for i in range(num_blocks):
      self.layers.append(multihead_attention(d_embedding,d_head,window_length,n_heads))
      self.layers.append(ff_block(d_embedding,0))
    self.layers.append(unembedding(vocab_size,d_embedding))
    self.go = torch.nn.Sequential(*self.layers)
  def forward(self,x,y):
    B,T = x.shape
    logits = self.go(x)
    logits = logits.view(B*T,self.vocab_size)
    y = y.view(B*T)
    loss = self.loss(logits,y)
    return logits, loss
  def generate(self,s,length):
    pass
     
