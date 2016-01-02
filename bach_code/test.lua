require 'nn'
require 'rnn'
require 'load'

dat = data.new('data/')
X = dat.train[1].X
y = dat.train[1].y

d = 10
n = X:size(2) -- should be 10
nX = torch.max(X)
nY = torch.max(y)

lstm = nn.Sequential()
lstm:add(nn.Sequencer(nn.LookupTable(nX, d)))
lstm:add(nn.Sequencer(nn.LSTM(d, d)))

model = nn.Sequential()
model:add(nn.Linear(d, nY))
model:add(nn.LogSoftMax())