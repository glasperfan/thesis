
require 'nn'
require 'rnn'
require 'hdf5'

f = hdf5.open("data/chorales_nn.hdf5")
X = f:read('chorale1X'):all()
y = f:read('chorale1y'):all()
metadata = f:read('metadata'):all()

-- hyper-parameters 
rho = 32 -- sequence length
hiddenSize = 10
nIndex = metadata[1][1] -- vocabulary length
nIndexOut = metadata[1][2]
lr = 0.01


-- build simple recurrent neural network
local r = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

local rnn = nn.Sequential()
   :add(r)
   :add(nn.Linear(hiddenSize, nIndexOut))
   :add(nn.LogSoftMax())

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
rnn = nn.Sequencer(rnn)
print(rnn:forward(nn.SplitTable(3):forward(X)))
print(rnn)

-- build criterion

criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

