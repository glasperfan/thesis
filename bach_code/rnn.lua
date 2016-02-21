
require 'nn'
require 'rnn'
require 'hdf5'

f = hdf5.open("data/chorales.hdf5")
g = hdf5.open("data/chorales_rnn.hdf5")
X = f:read('chorale0_X'):all()
y = g:read('chorale0_y'):all()
yall = g:read('yall'):all()
Xtrain = f:read('Xtrain'):all()
X = X[{ {}, {1, 10} }]:t()
Xtrain = Xtrain[{ {}, {1, 10} }]

-- hyper-parameters 
rho = 84 -- sequence length
hiddenSize = 200
nIndex = Xtrain:max() -- vocabulary length
nIndexOut = yall:max()
lr = 0.01


-- build simple recurrent neural network
local r_layer = nn.Recurrent(
   hiddenSize, nn.LookupTable(nIndex, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(), 
   rho
)

local model = nn.Sequential()
	:add(nn.Linear(hiddenSize, nIndexOut))
	:add(nn.LogSoftMax())

seq_model = nn.Sequential()
	:add(nn.Sequencer(r_layer))

-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
-- print(seq_model:forward(nn.SplitTable(2):forward(X)))

-- build criterion
criterion = nn.ClassNLLCriterion()

-- TRAIN --
model:reset()
for epoch = 1, 20 do
	nll = 0
	r_layer:forget()
	model:zeroGradParameters()
	input = nn.SplitTable(2):forward(X)
	h = seq_model:forward(input)
	out = model:forward(h[rho])
	nll = nll + criterion:forward(out, y)
	deriv = criterion:backward(out, y)
	d2 = model:backward(h[rho], deriv)


	-- Only the last layer gets a gradient from the output
	gradient = {}
	for t = 1, rho - 1 do
		gradient[t] = torch.zeros(X:size(1), hiddenSize)
	end
	gradient[rho] = d2
	seq_model:backward(input, gradient)

	-- Update the parameters
	model:updateParameters(lr)
	r_layer:updateParameters(lr)
	print("Epoch: ", epoch, nll / X:size(1))
end
