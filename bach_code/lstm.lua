require 'nn'
require 'rnn'
require 'hdf5'

g = hdf5.open("data/chorales_rnn.hdf5")
Xtrain, ytrain, Xtest, ytest = {}, {}, {}, {}
for i = 1, 260 do
	Xtrain[i] = g:read(string.format('train/chorale%d_X', i-1)):all()[{ {}, {1, 10} }]:t()
	ytrain[i] = g:read(string.format('train/chorale%d_y', i-1)):all()
end
for i = 1, 32 do
	Xtrain[i + 260] = g:read(string.format('dev/chorale%d_X', i-1)):all()[{ {}, {1, 10} }]:t()
	ytrain[i + 260] = g:read(string.format('dev/chorale%d_y', i-1)):all()
end
for i = 1, 33 do
	Xtest[i] = g:read(string.format('test/chorale%d_X', i-1)):all()[{ {}, {1, 10} }]:t()
	ytest[i] = g:read(string.format('test/chorale%d_y', i-1)):all()
end
metadata = g:read("metadata"):all()
g:close()

-- hyper-parameters 
d = 200
nY = metadata[1]
nV = metadata[2]
rho = metadata[3]
wsz = 4 -- window size
lr = 0.01

-- build simple recurrent neural network
lstm = nn.Sequential()
	:add(nn.LookupTable(nV, d))
	:add(nn.Sum(1))
	:add(nn.SplitTable(1))
	:add(nn.Sequencer(nn.FastLSTM(d,d)))
	-- :add(nn.Sequencer(nn.Dropout(0.5)))
	-- :add(nn.Sequencer(nn.FastLSTM(d,d)))

lstm:remember('both')

model = nn.Sequential()
	:add(nn.Linear(d, nY))
	:add(nn.LogSoftMax())

seq_model = nn.Sequencer(model)

-- print(rnn)
-- print(seq_model)


-- build criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- TRAIN --
lstm:reset()
seq_model:reset()
lstm:training()
last_score = 9999
for epoch = 1, 10 do
	nll_epoch = 0
	for i = 1, 260 do
		nll = 0
		lstm:zeroGradParameters()
		seq_model:zeroGradParameters()
		X = Xtrain[i]
		y = ytrain[i]
		h = lstm:forward(X)
		out = seq_model:forward(h)
		-- print(out)
		nll = nll + criterion:forward(out, y)
		deriv = criterion:backward(out, y)
		d2 = seq_model:backward(h, deriv)
		lstm:backward(X, d2)

		-- Update the parameters
		lstm:updateParameters(lr)
		model:updateParameters(lr)
		nll = nll / y:size(1)
		-- print("NLL for chorale: ", i, nll, y:size(1))
		nll_epoch = nll_epoch + nll
	end
	nll_epoch = nll_epoch / 260
	print("Epoch: ", epoch, nll_epoch)
	if nll_epoch > last_score then break end
	last_score = nll_epoch
end

-- TEST --
output_model = nn.Sequencer(model)
lstm:evaluate()
accuracies = {}
for i = 1, 33 do
	-- r_layer:forget()
	X = Xtest[i]
	y = ytest[i]
	h = lstm:forward(X)
	out = output_model(h)
	pred = {}
	seq_end = 0
	for j = 1, y:size(1) do
		if y[j] == nY then 
			seq_end = j - 1
			break
		end
		_, argmax = torch.max(out[j], 1)
		-- print(argmax)
		pred[j] = argmax[1]
		-- print(pred[j], y[j])
	end
	accuracies[i] = torch.mean(torch.eq(torch.IntTensor(pred), y:narrow(1,1,seq_end)):double())
	print(string.format("Chorale accuracy:\t%d\t%.2f%%", i, accuracies[i] * 100))
end
print(string.format("OVERALL ACCURACY: \t%.3f%%", torch.mean(torch.Tensor(accuracies)) * 100))
