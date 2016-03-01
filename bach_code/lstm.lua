require 'nn'
require 'rnn'
require 'hdf5'

g = hdf5.open("data/chorales_rnn.hdf5")
Xtrain, ytrain, Xtest, ytest = {}, {}, {}, {}
num_train, num_dev, num_test = 260, 32, 33
for i = 1, num_train do
	Xtrain[i] = g:read(string.format('train/chorale%d_X', i-1)):all()[{ {}, {1, 10} }]:t()
	ytrain[i] = g:read(string.format('train/chorale%d_y', i-1)):all()
end
for i = 1, num_dev do
	Xtrain[i + num_train] = g:read(string.format('dev/chorale%d_X', i-1)):all()[{ {}, {1, 10} }]:t()
	ytrain[i + num_train] = g:read(string.format('dev/chorale%d_y', i-1)):all()
end
for i = 1, num_test do
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

-- Build LSTM --
lstm = nn.Sequential()
	:add(nn.LookupTable(nV, d))
	:add(nn.Sum(1))
	:add(nn.SplitTable(1))
	:add(nn.Sequencer(nn.FastLSTM(d,d)))
	:add(nn.Sequencer(nn.Dropout(0.5)))
	:add(nn.Sequencer(nn.FastLSTM(d,d)))
	-- :add(nn.Sequencer(nn.Dropout(0.5)))
	-- :add(nn.Sequencer(nn.FastLSTM(d,d)))
	:add(nn.Sequencer(nn.Linear(d, nY)))
	:add(nn.Sequencer(nn.LogSoftMax()))

lstm:remember('both')

print(lstm)


-- build criterion
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

-- TRAIN --
lstm:reset()
lstm:training()
last_score = 9999
for epoch = 1, 100 do
	nll_epoch = 0
	for i = 1, num_train + num_dev do
		nll = 0
		lstm:zeroGradParameters()
		X = Xtrain[i]
		y = ytrain[i]
		out = lstm:forward(X)
		nll = nll + criterion:forward(out, y)
		deriv = criterion:backward(out, y)
		lstm:backward(X, deriv)

		-- Update the parameters
		lstm:updateParameters(lr)
		nll = nll / y:size(1)
		nll_epoch = nll_epoch + nll
	end
	nll_epoch = nll_epoch / (num_train + num_dev)
	print("Epoch: ", epoch, nll_epoch)
	if nll_epoch > last_score then break end
	last_score = nll_epoch
end

-- TEST --
function eval(Xt, yt)
	lstm:evaluate()
	accuracies = {}
	for i = 1, #Xt do
		X = Xt[i]
		y = yt[i]
		out = lstm:forward(X)
		pred = {}
		seq_end = 0
		for j = 1, y:size(1) do
			if y[j] == nY then 
				seq_end = j - 1
				break
			end
			_, argmax = torch.max(out[j], 1)
			pred[j] = argmax[1]
		end
		if seq_end == 0 then seq_end = y:size(1) end
		accuracies[i] = torch.mean(torch.eq(torch.IntTensor(pred), y:narrow(1,1,seq_end)):double())
		print(string.format("Chorale accuracy:\t%d\t%.2f%%", i, accuracies[i] * 100))
	end
	print(string.format("OVERALL ACCURACY: \t%.3f%%", torch.mean(torch.Tensor(accuracies)) * 100))
end

eval(Xtest, ytest)
eval(Xtrain, ytrain)