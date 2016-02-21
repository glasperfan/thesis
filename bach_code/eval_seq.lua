-- eval_seq.lua

--[[
	Experiment 2A: Baseline model with a sequence of 4 separately trained classifiers.
	Classifiers: Roman numeral, inversion, alto, and tenor voices.
--]]

-- Other libraries
require 'nn'
require 'hdf5'

--- Neural network ---
function make_model(max_index, output_size)
	-- Embedding sequence
	local embedding = nn.Sequential()
	embedding:add(nn.LookupTable(max_index, embedding_size))
	embedding:add(nn.Sum(1))

	-- Feed forward sequence
	local model = nn.Sequential()
	model:add(embedding)
	
	model:add(nn.Linear(embedding_size, hidden_size))
	model:add(nn.Sigmoid())
	model:add(nn.Linear(hidden_size, output_size))
	model:add(nn.LogSoftMax())

	-- Criterion: negative log likelihood
	local criterion = nn.ClassNLLCriterion()

	return model, criterion
end

--- Train the Model ---
function train(Xtrain, ytrain, model, criterion)
	for epoch = 1, epochs do
		local nll_arr = torch.Tensor(Xtrain:size(1))
		model:zeroGradParameters()
		for j = 1, Xtrain:size(1) do
			-- Forward
			local out = model:forward(Xtrain[j])
			nll_arr[j] = criterion:forward(out, ytrain[j])

			-- Backward
			local deriv = criterion:backward(out, ytrain[j])
			model:backward(Xtrain[j], deriv)
			model:updateParameters(learning_rate)
			model:zeroGradParameters()
		end
		print("Epoch:", epoch, torch.mean(nll_arr))
	end
end


-- Evaluation numeral subtask --
function eval(Xtest, ytest, model, criterion)
	-- Collect numeral predictions
	local nll = 0
	local pred = torch.IntTensor(ytest:size(1))
	for j = 1, Xtest:size(1) do
		out = model:forward(Xtest[j])
		_ , argmax = torch.max(out, 1)
		pred[j] = argmax[1]
		nll = nll + criterion:forward(out, ytest[j])
	end
	-- Evaluation
	local correct = torch.sum(torch.eq(pred, ytest))
	local count = pred:size(1)
	-- Results --
	print(string.format("Average nll: %.3f", nll / Xtest:size(1)))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
end



-- Constants
embedding_size = 250
hidden_size = 250
epochs = 6
learning_rate = 0.01

-- Load data
local f = hdf5.open("data/chorales.hdf5")
local Xtrain = f:read('Xtrain'):all()
local ytrain = f:read('ytrain'):all()
local Xdev = f:read('Xdev'):all()
local ydev = f:read('ydev'):all()
local Xtest = f:read('Xtest'):all()
local ytest = f:read('ytest'):all()
f:close()

Xtrain = torch.cat(Xtrain, Xdev, 1)			-- incorporate dev data into training
ytrain = torch.cat(ytrain, ydev, 1)

Xtrain = Xtrain[{{}, {1, 10}}]				-- remove Oracle-only data
Xtest = Xtest[{{}, {1, 10}}]

local Xall = torch.cat(Xtrain, Xtest, 1)	-- gather all data
local yall = torch.cat(ytrain, ytest, 1)

--[[
	argmax(x,y,z) P(X=x,Y=y,Z=z | Q) = argmax(x,y,z) P(X=x | Y=y, Z=z, Q) * P(Y=y | Z=z, Q) * P(Z=z | Q)
	  								 ~ argmax(z) P(Z=z | Q), argmax(y) P(Y=y | Z=z, Q), argmax(x) P(X=x | Y=y, Z=z, Q)
--]]

-- Build models and train/test data --
max_idx = Xall:max()
num_output_types = yall:size(2)
models = {}
criterions = {}
train_data = {}
test_data = {}
max_indices = {}
for i = 1, num_output_types do
	local source = Xtrain												-- Create train/test input
	local source_test = Xtest
	if i > 1 then
		source = torch.cat(train_data[i - 1], ytrain[{ {} , i - 1 }], 2)
		source_test = torch.cat(test_data[i - 1], ytest[{ {}, i - 1 }], 2)
		function increase_index(x) return x + max_idx end
		source[{ {}, source:size(2) }]:apply(increase_index)
		source_test[{ {}, source_test:size(2) }]:apply(increase_index)
	end
	max_idx = torch.cat(source, source_test, 1):max()
	local target = ytrain[{ {}, i }]									-- ytrain
	local target_test = ytest[{ {}, i }]								-- ytest
	local model, criterion = make_model(max_idx, yall[{ {}, i}]:max())	-- create the model
	train(source, target, model, criterion)								-- train
	eval(source_test, target_test, model, criterion)					-- eval
	models[i] = model 													-- store data and model for full eval
	train_data[i] = source
	test_data[i] = source_test
	max_indices[i] = max_idx
end

-- Evaluate ---
preds = torch.IntTensor(ytest:size())
accuracy = torch.zeros(ytest:size(1))
for j = 1, Xtest:size(1) do								-- Iterate over the test datapoints
	probs = torch.ones(ytrain:size(1))					-- Create a record of the probability of each possible outcome
	for k = 1, ytrain:size(1) do						-- Iterate over all possible output combinations
		for m = 1, num_output_types do 					-- Iterate over the different classifiers
			local model = models[m]
			local testX = test_data[m][j]
			for n = 1, m - 1 do
				testX[Xtest:size(2) + n] = ytrain[k][n] + max_indices[n]
			end
			local testy = ytrain[k][m]
			local out = model:forward(testX)						-- Output a distribution P(X = x | Q ...) for all x
			probs[k] = probs[k] * out[testy]						-- we're only interested in a specific X = x case
		end
	end
	valmax, argmax = torch.max(probs, 1)							--- Get the argmax(x,y,z)
	preds[j] = ytrain[argmax[1]]
	accuracy[j] = torch.eq(ytest[j], preds[j]):float():mean()
	print(string.format("Current example accuracy: %.2f%%", accuracy[j] * 100))
	print(string.format("Running example accuracy: %.2f%%", accuracy:sum() / j * 100))
	for n = 1, ytest[j]:size(1) do print(ytest[j][n], preds[j][n]) end
end



