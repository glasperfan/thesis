-- eval_sum.lua

--[[
	Experiment 2C: Softmax
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
	model:add(nn.Tanh())
	model:add(nn.Linear(hidden_size, output_size))
	model:add(nn.LogSoftMax())

	-- Criterion: negative log likelihood
	local criterion = nn.ClassNLLCriterion()

	return model, criterion
end

-- Multiclass logistic regression --
function multiclass_logistic_regression(max_index, output_size)
	-- Embedding sequence
	local embedding = nn.Sequential()
	embedding:add(nn.LookupTable(max_index, output_size))
	embedding:add(nn.Sum(1))

	-- Feed forward sequence
	local model = nn.Sequential()
	model:add(embedding)
	model:add(nn.LogSoftMax())

	-- Criterion: negative log likelihood
	local criterion = nn.ClassNLLCriterion()

	return model, criterion
end


--- Train the Model ---
function train(Xtrain, ytrain, Xtest, ytest, model, criterion)
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
		-- print(torch.mean(nll_arr))
		print("Epoch:", epoch, torch.mean(nll_arr))
		eval(Xtrain, ytrain, model, criterion)
		eval(Xtest, ytest, model, criterion)
	end
end


-- Evaluation numeral subtask --
function eval(Xtest, ytest, model, criterion)
	-- Collect numeral predictions
	local nll_arr = torch.Tensor(Xtest:size(1))
	local pred = torch.IntTensor(ytest:size(1))
	for j = 1, Xtest:size(1) do
		out = model:forward(Xtest[j])
		expout = torch.Tensor(out:size(1)):copy(out):exp()
		-- print(out[ytest[j]], expout[ytest[j]])
		_ , argmax = torch.max(out, 1)
		pred[j] = argmax[1]
		nll_arr[j] = criterion:forward(out, ytest[j])
	end
	-- Evaluation
	local correct = torch.sum(torch.eq(pred, ytest))
	local count = pred:size(1)
	-- 	Print some output --
	-- for i = count - 100, count do print(pred[i], ytest[i]) end 
	-- Results --
	print(string.format("Average nll: %.3f", torch.mean(nll_arr)))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
end

function main() 
	-- Contants
	embedding_size = 250
	hidden_size = 250
	epochs = 20
	learning_rate = 0.01

	-- Load data
	local f = hdf5.open("data/chorales.hdf5")
	local Xtrain = f:read('Xtrain'):all()
	local Xdev = f:read('Xdev'):all()
	local Xtest = f:read('Xtest'):all()
	f:close()

	Xtrain = Xtrain[{ {}, {1,10} }]
	Xdev = Xdev[{ {}, {1,10} }]
	Xtest = Xtest[{ {}, {1,10} }]
	Xtrain = torch.cat(Xtrain, Xdev, 1)
	local Xall = torch.cat(Xtrain, Xtest, 1)

	local f = hdf5.open("data/chorales_sm.hdf5")
	local ytrain = f:read('ytrainfeat'):all() -- this contains both the training and dev sets
	local ytest = f:read('ytestfeat'):all()
	local yall = f:read('yallfeat'):all()
	local max_index = torch.max(Xall)
	local output_size = torch.max(yall)

	-- Create global models and criterion
	model, criterion = make_model(max_index, output_size)

	-- Train
	train(Xtrain, ytrain, Xtestex, ytestex, model, criterion)
end

main()