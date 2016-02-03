-- eval.lua

--[[
	Experiment 2: Baseline model with the inversion decision.
	The correct Roman numeral is provided as a feature.
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
		-- eval_inv(Xtrain, ytrain, model, criterion)
		-- eval_inv(Xtest, ytest, model, criterion)
	end
	eval_inv(Xtrain, ytrain, model, criterion)
	eval_inv(Xtest, ytest, model, criterion)
end



-- Evaluation numeral subtask --
function eval_inv(Xtest, ytest, model, criterion)
	-- Collect numeral predictions
	local nll_arr = torch.Tensor(Xtest:size(1))
	local pred = torch.IntTensor(ytest:size(1))
	for j = 1, Xtest:size(1) do
		out = model:forward(Xtest[j])
		_ , argmax = torch.max(out, 1)
		pred[j] = argmax[1]
		nll_arr[j] = criterion:forward(out, ytest[j])
	end
	-- Evaluation
	local correct = torch.sum(torch.eq(pred, ytest))
	local count = pred:size(1)
	local nt = torch.ne(ytest, 1) -- not root
	local correct_nt = torch.sum(torch.eq(pred[nt], ytest[nt]))
	local count_nt = ytest[nt]:size(1)
	local nd = torch.ne(ytest[nt], 2) -- not root or first inversion
	local correct_ntod = torch.sum(torch.eq(pred[nt][nd], ytest[nt][nd]))
	local count_ntod = ytest[nt][nd]:size(1)
	-- 	Print some output --
	-- for i = count - 100, count do print(pred[i], ytest[i]) end 
	-- Results --
	print(string.format("Average nll: %.3f", torch.mean(nll_arr)))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
	print(string.format("Percentage correct (not root): %.2f%%", correct_nt / count_nt * 100.0))
	print(string.format("Percentage correct (not root or first inversion): %.2f%%", correct_ntod / count_ntod * 100.0))
end



function cross_val()
	for i, sz in ipairs({100, 200, 250}) do
		for i, lr in ipairs({0.1, 0.01, 0.001}) do
			print(sz, lr)
			embedding_size = sz
			hidden_size = sz
			learning_rate = lr
			local model, criterion = make_model(maxidx, outsz)
			train(Xtrain, ytrain, Xdev, ydev, model, criterion)
		end
	end
end

function main() 
	-- Contants
	embedding_size = 250
	hidden_size = 250
	epochs = 20
	learning_rate = 0.001

	-- Load data (created in Initial Experiments.ipynb)
	local f = hdf5.open("data/choralesinv.hdf5")
	Xtrain = f:read('Xtrain'):all()
	ytrain = f:read('ytrain'):all()
	Xdev = f:read('Xdev'):all()
	ydev = f:read('ydev'):all()
	Xtest = f:read('Xtest'):all()
	ytest = f:read('ytest'):all()
	Xall = f:read('Xall'):all()
	yall = f:read('yall'):all()
	f:close()

	-- Aggregate training and test sets
	maxidx = torch.max(Xall)
	outsz = torch.max(yall)
	
	-- cross_val()

	-- Create global models and criterion
	model, criterion = make_model(maxidx, outsz)

	-- Train
	print("# Training inversion subtask")
	train(Xtrain, ytrain, Xtest, ytest, model, criterion)
end

main()