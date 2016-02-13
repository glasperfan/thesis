-- eval.lua

--[[
	Experiment 1A: Baseline model with two-step harmonic decision process.
	The selection of harmony is divided into two steps: the choice of numeral followed by the choice of inversion.
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

function isnan(x) return x ~= x end

--- Train the Model ---
function train(Xtrain, ytrain, Xtest, ytest, model, criterion)
	for epoch = 1, epochs do
		local nll = torch.zeros(Xtrain:size(1))
		model:zeroGradParameters()
		for j = 1, Xtrain:size(1) do
			-- Forward
			local out = model:forward(Xtrain[j])
			nll[j] = criterion:forward(out, ytrain[j])
			-- Backward
			local deriv = criterion:backward(out, ytrain[j])
			model:backward(Xtrain[j], deriv)
			model:updateParameters(learning_rate)
			model:zeroGradParameters()
		end
		-- print(torch.mean(nll_arr))
		print("Epoch:", epoch, torch.mean(nll))
		eval_num(Xtrain, ytrain, model, criterion)
		eval_num(Xtest, ytest, model, criterion)
	end
end



-- Evaluation numeral subtask --
function eval_num(Xtest, ytest, model, criterion)
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
	-- Print some output --
	-- for i = count - 100, count do print(pred[i], ytest[i]) end
	-- Results --
	print(string.format("Average nll: %.3f", nll / Xtest:size(1)))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
end



function main() 
	-- Contants
	embedding_size = 300
	hidden_size = 300
	epochs = 20
	learning_rate = 0.005

	-- Create the data loader class.
	local f = hdf5.open("data/chorales.hdf5")
	local Xtrain = f:read('Xtrain'):all()
	local ytrain = f:read('ytrain'):all()
	local Xtest = f:read('Xtest'):all()
	local ytest = f:read('ytest'):all()
	f:close()
	
	-- Select the training data for Roman numeral task --
	local Xtrain_num = Xtrain[{ {}, {1,10} }]
	local Xtest_num = Xtest[{ {}, {1,10} }]
	local ytrain_num = ytrain[{ {}, 5 }]
	local ytest_num = ytest[{ {}, 5 }]

	-- Aggregate training and test sets
	local Xall_num = torch.cat(Xtrain_num, Xtest_num, 1)
	local yall_num = torch.cat(ytrain_num, ytest_num, 1)
	
	local maxi_num = torch.max(Xall_num)
	local outsz_num = torch.max(yall_num)
	
	-- Create global models and criterion
	model_num, criterion_num = make_model(maxi_num, outsz_num)

	-- Train
	print("# Training numeral subtask")
	train(Xtrain_num, ytrain_num, Xtest_num, ytest_num, model_num, criterion_num)
end

main()