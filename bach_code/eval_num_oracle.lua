-- eval.lua

--[[
	Experiment 1A: Baseline model with two-step harmonic decision process.
	The selection of harmony is divided into two steps: the choice of numeral followed by the choice of inversion.
--]]

-- Other libraries
require 'nn'
require 'hdf5'

function wait()
	local answer
	repeat
	   io.write("continue with this operation (y/n)? ")
	   io.flush()
	   answer=io.read()
	until answer=="y" or answer=="n"
end

--- Make the Model ---
function make_model(max_index, output_size)
	-- Embedding sequence
	local embedding = nn.Sequential()
	embedding:add(nn.LookupTable(max_index, embedding_size))
	embedding:add(nn.Sum(1))

	-- Feed forward sequence
	local model = nn.Sequential()
	model:add(embedding)
	-- model:add(nn.Dropout(0.5))
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
		local nll = 0
		model:zeroGradParameters()
		for j = 1, Xtrain:size(1) do
			-- Forward
			local out = model:forward(Xtrain[j])
			nll = nll + criterion:forward(out, ytrain[j])

			-- Backward
			local deriv = criterion:backward(out, ytrain[j])
			model:backward(Xtrain[j], deriv)
		end
		-- Update parameters
		model:updateParameters(learning_rate)
		print("Epoch:", epoch, nll)
		eval_num(Xtest, ytest, model, criterion)
	end
end



-- Evaluation numeral subtask --
function eval_num(Xtest, ytest, model, criterion)
	-- Collect numeral predictions
	local nll = 0
	local pred = torch.IntTensor(Xtest:size(1))
	for j = 1, Xtest:size(1) do
		-- Add in previous numeral choice
		if signal_col[j] == 1 then 
			Xtest[j][prev_harm_idx] = prev_harm_max 
		else 
			Xtest[j][prev_harm_idx] = pred[j - 1] + prev_harm_min - 1 
		end
		out = model:forward(Xtest[j])
		_ , argmax = torch.max(out, 1)
		pred[j] = argmax[1]
		nll = nll + criterion:forward(out, ytest[j])
	end
	-- Evaluation
	local correct = torch.sum(torch.eq(pred, ytest))
	local count = pred:size(1)
	local nt = torch.ne(ytest, 1) -- not tonic
	local correct_nt = torch.sum(torch.eq(pred[nt], ytest[nt]))
	local count_nt = ytest[nt]:size(1)
	local nd = torch.ne(ytest[nt], 3) -- not tonic or dominant
	local correct_ntod = torch.sum(torch.eq(pred[nt][nd], ytest[nt][nd]))
	local count_ntod = ytest[nt][nd]:size(1)
	-- 	Print some output --
	for i = count - 100, count do print(pred[i], ytest[i]) end 
	-- Results --
	print(string.format("Nll: %.3f", nll))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
	print(string.format("Percentage correct (not tonic): %.2f%%", correct_nt / count_nt * 100.0))
	print(string.format("Percentage correct (not tonic or dominant): %.2f%%", correct_ntod / count_ntod * 100.0))
end




function main() 
	-- Contants
	embedding_size = 50
	hidden_size = 50
	epochs = 1000
	learning_rate = 0.00001

	-- Create the data loader class.
	local f = hdf5.open("data/all_data.hdf5")
	local Xtrain = f:read('X_train'):all()
	local ytrain = f:read('y_train'):all()
	local Xtest = f:read('X_test'):all()
	local ytest = f:read('y_test'):all()
	signal_col = f:read("test_signal_col"):all()
	f:close()
	
	-- use all data, including previous harmony (numeral[feature 11] only)
	prev_harm_idx = 11

	-- Select the training data for Roman numeral task --
	local Xtrain_num = Xtrain[{ {}, {1, prev_harm_idx} }]
	local Xtest_num = Xtest[{ {}, {1, prev_harm_idx} }]
	local ytrain_num = ytrain[{ {}, 1 }]
	local ytest_num = ytest[{ {}, 1 }]

	-- Aggregate training and test sets
	local Xall_num = torch.cat(Xtrain_num, Xtest_num, 1)
	local yall_num = torch.cat(ytrain_num, ytest_num, 1)

	-- Previous harmony range "no numeral"?
	prev_harm_col = Xall_num[{ {}, prev_harm_idx }]
	prev_harm_min = prev_harm_col:min() -- I chord
	prev_harm_max = prev_harm_col:max() -- "no numeral"

	local maxi_num = Xall_num:max()
	local outsz_num = yall_num:max()
	
	-- Create global models and criterion
	model_num, criterion_num = make_model(maxi_num, outsz_num)
	
	-- Train
	print("# Training numeral subtask")
	train(Xtrain_num, ytrain_num, Xtest_num, ytest_num, model_num, criterion_num)
end

main()