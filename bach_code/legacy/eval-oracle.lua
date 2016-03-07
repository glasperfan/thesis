-- eval.lua

--[[
	Experiment 1d: Oracle experiment. 
	Will knowing the previous harmony improve on the baseline model?
--]]

require 'nn'
require 'hdf5'


--- Make the Model ---
function make_model(max_index, embedding_size, hidden_size, output_size)
	-- Embedding sequence
	local embedding = nn.Sequential()
	embedding:add(nn.LookupTable(max_index, embedding_size))
	embedding:add(nn.Sum(1))

	-- Feed forward sequence
	local model = nn.Sequential()
	model:add(embedding)
	-- if is_training then model:add(nn.Dropout(0.5)) end
	model:add(nn.Linear(embedding_size, hidden_size))
	model:add(nn.Sigmoid())
	model:add(nn.Linear(hidden_size, output_size))
	model:add(nn.LogSoftMax())

	-- Criterion: negative log likelihood
	local criterion = nn.ClassNLLCriterion()

	return model, criterion
end



--- Train the Model ---
function train(X_train, y_train, X_test, y_test, model, criterion, learning_rate, epochs)
	for epoch = 1, epochs do
		local nll = 0
		model:zeroGradParameters()
		for j = 1, X_train:size(1) do
			-- Forward
			local out = model:forward(X_train[j])
			local nll = nll + criterion:forward(out, y_train[j])

			-- Backward
			local deriv = criterion:backward(out, y_train[j])
			model:backward(X_train[j], deriv)
		end
		-- Update parameters
		model:updateParameters(learning_rate)
		print("Epoch:", epoch, nll)
		eval(X_test, y_test)
	end
end


-- Evaluation with NLL --
function eval(X_test, y_test)
	local nll_arr = {}
	local count = 0.0
	local correct = 0.0
	local count_nt = 0.0
	local correct_nt = 0.0
	local prev_pred = 0
	local h_start_idx = max_index - output_size - 1
	for i = 1, X_test:size(1) do
		input = X_test[i]
		goal = y_test[i]
		-- If it's the beginning of the chorale, set the previous chord to a dominant harmony --
		if input[input:size(1)] == 1 then input[input:size(1)] = 3 + h_start_idx end
		-- Otherwise, set it to the previous harmony chosen --
		if input[input:size(1)] ~= 1 then input[input:size(1)] = prev_pred + h_start_idx end

		-- Make a prediction and calculate NLL
		out = model:forward(input)
	    nll_arr[i] = criterion:forward(out, goal)
		max, argmax = torch.max(out, 1)
		pred = argmax[1]
		
		-- Update accuracy count
		if pred == goal then correct = correct + 1 end
		count = count + 1
		if goal ~= 1 and goal ~= 13 then
			if pred == goal then correct_nt = correct_nt + 1 end
			count_nt = count_nt + 1
		end
		-- Remember the previous computation
		prev_pred = pred

		-- Print a sample
		if i < 500 and i < 550 then print(pred, goal) end
	end
	nll_arr = torch.Tensor(nll_arr)
	print(string.format("Average nll: %.3f", torch.mean(nll_arr) ))
	print(string.format("Total nll: %.3f", torch.sum(nll_arr) ))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
	print(string.format("Percentage correct (not tonic): %.2f%%", correct_nt / count_nt * 100.0))
	return nll
end



function main() 
	-- Contants
	local embedding_size = 100
	local hidden_size = 100
	local epochs = 300
	local learning_rate = 0.00001

	-- Load data
	local f = hdf5.open("data/oracle.hdf5", 'r')
	local X_train = f:read('X_train'):all()
	local X_test = f:read('X_test'):all()
	local y_train = f:read('y_train'):all()
	local y_test = f:read('y_test'):all()
	local X_all = torch.cat(X_train, X_test, 1)
	local y_all = torch.cat(y_train, y_test, 1)	
	max_index = torch.max(X_all)
	output_size = torch.max(y_all)
	
	model, criterion = make_model(max_index, embedding_size, hidden_size, output_size)

	-- Train and evaluate
	train(X_train, y_train, X_test, y_test, model, criterion, learning_rate, epochs)
end

main()