-- eval.lua

--[[
	Experiment 1: Construct a vanilla neural network for learning on the chorales.
	Adapted from harvardnlp/group/torch/rnn_lm/simple/main.lua.
--]]

-- Load the data class.
require 'load'

-- Other libraries
require 'nn'




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
function train(X, y, model, criterion, learning_rate, epochs, data_obj)
	for epoch = 1, epochs do
		nll = 0
		model:zeroGradParameters()
		for j = 1, X:size(1) do
			-- Forward
			out = model:forward(X[j])
			nll = nll + criterion:forward(out, y[j])

			-- Backward
			deriv = criterion:backward(out, y[j])
			model:backward(X[j], deriv)
		end
		-- Update parameters
		model:updateParameters(learning_rate)
		print("Epoch:", epoch, nll)
		eval(data_obj)
	end
end



-- Training with stochastic gradient descent --
function trainSGD(X, y, model, criterion, learning_rate, epochs, data_obj)
	dataset = {}
	function dataset:size() return X:size(1) end
	for i = 1, X:size(1) do
		dataset[i] = {X[i], y[i]}
	end
	trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = learning_rate
	trainer.maxIteration = epochs
	trainer:train(dataset)

	eval(data_obj, model, criterion)
end



-- Evaluation with NLL --
function eval(data_obj)
	data_obj:reset_test_pointer()
	local nll_arr = {}
	local count = 0.0
	local correct = 0.0
	local count_nt = 0.0
	local correct_nt = 0.0
	while true do
		input, goal = data_obj:next_test()
		out = model:forward(input)
	    nll_arr[data_obj.test_pointer + 1] = criterion:forward(out, goal)
		-- Measure accuracy
		for i = 1, input:size(1) do
				max, argmax = torch.max(model:forward(input[i]), 1)
				answer = goal[i]
				pred = argmax[1]
				if pred == answer then correct = correct + 1 end
				count = count + 1
				if answer ~= 1 and answer ~= 13 then
					if pred == answer then correct_nt = correct_nt + 1 end
					count_nt = count_nt + 1
				end
				-- Print results from one chorale
				if data_obj.test_pointer == 0 then print(pred, answer) end
		end
		-- Print results from one chorale, and then break
		if data_obj.test_pointer == 0 then break end
	end
	nll_arr = torch.Tensor(nll_arr)
	print(string.format("Average nll: %.3f", torch.mean(nll_arr) ))
	print(string.format("Total nll: %.3f", torch.sum(nll_arr) ))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
	print(string.format("Percentage correct (not tonic): %.2f%%", correct_nt / count_nt * 100.0))
	return nll
end


-- Cross-validation with NLL --
-- function cross_val(X, y, chorale_data, model, criterion, max_index, output_size)
-- 	local embedding_sizes = {10, 20, 30, 40, 80}
-- 	local hidden_sizes = {5, 15, 30, 60}
-- 	local epochs = {10, 30}
-- 	local learning_rates = {0.1, 0.01, 0.001, 0.0001, 0.00001}
-- 	local best_params = {nll = 10000, embedding = nil, hidden = nil, epoch = nil, learning = nil}
-- 	for i1, embedding in ipairs(embedding_sizes) do
-- 		for i2, hidden_size in ipairs(hidden_sizes) do
-- 			for i3, epoch in ipairs(epochs) do
-- 				for i4, learning_rate in ipairs(learning_rates) do
-- 					print("Using...", embedding, hidden_size, epoch, learning_rate)
-- 					local model, criterion = make_model(max_index, embedding, hidden_size, output_size)
-- 					trainSGD(X, y, model, criterion, learning_rate, epoch)
-- 					nll = eval(chorale_data, model, criterion)
-- 					print(nll)
-- 					print(best_params.nll)
-- 					if nll < best_params.nll then
-- 						print("Found better...")
-- 						best_params = {nll = nll, embedding = embedding, hidden_size = hidden_size, epoch = epoch, learning = learning_rate}
-- 					end
-- 				end
-- 			end
-- 		end
-- 	end
-- 	print("BEST:")
-- 	print(best_params)
-- end




function main() 
	-- Contants
	local embedding_size = 100
	local hidden_size = 200
	local epochs = 500
	local learning_rate = 0.00001

	-- Create the data loader class.
	local data_obj = data.new('data/')
	local X = data_obj.train.all.X
	local y = data_obj.train.all.y
	local max_index = torch.max(data_obj.all.X)
	local output_size = torch.max(data_obj.all.y)
	
	model, criterion = make_model(max_index, embedding_size, hidden_size, output_size)

	-- Train
	train(X, y, model, criterion, learning_rate, epochs, data_obj)
	-- trainSGD(X, y, model, criterion, learning_rate, epochs, data_obj)
	-- eval(data, testing_model, criterion)
end

main()