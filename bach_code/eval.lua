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
function make_model(max_index, embedding_size, hidden_size, output_size, is_training)
	-- Embedding sequence
	embedding = nn.Sequential()
	embedding:add(nn.LookupTable(max_index, embedding_size))
	embedding:add(nn.Sum(1))

	-- Feed forward sequence
	model = nn.Sequential()
	model:add(embedding)
	if is_training then model:add(nn.Dropout(0.5)) end
	model:add(nn.Linear(embedding_size, 100))
	model:add(nn.Sigmoid())
	model:add(nn.Linear(100, 75))
	model:add(nn.Sigmoid())
	model:add(nn.Linear(75, output_size))
	model:add(nn.LogSoftMax())

	-- Criterion: negative log likelihood
	criterion = nn.ClassNLLCriterion()

	return model, criterion
end



--- Train the Model ---
function train(X, y, model, criterion, epochs, learning_rate)
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
	end
end



-- Training with stochastic gradient descent --
function trainSGD(X, y, model, criterion, learning_rate, epochs)
	dataset = {}
	function dataset:size() return X:size(1) end
	for i = 1, X:size(1) do
		dataset[i] = {X[i], y[i]}
	end
	trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = learning_rate
	trainer.maxIteration = epochs
	trainer:train(dataset)
end



-- Evaluation with NLL --
function eval(data, model, criterion)
	data:reset_test_pointer()
	local nll = 0
	while true do
		local chorale = data:next_test()
		local X = chorale.X
		local y = chorale.y
		for i = 1, X:size(1) do
			local guess = model:forward(X[i])
			local maxg, indexg = torch.max(guess, 1)
			-- print(guess:size())
			local answer = y[i]
			if indexg == answer then nll = nll + 1 end
			print(indexg, answer)
		end
		-- local out = model:forward(X)
		-- print(X:size(), y:size(), out:size())
		-- local maxs, indices = torch.max(out, 2)
		-- print(X:size())
		-- local cost = criterion:forward(out, y)
		-- print(cost)
      	-- nll = nll + cost
		if data.test_pointer == 0 then break end
	end
	print("Total cost:", nll)
	return nll
end


-- Cross-validation with NLL --
function cross_val(X, y, chorale_data, model, criterion, max_index, output_size)
	local embedding_sizes = {10, 20, 30, 40, 80}
	local hidden_sizes = {5, 15, 30, 60}
	local epochs = {10, 30}
	local learning_rates = {0.1, 0.01, 0.001, 0.0001, 0.00001}
	local best_params = {nll = 10000, embedding = nil, hidden = nil, epoch = nil, learning = nil}
	for i1, embedding in ipairs(embedding_sizes) do
		for i2, hidden_size in ipairs(hidden_sizes) do
			for i3, epoch in ipairs(epochs) do
				for i4, learning_rate in ipairs(learning_rates) do
					print("Using...", embedding, hidden_size, epoch, learning_rate)
					local model, criterion = make_model(max_index, embedding, hidden_size, output_size)
					trainSGD(X, y, model, criterion, learning_rate, epoch)
					nll = eval(chorale_data, model, criterion)
					print(nll)
					print(best_params.nll)
					if nll < best_params.nll then
						print("Found better...")
						best_params = {nll = nll, embedding = embedding, hidden_size = hidden_size, epoch = epoch, learning = learning_rate}
					end
				end
			end
		end
	end
	print("BEST:")
	print(best_params)
end




function main() 
   	-- Contants (TODO: create command-line arguments)
	local embedding_size = 100
	local hidden_size = 2000
	local epochs = 20
	local learning_rate = 0.01

   -- Create the data loader class.
   local chorale_data = data.new('data/')
   local X = chorale_data.train.all.X
   local y = chorale_data.train.all.y
   local max_index = torch.max(chorale_data.all.X)
   local output_size = torch.max(chorale_data.all.y)
   -- TODO: add validation and test sets
   -- local valid_data = data.new(opt, opt.val_data_file, opt.use_chars)
   training_model, criterion = make_model(max_index, embedding_size, hidden_size, output_size, true)
   testing_model, criterion2 = make_model(max_index, embedding_size, hidden_size, output_size, false)

   -- Train
   -- train(X, y, model, criterion, epochs, learning_rate)
   trainSGD(X, y, model, criterion, learning_rate, epochs, true)
   eval(chorale_data, testing_model, criterion)
   -- cross_val(X, y, chorale_data, model, criterion, max_index, output_size)
end

main()