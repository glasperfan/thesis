-- eval.lua

--[[
	Experiment 1: Construct a vanilla neural network for learning on the chorales.
	Adapted from harvardnlp/group/torch/rnn_lm/simple/main.lua.
--]]

-- Other libraries
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
function train()
	for epoch = 1, epochs do
		local nll = 0
		model:zeroGradParameters()
		for j = 1, Xtrain:size(1) do
			-- Forward
			out = model:forward(Xtrain[j])
			nll = nll + criterion:forward(out, ytrain[j])

			-- Backward
			deriv = criterion:backward(out, ytrain[j])
			model:backward(Xtrain[j], deriv)
		end
		-- Update parameters
		model:updateParameters(learning_rate)
		print("Epoch:", epoch, nll)
		eval()
	end
end



-- Evaluation with NLL --
function eval()
	local nll_arr = {}
	local count = 0.0
	local correct = 0.0
	local count_nt = 0.0
	local correct_nt = 0.0
	for i = 1, Xtest:size(1) do
		local out = model:forward(Xtest[i])
		local max, argmax = torch.max(out, 1)
		local pred = argmax[1]
		local goal = ytest[i]
		nll_arr[i] = criterion:forward(out, goal)
		
		-- Measure accuracy
		if pred == goal then correct = correct + 1 end
		count = count + 1
		if goal ~= 1 and goal ~= 13 then
			if pred == goal then correct_nt = correct_nt + 1 end
			count_nt = count_nt + 1
		end

		-- Print some output --
		if i <= Xtest:size(1) and i > Xtest:size(1) - 100 then print(pred, goal) end 
	end
	nll_arr = torch.Tensor(nll_arr)
	print(string.format("Average nll: %.3f", torch.mean(nll_arr) ))
	print(string.format("Total nll: %.3f", torch.sum(nll_arr) ))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
	print(string.format("Percentage correct (not tonic): %.2f%%", correct_nt / count_nt * 100.0))
end




function main() 
	-- Contants
	embedding_size = 100
	hidden_size = 100
	epochs = 500
	learning_rate = 0.00001

	-- Create the data loader class.
	local f = hdf5.open("data/all_data.hdf5")
	Xtrain = f:read('X_train'):all()
	ytrain = f:read('y_train'):all()
	Xtest = f:read('X_test'):all()
	ytest = f:read('y_test'):all()
	f:close()
	
	-- Remove the harmony features for the baseline model --
	Xtrain = Xtrain[{ {}, {1,10}}]
	Xtest = Xtest[{ {}, {1,10}}]

	Xall = torch.cat(Xtrain, Xtest, 1)
	yall = torch.cat(ytrain, ytest, 1)

	local max_index = torch.max(Xall)
	local output_size = torch.max(yall)
	
	model, criterion = make_model(max_index, embedding_size, hidden_size, output_size)

	-- Train
	train()
end

main()