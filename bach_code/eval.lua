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
			out = model:forward(Xtrain[j])
			nll = nll + criterion:forward(out, ytrain[j])

			-- Backward
			deriv = criterion:backward(out, ytrain[j])
			model:backward(Xtrain[j], deriv)
		end
		-- Update parameters
		model:updateParameters(learning_rate)
		print("Epoch:", epoch, nll)
		eval_num(Xtest, ytest, model, criterion)
	end
end



-- Evaluation inversion subtask --
function eval_num(Xtest, ytest, is_oracle)
	-- Collect numeral predictions
	local out = model_num:forward(Xtest:t())
	local _, pred = torch.max(out, 2)
	print(torch.max(pred), torch.min(pred), torch.max(ytest), torch.min(ytest))
	nll = criterion_num:forward(ytest:double(), )
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


-- Evaluation inversion subtask --
function eval_inv(Xtest, ytest, model, criterion)
	local nll_arr = {}
	local count = 0.0
	local correct = 0.0
	local tp = 0.0 -- true positives (pred = 1, goal = 1)
	local fp = 0.0 -- false positives (pred = 1, goal = 2)
	local tn = 0.0 -- true negatives (pred = 2, goal = 2)
	local fn = 0.0 -- false negatives (pred = 2, goal = 1)
	for i = 1, Xtest:size(1) do
		local out = model:forward(Xtest[i])
		local max, argmax = torch.max(out, 1)
		local pred = argmax[1]
		local goal = ytest[i]
		nll_arr[i] = criterion:forward(out, goal)
		
		-- Measure accuracy
		if pred == goal then correct = correct + 1 end
		count = count + 1
		if pred == 1 and goal == 1 then tp = tp + 1 end
		if pred == 1 and goal == 2 then fp = fp + 1 end
		if pred == 2 and goal == 2 then tn = tn + 1 end
		if pred == 2 and goal == 1 then fn = fn + 1 end

		-- Print some output --
		if i <= Xtest:size(1) and i > Xtest:size(1) - 100 then print(pred, goal) end 
	end
	nll_arr = torch.Tensor(nll_arr)
	print(string.format("Average nll: %.3f", torch.mean(nll_arr) ))
	print(string.format("Total nll: %.3f", torch.sum(nll_arr) ))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
	-- print(string.format("Sensitivity (true positive rate): %.2f%%", tp / (tp + fn) * 100.0 ))
	-- print(string.format("Specificity (true negative rate): %.2f%%", tn / (tn + fp) * 100.0 ))
	-- print(string.format("F1 score: %.3f", (2 * tp) / (2 * tp + fp + fn) ))
	-- print(string.format("Percentage correct (not root): %.2f%%", correct_n1 / count_n1 * 100.0))
	-- print(string.format("Percentage correct (not root or 6): %.2f%%", correct_n2 / count_n2 * 100.0))
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
	f:close()
	
	-- Select the training data for Roman numeral task --
	local Xtrain_num = Xtrain[{ {}, {1,10} }]
	local Xtest_num = Xtest[{ {}, {1,10} }]
	local ytrain_num = ytrain[{ {}, 1 }]
	local ytest_num = ytest[{ {}, 1 }]

	-- Select the training data for the inversion task --
	local Xtrain_inv = torch.cat(Xtrain[{ {}, {1,10} }], ytrain[{ {}, 1}], 2)
	local Xtest_inv = torch.cat(Xtest[{ {}, {1,10} }], ytest[{ {}, 1}], 2)
	local ytrain_inv = ytrain[{ {}, 2 }]
	local ytest_inv = ytest[{ {}, 2 }]
	
	-- Remove examples in root position and 1st inversion
	-- mask = ytrain_inv:gt(2)
	-- mask_test = ytest_inv:gt(2)
	-- indices = torch.linspace(1,mask:size(1),mask:size(1)):long()
	-- indices_test = torch.linspace(1,mask_test:size(1),mask_test:size(1)):long()
	-- selected = indices[mask:eq(1)]
	-- selected_test = indices_test[mask_test:eq(1)]
	-- Xtrain_inv = Xtrain_inv:index(1, selected)
	-- Xtest_inv = Xtest_inv:index(1, selected_test)
	-- ytrain_inv = ytrain_inv:index(1, selected):apply(function(x) return x - 2 end)
	-- ytest_inv = ytest_inv: index(1, selected_test):apply(function(x) return x - 2 end)
	-- ytrain_inv:apply(function(x) if x == 1 then return 1 else return 2 end end)
	-- ytest_inv:apply(function(x) if x == 1 then return 1 else return 2 end end)

	-- Index correction
	local maxi_inv = torch.max(Xtrain_inv[{ {}, {1, 10} }])
	Xtrain_inv[{ {}, 11 }]:apply(function(x) return x + maxi_inv end)
	Xtest_inv[{ {}, 11 }]:apply(function(x) return x + maxi_inv end)

	-- Aggregate training and test sets
	local Xall_num = torch.cat(Xtrain_num, Xtest_num, 1)
	local yall_num = torch.cat(ytrain_num, ytest_num, 1)
	local Xall_inv = torch.cat(Xtrain_inv, Xtest_inv, 1)
	local yall_inv = torch.cat(ytrain_inv, ytest_inv, 1)
	
	local maxi_num = torch.max(Xall_num)
	local maxi_inv = torch.max(Xall_inv)
	local outsz_num = torch.max(yall_num)
	local outsz_inv = torch.max(yall_inv)
	
	-- Create global models and criterion
	model_num, criterion_num = make_model(maxi_num, outsz_num)
	model_inv, criterion_inv = make_model(maxi_inv, outsz_inv)
	
	-- Train
	-- print("# Training numeral subtask")
	-- train(Xtrain_num, ytrain_num, model_num, criterion_num)
	-- print()
	print("# Training numeral subtask")
	train(Xtrain_num, ytrain_num, Xtest_num, ytest_num, model_num, criterion_num)
	-- print("# Training inversion subtask")
	-- train(Xtrain_inv, ytrain_inv, Xtest_inv, ytest_inv, model_inv, criterion_inv)

	-- Evaluate
	-- eval_inv(Xtest_inv, ytest_inv, model_inv, criterion_inv)
	-- eval(Xtest_num, ytest_num, Xtest_inv, Xtest_inv, maxi_inv)

end

main()