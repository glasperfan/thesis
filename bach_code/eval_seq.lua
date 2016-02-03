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
	local count = Xtest:size(1)
	-- Results --
	print(string.format("Average nll: %.3f", torch.mean(nll_arr)))
	print(string.format("Percentage correct: %.2f%%", correct / count * 100.0))
end


-- Evaluate the entire sequence of tasks
function eval_seq()
	correctnum = 0
	correctinv = 0
	correctalt = 0
	correctten = 0
	correcthar = 0 -- Harmony (numeral + inversion) correct
	correctall = 0 -- All aspects correct
	nllnum = 0.0
	nllinv = 0.0
	nllalt = 0.0
	nllten = 0.0
	for j = 1, Xtestnum:size(1) do
		-- Numeral --
		input = Xtestnum[j]
		out = modelnum:forward(input)
		nllnum = nllnum + criterionnum:forward(out, ytestnum[j])
		_ , argmax = torch.max(out, 1)
		prednum = argmax[1]
		prednum_indexed = prednum + Xallnum:max() - 1
		
		-- Inversion --
		input = input:cat(torch.IntTensor({prednum_indexed}))
		out = modelinv:forward(input)
		nllinv = nllinv + criterioninv:forward(out, ytestinv[j])
		_ , argmax = torch.max(out, 1)
		predinv = argmax[1]
		predinv_indexed = predinv + Xallinv:max() - 1

		-- Alto --
		input = input:cat(torch.IntTensor({predinv_indexed}))
		out = modelalt:forward(input)
		nllalt = nllalt + criterionalt:forward(out, ytestalt[j])
		_ , argmax = torch.max(out, 1)
		predalt = argmax[1]
		predalt_indexed = predalt + Xallalt:max() - 1

		-- Tenor --
		input = input:cat(torch.IntTensor({predalt_indexed}))
		out = modelten:forward(input)
		nllten = nllten + criterionten:forward(out, ytestten[j])
		_ , argmax = torch.max(out, 1)
		predten = argmax[1]
		predten_indexed = predten + Xallten:max() - 1

		if prednum == ytestnum[j] then correctnum = correctnum + 1 end
		if predinv == ytestinv[j] then correctinv = correctinv + 1 end
		if predalt == ytestalt[j] then correctalt = correctalt + 1 end
		if predten == ytestten[j] then correctten = correctten + 1 end
		if prednum == ytestnum[j] and predinv == ytestinv[j] then
			correcthar = correcthar + 1
		end
		if prednum == ytestnum[j] and predinv == ytestinv[j] and predalt == ytestalt[j] and predten == ytestten[j] then
			correctall = correctall + 1
		end
	end

	print("# Roman numeral task")
	print(string.format("Average nll: %.3f", nllnum / ytestnum:size(1)))
	print(string.format("Percentage correct: %.2f%%", correctnum / ytestnum:size(1) * 100.0))
	print("# Inversion task")
	print(string.format("Average nll: %.3f", nllinv / ytestnum:size(1)))
	print(string.format("Percentage correct: %.2f%%", correctinv / ytestnum:size(1) * 100.0))
	print("# Alto voice task")
	print(string.format("Average nll: %.3f", nllalt / ytestnum:size(1)))
	print(string.format("Percentage correct: %.2f%%", correctalt / ytestnum:size(1) * 100.0))
	print("# Tenor voice task")
	print(string.format("Average nll: %.3f", nllten / ytestnum:size(1)))
	print(string.format("Percentage correct: %.2f%%", correctten / ytestnum:size(1) * 100.0))
	print("# Harmonic accuracy")
	print(string.format("Percentage correct: %.2f%%", correcthar / ytestnum:size(1) * 100.0))
	print("# Perfect accuracy")
	print(string.format("Percentage correct: %.2f%%", correctall / ytestnum:size(1) * 100.0))
end



function main() 
	-- Contants
	embedding_size = 250
	hidden_size = 250
	epochs = 20
	learning_rate = 0.01

	-- Load data
	local f = hdf5.open("data/choralesseq.hdf5")
	Xtrainnum = f:read('Xtrainnum'):all()
	ytrainnum = f:read('ytrainnum'):all()
	Xdevnum = f:read('Xdevnum'):all()
	ydevnum = f:read('ydevnum'):all()
	Xtestnum = f:read('Xtestnum'):all()
	ytestnum = f:read('ytestnum'):all()
	Xallnum = f:read('Xallnum'):all()
	yallnum = f:read('yallnum'):all()

	Xtraininv = f:read('Xtraininv'):all()
	ytraininv = f:read('ytraininv'):all()
	Xdevinv = f:read('Xdevinv'):all()
	ydevinv = f:read('ydevinv'):all()
	Xtestinv = f:read('Xtestinv'):all()
	ytestinv = f:read('ytestinv'):all()
	Xallinv = f:read('Xallinv'):all()
	yallinv = f:read('yallinv'):all()

	Xtrainalt = f:read('Xtrainalt'):all()
	ytrainalt = f:read('ytrainalt'):all()
	Xdevalt = f:read('Xdevalt'):all()
	ydevalt = f:read('ydevalt'):all()
	Xtestalt = f:read('Xtestalt'):all()
	ytestalt = f:read('ytestalt'):all()
	Xallalt = f:read('Xallalt'):all()
	yallalt = f:read('yallalt'):all()

	Xtrainten = f:read('Xtrainten'):all()
	ytrainten = f:read('ytrainten'):all()
	Xdevten = f:read('Xdevten'):all()
	ydevten = f:read('ydevten'):all()
	Xtestten = f:read('Xtestten'):all()
	ytestten = f:read('ytestten'):all()
	Xallten = f:read('Xallten'):all()
	yallten = f:read('yallten'):all()

	f:close()

	-- Make models --
	modelnum, criterionnum = make_model(Xallnum:max(), yallnum:max())
	modelinv, criterioninv = make_model(Xallinv:max(), yallinv:max())
	modelalt, criterionalt = make_model(Xallalt:max(), yallalt:max())
	modelten, criterionten = make_model(Xallten:max(), yallten:max())

	-- Train each model --
	train(Xtrainnum, ytrainnum, modelnum, criterionnum)
	train(Xtraininv, ytraininv, modelinv, criterioninv)
	train(Xtrainalt, ytrainalt, modelalt, criterionalt)
	train(Xtrainten, ytrainten, modelten, criterionten)

	-- Evaluate each model --
	-- print("# Evaluating Roman numeral model")
	-- eval(Xtestnum, ytestnum, modelnum, criterionnum)
	-- eval(Xtestinv, ytestinv, modelinv, criterioninv)
	-- eval(Xtestalt, ytestalt, modelalt, criterionalt)
	-- eval(Xtestten, ytestten, modelten, criterionten)

	-- Evaluate sequence --
	eval_seq()
end

main()