require 'nn'
require 'rnn'
require 'load-rec'

dat = data.new('data/')
X_train, y_train, X_test, y_test = data:all_data()

d = 25
n = X_train:size(2)

nX, nY = dat:max_indices()
r = nn.Recurrent(d, nn.LookupTable(nX, d), nn.Linear(d,d), nn.Tanh(), n)

h = {}
for i = 1, n do
	inputs = X_train[{{}, i}]
	h[i] = r:forward(inputs)
end
-- lstm = nn.Sequential()
-- lstm:add(nn.Sequencer(nn.LookupTable(nX, d)))
-- lstm:add(nn.Sequencer(nn.LSTM(d, d)))

model = nn.Sequential()
model:add(nn.Linear(d, nY))
model:add(nn.LogSoftMax())
dist = model:forward(h[n])

criterion = nn.ClassNLLCriterion()

model:reset()
learning_rate = 0.01
for epoch = 1, 20 do
	nll = 0
	r:forget()
	model:zeroGradParameters()
	h = {}
	for i = 1, n do
		inputs = X_train[{{}, i}]
		h[i] = r:forward(inputs)
		if i ~= n then
			r:backward(inputs, torch.zeros(X_train:size(1), d))
		end
	end

	out = model:forward(h[n])
	print(out:size())
	print(y_train:size())
	nll = nll + criterion:forward(out, y_train)

	deriv = criterion:backward(out, y_train)
	d2 = model:backward(h[n], deriv)

	r:backward(X_train[{{}, n}], d2)
	r:backwardThroughTime()

	model:updateParameters(learning_rate)
	r:updateParameters(learning_rate)
	print("Epoch:", epoch, nll)
end

