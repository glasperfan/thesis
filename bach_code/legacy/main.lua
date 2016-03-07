require 'rnn'
require 'hdf5'
require 'nngraph'

cmd = torch.CmdLine()

cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-epochs', 10, 'number of training epoch')
cmd:option('-learning_rate', 1, '')
cmd:option('-max_grad_norm', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-dropoutProb', 0.5, 'dropoff param')

cmd:option('-data_file','data/','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','data/','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-param_init', 0.05, 'initialize parameters at')

-- CharCNN Options
cmd:option('-cudnn', 0, 'use cudnn')
cmd:option('-use_chars', 0, 'use characters')
cmd:option('-char_vec_size', 15, 'dimensionality of word embeddings')
cmd:option('-feature_maps', '{50,100,150,200,200,200,200}', 'number of feature maps in the CNN')
cmd:option('-kernels', '{1,2,3,4,5,6,7}', 'conv net kernel widths')

-- Highway Options
cmd:option('-highway_layers', 2, 'number of highway layers')

opt = cmd:parse(arg)


-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file, use_chars)
   local f = hdf5.open(data_file, 'r')
   self.target  = f:read('target'):all()
   self.use_chars = use_chars
   self.target_output = f:read('target_output'):all()
   self.target_size = f:read('target_size'):all()[1]

   self.length = self.target:size(1)
   self.seqlength = self.target:size(3)
   self.batchlength = self.target:size(2)
   if use_chars == 1 then 
      self.target = f:read('target_char'):all()
      self.charlength = self.target:size(4)
   end
end

function data:size()
   return self.length
end

function data.__index(self, idx)
   local input, target
   if type(idx) == "string" then
      return data[idx]
   elseif self.use_chars == 1 then 
      input = nn.View(self.seqlength, self.batchlength * self.charlength):forward(
               self.target[idx]:transpose(1, 2):float()):cuda():contiguous()
      target = nn.SplitTable(2):forward(self.target_output[idx]:float():cuda())

   else      
      input = self.target[idx]:transpose(1, 2):float():cuda()
      target = nn.SplitTable(2):forward(self.target_output[idx]:float():cuda())
   end
   return {input, target}
end

-- TDNN Unit
function make_tdnn(length, input_size, feature_maps, kernels)
   local layer1_concat, output
   local input = nn.Identity()() 
   local layer1 = {}
   for i = 1, #kernels do
      local reduced_l = length - kernels[i] + 1
      local pool_layer
      if opt.cudnn == 1 then
         local conv = cudnn.SpatialConvolution(1, feature_maps[i], input_size,
                                               kernels[i], 1, 1, 0)
         local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
         pool_layer = nn.Max(3)(nn.Max(3)(cudnn.SpatialMaxPooling(1, reduced_l, 1, 1, 0, 0)(
                                             nn.Tanh()(conv_layer))))
      else
         local conv = nn.TemporalConvolution(input_size, feature_maps[i], kernels[i])
         local conv_layer = conv(input)
         pool_layer = nn.TemporalMaxPooling(reduced_l)(nn.Tanh()(conv_layer))
         pool_layer = nn.Max(2)(pool_layer)         
      end
      table.insert(layer1, pool_layer)
   end
   print(#layer1)
   if #kernels > 1 then
      layer1_concat = nn.JoinTable(2)(layer1)
      output = layer1_concat
   else
      output = layer1[1]
   end
   return nn.gModule({input}, {output})
end

function make_highway(size, num_layers, bias, f)
    local output, transform_gate, carry_gate
    local num_layers = num_layers or 1
    local bias = bias or -2
    local f = f or nn.ReLU()
    local input = nn.Identity()()
    local inputs = {[1]=input}
    for i = 1, num_layers do        
        output = f(nn.Linear(size, size)(inputs[i]))
        transform_gate = nn.Sigmoid()(
           nn.AddConstant(bias)(nn.Linear(size, size)(inputs[i])))
        carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
        output = nn.CAddTable()({
              nn.CMulTable()({transform_gate, output}),
              nn.CMulTable()({carry_gate, inputs[i]})})
        table.insert(inputs, output)
    end
    return nn.gModule({input},{output})
end


function train(data, valid_data, model, criterion)
   local last_score = 1e9
   local params, grad_params = model:getParameters()
   params:uniform(-opt.param_init, opt.param_init)
   for epochs = 1, opt.epochs do
      model:training()
      for i = 1, data:size() do 
         model:zeroGradParameters()
         local d = data[i]
         input, goal = d[1], d[2]
         local out = model:forward(input)
         model:print_size()
         exit()
         local loss = criterion:forward(out, goal)         
         deriv = criterion:backward(out, goal)
         model:backward(input, deriv)
         -- Grad Norm.
         local grad_norm = grad_params:norm()
         if grad_norm > opt.max_grad_norm then
            grad_params:mul(opt.max_grad_norm / grad_norm)
         end
         
         params:add(grad_params:mul(-opt.learning_rate))

         -- Kill padding words
         for j = 1, #model.lookups_zero do
            model.lookups_zero[j].weight[1]:zero()
         end
         
         if i % 100 == 0 then
            print(i, data:size(),
                  math.exp(loss/ data.seqlength), opt.learning_rate)
         end
      end
      local score = eval(valid_data, model)
      if score > last_score - 1 then
         opt.learning_rate = opt.learning_rate / 2
      end
      last_score = score
   end
end

function eval(data, model)
   -- Validation
   model:evaluate()
   local nll = 0
   local total = 0 
   for i = 1, data:size() do
      local d = data[i]
      local input, goal = d[1], d[2]
      out = model:forward(input)
      nll = nll + criterion:forward(out, goal) * data.batchlength
      total = total + data.seqlength * data.batchlength
   end
   local valid = math.exp(nll / total)
   print("Valid", valid)
   return valid
end

function make_model(train_data)
   local model = nn.Sequential()
   model.lookups_zero = {}
   if opt.use_chars == 0 then 
      model:add(nn.LookupTable(train_data.target_size, opt.word_vec_size))
      model:add(nn.SplitTable(1, 3))
   else

      table.insert(model.lookups_zero, char_lookup)
      local char_lookup = nn.LookupTable(train_data.target_size, opt.char_vec_size)
      model.lookups_zero = {char_lookup}
      model:add(char_lookup)
      model:add(nn.View(train_data.seqlength, train_data.batchlength,
                        train_data.charlength, opt.char_vec_size))
      model:add(nn.SplitTable(1, 2))
      model:add(nn.Sequencer(nn.Copy(CudaTensor, CudaTensor, true)))
      
      local tdnn_input = make_tdnn(train_data.charlength, opt.char_vec_size,
                                   opt.feature_maps, opt.kernels)
      model:add(nn.Sequencer(tdnn_input))

      local input_size_L = torch.Tensor(opt.feature_maps):sum()
      if opt.highway_layers > 0 then
         local highway_mlp = make_highway(input_size_L, opt.highway_layers)
         model:add(nn.Sequencer(highway_mlp))
      end
      -- Map to the RNN size.
      model:add(nn.Sequencer(nn.Linear(input_size_L, opt.rnn_size)))
   end

   model:add(nn.Sequencer(nn.FastLSTM(opt.word_vec_size, opt.rnn_size)))   
   for j = 2, opt.num_layers do
      -- model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
      model:add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size)))
   end

   -- model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   model:add(nn.Sequencer(nn.Linear(opt.rnn_size, 1)))
   model:add(nn.Sequencer(nn.LogSoftMax()))

   model:remember('both') 
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
   
   return model, criterion
end

function main() 
    -- parse input params
   opt = cmd:parse(arg)
   loadstring('opt.kernels = ' .. opt.kernels)() 
   loadstring('opt.feature_maps = ' .. opt.feature_maps)() 

   
   if opt.gpuid >= 0 then
      -- Use CuDNN for temporal convolution.
      if not cudnn then require 'cudnn' end

      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.gpuid + 1)
   end
   
   -- Create the data loader class.
   local train_data = data.new(opt, opt.data_file, opt.use_chars)
   local valid_data = data.new(opt, opt.val_data_file, opt.use_chars)
   model, criterion = make_model(train_data)
   
   if opt.gpuid >= 0 then
      model:cuda()
      criterion:cuda()
   end
   train(train_data, valid_data, model, criterion)
end

-- Use this to permute the internal state of RNN/LSTM type modules
function nn.Module:print_size()
   if self.modules then
      for i,module in ipairs(self.modules) do
         if torch.type(module) == "nn.FastLSTM" then
            for k, v in pairs(module) do
               local t = torch.type(v)
               print(k, t)
               if t == "table"  then
                  for k2, v2 in pairs(v) do 
                     print(k2, v2:size())
                  end
               elseif t == "torch.CudaTensor" then
                  print(k, v:size())
               end
            end
         else
            module:print_size()
         end
      end
   end
   return self
end


main()
