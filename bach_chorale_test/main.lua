require "rnn"
data = require "data"
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an RNN-LM')
cmd:text()
cmd:text('Options')

cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-epochs', 10, 'number of training epoch')
cmd:option('-batch_size', 10, '')
cmd:option('-learning_rate', 0.1, '')

cmd:option('-data_file','data/','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','data/','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
opt = cmd:parse(arg)



function train(data, valid_data, model, criterion)
   for epochs = 1, opt.epochs do
      data:reset_batch_pointer()
      nll = 0
      while true do 
         input, goal = data:next_batch()
         goal_tab = nn.SplitTable(1):forward(goal)

         model:zeroGradParameters()
         out = model:forward(input.target)
         nll = nll + criterion:forward(out, goal_tab)

         deriv = criterion:backward(out, goal_tab)
         model:backward(input.target, deriv)

         -- Do the LSTM-y stuff.
         model:updateParameters(opt.learning_rate)
         if data.batch_idx % 100 == 0 then
            print(data.batch_idx)
         end
         if data.batch_idx == 1 then
            break
         end
      end
      eval(valid_data)
   end
end

function eval(data)
   data:reset_batch_pointer()
   nll = 0
   while true do 
      input, goal = data:next_batch()
      
      out = model:forward(input.target)
      nll = nll + criterion:forward(out, goal)
      if data.batch_idx == 1 then
         break
      end
   end
   print(nll)
end

function make_model(data)
   local model = nn.Sequential()

   model:add(nn.LookupTable(data.target_size, opt.word_vec_size))
   model:add(nn.Dropout(opt.dropoutProb))
   model:add(nn.SplitTable(1, 2))
   
   model:add(nn.Sequencer(nn.LSTM(opt.word_vec_size, opt.rnn_size)))
   for j = 2, opt.num_layers do
      model:add(nn.Sequencer(nn.LSTM(opt.rnn_size, opt.rnn_size)))
      model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   end
   
   model:add(nn.Sequencer(nn.Linear(opt.rnn_size, data.target_size)))
   model:add(nn.Sequencer(nn.LogSoftMax()))
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
   return model, criterion
end

function main() 
    -- parse input params
   opt = cmd:parse(arg)

   
   if opt.gpuid >= 0 then
       print('using CUDA on GPU ' .. opt.gpuid .. '...')
       require 'cutorch'
       require 'cunn'
       cutorch.setDevice(opt.gpuid + 1)
   end
   
   -- Create the data loader class.

   loader = data.new(opt, opt.data_file)
   dev_loader = data.new(opt, opt.val_data_file)
   model, criterion = make_model(loader)
   
   protos = {}
   
   
   if opt.gpuid >= 0 then
      model:cuda()
      criterion:cuda()
   end
   
   train(loader, dev_loader, model, criterion)
end
main()