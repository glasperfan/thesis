require 'hdf5'

local data = torch.class("data")

-- Adapted from /group/torch/rnn_lm/data.lua
function data:__init(data_file)
   local f = hdf5.open(data_file, 'r')
   self.target  = f:read('y'):all()
   self.source  = f:read('X'):all()
   self.shapes = f:read('shapes'):all()
   self.source_size = self.shapes[0]
   self.feature_size = self.shapes[1]
   self.batch_idx = 1
   self.split_sizes = self.target:size(1)
end

function data:reset_batch_pointer()
   self.batch_idx = 1
end

function data:next_batch()
    local timer = torch.Timer()
    self.batch_idx = self.batch_idx + 1
    if self.batch_idx > self.split_sizes then
        -- Cycle around to beginning
        self.batch_idx = 1 
    end

    -- Pull out the correct next batch.
    local idx = self.batch_idx
    local sources = self.source:index(1, self.indices[idx]:view(-1))
    sources = sources:view(self.indices[idx]:size(1), 
                           self.indices[idx]:size(2), 100)
    -- TODO: Concat all the batches.

    local input = {target = self.target[idx], 
                   source = sources}
    local output = self.target_output[idx]
    if opt.gpuid >= 0 then 
        input.target = input.target:float():cuda()
        input.source = input.source:float():cuda()
        output = output:float():cuda()
    end
    -- print(timer:time().real)
    -- print("target source output")
    -- print(input.target[1], input.source[1], output[1])
    return input, output
end

return data