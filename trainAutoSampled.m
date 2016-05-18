function [ output_args ] = trainSampled( Inputs_data, Targets_data, SamplingTree, treeshold )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Nleaf = size(SamplingTree,2);
I = zeros(size(Inputs_data,1), Nleaf);

for branch = 1:Nleaf-1
    
    [ Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights ] = RealTimeELMtrain( Inputs_data, Targets_data, 30, 10 );
    Outputs = RealTimeELMtest( Inputs_data, Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights );
    x = Decide(Outputs);
    I1 = find(x>treeshold);
    I2 = find(x<=treeshold);
    
    rdd de ta soeur, ok
    bah je ne sais pas pourquoi c'est ouvert moi
    
end


end

