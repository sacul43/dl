function [ Network_weights ] = trainELMAutoEncoder( Inputs, Layers )
%ELMAutoEncoder Creates an auto-encoder by pretreating the inputs 
%   Detailed explanation goes here

I = cell(1,size(Layers,2) + 1);
Network_weights = cell(1,size(Layers,2));
I{1} = Inputs;

for Layer = 1 : size(Layers,2)
    
    [ ~, ~, Output_weights] = ClassicELMtrain( I{Layer}, I{Layer}, Layers(Layer) );
    Network_weights{Layer} = Output_weights';
    I{Layer + 1} = crossLayer( I{Layer}, Network_weights{Layer});
    
end


end

