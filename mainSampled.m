%% Loading the dataset
Inputs_train = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/train-images.idx3-ubyte');
Inputs_train = Inputs_train';
Inputs_test = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/t10k-images.idx3-ubyte');
Inputs_test = Inputs_test';

Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
Targets_train = zeros(10, size(Ot,1), 2);
for j = 1:10
    for i = 1:size(Ot,1)
        Targets_train(Ot(i)+1,i,1) = 1;        
    end
end
clear Ot

Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/t10k-labels.idx1-ubyte');
Targets_test = zeros(10, size(Ot,1), 2);
for j = 1:10
    for i = 1:size(Ot,1)
        Targets_test(Ot(i)+1,i,1) = 1;
    end
end
clear Ot

%% train
Nneurons = 100;
Nbatch = 100;
Input_size = 784;
Nlabel = 10;

Input_weights = zeros(Nlabel,Input_size,Nneurons);
Input_biases = zeros(Nlabel,1,Nneurons);
Sorting_weights = zeros(Nlabel,784,1);
Batch_bounds = zeros(Nlabel,Nbatch,2);
Output_weights = zeros(Nlabel,Nbatch,Nneurons,2);

trainOutputs = zeros(Nlabel,size(Targets_train,2),2);
testOutputs = zeros(Nlabel,size(Targets_test,2),2);

for i = 1:10
    [Iw, Ib, Sw, Bb, Ow] = RealTimeELMtrain( Inputs_train, squeeze(Targets_train(i,:,:)), Nneurons, Nbatch );

    trainOutputs(i,:,:) = RealTimeELMtest( Inputs_train, Iw, Ib, Sw, Bb, Ow );
    testOutputs(i,:,:) = RealTimeELMtest( Inputs_test, Iw, Ib, Sw, Bb, Ow );
    
    Input_weights(i,:,:) = Iw;
    Input_biases(i,:,:) = Ib;
    Sorting_weights(i,:,:) = Sw;
    Batch_bounds(i,:,:) = Bb;
    Output_weights(i,:,:,:) = Ow;
end

%% Loading data for test

Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
Targets_train = zeros(size(Ot,1), 10);
for i = 1:size(Ot,1)
    Targets_train(i,Ot(i)+1) = 1;
end
clear Ot

Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/t10k-labels.idx1-ubyte');
Targets_test = zeros(size(Ot,1), 10);
for i = 1:size(Ot,1)
    Targets_test(i,Ot(i)+1) = 1;
end
clear Ot

Outputs = squeeze(trainOutputs(:,:,1))';
disp(['train accuracy is ' num2str(100*mean(Single_compare(Outputs, Targets_train))) '%']);
Outputs = squeeze(testOutputs(:,:,1))';
disp(['test accuracy is ' num2str(100*mean(Single_compare(Outputs, Targets_test))) '%']);
