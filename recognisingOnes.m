%% Loading the dataset
tic
Inputs_data = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/train-images.idx3-ubyte');
Inputs_data = Inputs_data';
Inputs_test = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/t10k-images.idx3-ubyte');
Inputs_test = Inputs_test';

for j = 0:9
    
    Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
    Targets_data = zeros(size(Ot,1), 2);
    for i = 1:size(Ot,1)
        if Ot(i) == j
            Targets_data(i,1) = 1;
        else
            Targets_data(i,2) = 1;
        end
    end
    clear Ot
    
    Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/t10k-labels.idx1-ubyte');
%     I = find(Ot==j);
    Targets_test = zeros(size(Ot,1), 2);
    for i = 1:size(Ot,1)
        if Ot(i) == j
            Targets_test(i,1) = 1;
        else
            Targets_test(i,2) = 1;
        end
    end
    clear Ot
    
    %% train
    
    [Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights] = RealTimeELMtrain( Inputs_data, Targets_data, 300, 10 );
    Outputs = RealTimeELMtest( Inputs_data, Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights );
    disp([ 'Recognising ' num2str(j) ' labels, accuracy on train data : ' num2str(100*mean(Single_compare(Outputs, Targets_data))) '%'] );
    Outputs = RealTimeELMtest( Inputs_test, Input_weights, Input_biases, Sorting_weights, Batch_bounds, Output_weights );
    disp([ 'Recognising ' num2str(j) ' labels, accuracy on test data : ' num2str(100*mean(Single_compare(Outputs, Targets_test))) '%'] );
    
end