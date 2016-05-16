%% Loading the dataset
tic
Inputs_data = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/train-images.idx3-ubyte');
Inputs_data = Inputs_data';
Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
Targets_data = zeros(60000, 10);
for i = 1:60000
    Targets_data(i,Ot(i)+1) = 1;
end
clear Ot
disp(['Dataset loaded in ' num2str(toc) 's']);

%% Training the network
tic
disp('Training the network ...');

[ ftInput_weights, ftInput_biases, ftSorting_weights, ftBatch_bounds, ftOutput_weights ] = RealTimeELMtrain( Inputs_data, Targets_data, 1000, 10 );
disp(['Network trained in ' num2str(toc) 's']);

%% Training error
ftOutputs = RealTimeELMtest( Inputs_data, ftInput_weights, ftInput_biases, ftSorting_weights, ftBatch_bounds, ftOutput_weights );
disp(['first training accuracy is ' num2str(100*mean(Single_compare(ftOutputs, Targets_data))) '%']);

[~, Output_labels] = max(ftOutputs, [], 2);
[~, Target_labels] = max(Targets_data, [], 2);

Delta = Output_labels == Target_labels;

E = find(Delta);
D = find(~Delta);
EasyInputs = Inputs_data(E,:);
EasyTargets = Targets_data(E,:);
DifficultInputs = Inputs_data(D,:);
DifficultTargets = Targets_data(D,:);


%% Second train

disp('Training the network again...');

[ etInput_weights, etInput_biases, etSorting_weights, etBatch_bounds, etOutput_weights ] = RealTimeELMtrain( EasyInputs, EasyTargets, 1000, 10 );
disp(['Easy network trained in ' num2str(toc) 's']);

[ dtInput_weights, dtInput_biases, dtSorting_weights, dtBatch_bounds, dtOutput_weights ] = RealTimeELMtrain( DifficultInputs, DifficultTargets, 100, 10 );
disp(['Difficult network trained in ' num2str(toc) 's']);

%% Training error
etOutputs = RealTimeELMtest( EasyInputs, etInput_weights, etInput_biases, etSorting_weights, etBatch_bounds, etOutput_weights );
disp(['easy training accuracy is ' num2str(100*mean(Single_compare(etOutputs, EasyTargets))) '%']);

dtOutputs = RealTimeELMtest( DifficultInputs, dtInput_weights, dtInput_biases, dtSorting_weights, dtBatch_bounds, dtOutput_weights );
disp(['difficult training accuracy is ' num2str(100*mean(Single_compare(dtOutputs, DifficultTargets))) '%']);

%% Test Error

Inputs_data = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/t10k-images.idx3-ubyte');
Inputs_data = Inputs_data';
Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/t10k-labels.idx1-ubyte');
Targets_data = zeros(10000, 10);
for i = 1:10000
    Targets_data(i,Ot(i)+1) = 1;
end
clear Ot

Outputs = RealTimeELMtest( Inputs_data, ftInput_weights, ftInput_biases, ftSorting_weights, ftBatch_bounds, ftOutput_weights );
x = Decide(Outputs);

E = find(x>0.1);
D = find(x<=0.1);
EasyInputs = Inputs_data(E,:);
EasyTargets = Targets_data(E,:);
DifficultInputs = Inputs_data(D,:);
DifficultTargets = Targets_data(D,:);

etOutputs = RealTimeELMtest( EasyInputs, etInput_weights, etInput_biases, etSorting_weights, etBatch_bounds, etOutput_weights );
disp(['easy training train accuracy is ' num2str(100*mean(Single_compare(etOutputs, EasyTargets))) '%']);

dtOutputs = RealTimeELMtest( DifficultInputs, dtInput_weights, dtInput_biases, dtSorting_weights, dtBatch_bounds, dtOutput_weights );
disp(['difficult training train accuracy is ' num2str(100*mean(Single_compare(dtOutputs, DifficultTargets))) '%']);


dOutput = Single_compare(dtOutputs, DifficultTargets);
eOutput = Single_compare(etOutputs, EasyTargets);

Output = zeros(size(Targets_data,1),1);
Output(E) = eOutput;
Output(D) = dOutput;
disp( num2str(mean(Output)));