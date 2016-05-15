%% Loading the dataset
tic
Inputs = loadMNISTImages('E:/Deep Learning/ELM Code Matlab/train-images.idx3-ubyte');
Inputs = Inputs';
Ot = loadMNISTLabels('E:/Deep Learning/ELM Code Matlab/train-labels.idx1-ubyte');
Targets = zeros(60000, 10);
for i = 1:60000
    Targets(i,Ot(i)+1) = 1;
end
clear Ot
disp(['Dataset loaded in ' num2str(toc) 's']);

%% Training the network
tic
disp('Training the autoencoder ...');
Network_weights = trainELMAutoEncoder( Inputs, [ 784 20 ] );
disp(['Network pre-trained in ' num2str(toc) 's']);

%% Training error
Outputs = crossNetwork( Inputs, Network_weights );

%% Testing the network
figure
for i = 1:18
    subplot(3,6,i)
    imagesc(reshape(Outputs(38*i,:),28,28));
    colormap(gray)
end






