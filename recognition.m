%Program to recognise hand-written digits using Deep CNN
%Giving path of Dataset folder
digitDatasetPath='D:\ml_project\mnist_test.csv';

%Reading image from digit dataset folder
digitimages= imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%Distributing images in training and testing
numTrainFiles=0.75;
[TrainImages, TestImages]= splitEachLabel(digitimages, numTrainFiles, 'randomize');


%-----------------------------Building CNN-------------------------------------------------------------------------------------------------------------------------------------------------------------

layers=[
             imageInputLayer([28 28 3], 'Name', 'Input')
             
             convolution2dLayer(3,8, 'Padding', 'same', 'Name', 'Conv_1')
             batchNormalizationLayer('Name', 'BN_1')
             reluLayer('Name','Relu_1')
             maxPooling2dLayer(2, 'Stride', 2, 'Name', 'Maxpool_1')
             
              convolution2dLayer(3,16, 'Padding', 'same', 'Name', 'Conv_2')
             batchNormalizationLayer('Name', 'BN_2')
             reluLayer('Name','Relu_2')
             maxPooling2dLayer(2, 'Stride', 2, 'Name', 'Maxpool_2')
             
              convolution2dLayer(3,32, 'Padding', 'same', 'Name', 'Conv_3')
             batchNormalizationLayer('Name', 'BN_3')
             reluLayer('Name','Relu_3')
             maxPooling2dLayer(2, 'Stride', 2, 'Name', 'Maxpool_3')
             
              convolution2dLayer(3,64, 'Padding', 'same', 'Name', 'Conv_4')
             batchNormalizationLayer('Name', 'BN_4')
             reluLayer('Name','Relu_4')
             maxPooling2dLayer(2, 'Stride', 2, 'Name', 'Maxpool_4')
             
             fullyConnectedLayer(10,'Name', 'FC')
             softmaxLayer('Name', 'Softmax');
             classificationLayer('Name', 'Output Classification');
             ];
             
             %Plotting network structure
             lgraph=layerGraph(layers);
             plot(lgraph);
             analyzeNetwork(layers);
             
             %----------------------Training Options--------------------------------------------------------------------------------------------------------------------------------------------------
             options=trainingOptions('sgdm','InitialLearnRate',0.1,'MaxEpochs', 20,'Shuffle', 'every-epoch', 'ValidationData', TestImages, 'ValidationFrequency',100,'Verbose', false,'Plots','training-progress');
             
             net=trainNetwork(TrainImages, layers, options);%network training
             YPred= classify(net, TestImages);
             YValidation= TestImages.Labels;
             figure,plotconfusion( YValidation, YPred);
             cm.ColumnSummary = 'column-normalized';
             cm.RowSummary = 'row-normalized';
             cm.Title = 'CIFAR-10 Confusion Matrix';
             figure,plotconfusion( YPred, YValidation);
             cm.ColumnSummary = 'column-normalized';
             cm.RowSummary = 'row-normalized';
             cm.Title = 'CIFAR-10 Confusion Matrix';

             %confusionmat(YValidation, YPred)
             accuracy=sum(YPred==YValidation)/numel(YValidation)
             
             
             
          
             