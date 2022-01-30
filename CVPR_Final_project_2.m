clear
close all
clc
%% The dataset
trainDatasetPath = fullfile('train');
imds = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

labelCount = countEachLabel(imds);
disp(labelCount)

% Set the image size
imageSize = [64 64];

% Split the provided training set in 85% for actual training set and 15%
% to be used as validation set
quotaForEachLabel = 0.85;
[imdsTrain, imdsValidation] = splitEachLabel(imds, quotaForEachLabel, 'randomize');

%% Data augmentation
% Configure image data augmentation. For our context, left to right
% reflections can result effective. Each image is reflected horizontally 
% with 50% probability when training the network.
aug = imageDataAugmenter('RandXReflection', true);

% Transform batches to augment and resize image data
augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'DataAugmentation', aug);
augImdsValidation = augmentedImageDatastore(imageSize, imdsValidation);

% If you would like to perform further augmentations, launch the
% second-to-last section instead of this one
%% Network design and training
layers = [
    imageInputLayer([64 64 1], 'Name', 'input', 'Normalization', 'rescale-zero-one')
    convolution2dLayer(3, 8, 'Stride', 1, 'Padding', 'same', ...
                        'Weights', 0.01.*randn(3,3,1,8), 'Bias', zeros(1,1,8), ...
                        'Name', 'conv_1')
    batchNormalizationLayer('Name', 'BN_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
    convolution2dLayer(5, 16, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'BN_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    convolution2dLayer(7, 32, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'BN_3')
    reluLayer('Name', 'relu_3')
    dropoutLayer(0.75, 'Name', 'dropout')
    fullyConnectedLayer(15, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

lgraph = layerGraph(layers);
analyzeNetwork(lgraph);

% Training options
miniBatchSize = 64;
valFrequency = floor(numel(augImdsTrain.Files)/(2*miniBatchSize));
options = trainingOptions('sgdm',...
                          'InitialLearnRate', 0.01, ...
                          'MaxEpochs', 30, ...
                          'MiniBatchSize', miniBatchSize, ...
                          'Shuffle', 'every-epoch', ...
                          'ValidationData', augImdsValidation, ...
                          'ValidationFrequency', valFrequency, ...
                          'ValidationPatience', 4, ...
                          'ExecutionEnvironment', 'parallel', ...
                          'Plots', 'training-progress');

% Train the net
net = trainNetwork(augImdsTrain, layers, options);

% Evaluate performance on test set
testDatasetPath  = fullfile('test');

imdsTest = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
augImdsTest = augmentedImageDatastore(imageSize, imdsTest);

% Apply the network to the test set
YPredicted = classify(net, augImdsTest);
YTest = imdsTest.Labels;

% Overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
plotconfusion(YTest, YPredicted)

%% Ensemble of networks
% Get an ensemble, sometimes referred as a committee of convnets, of
% independently trained network. Since it's a multiclass classification
% problem, the prediction will consider the majority voted classes by each
% netowork
numberOfNets = 5;

% Check if it's already present an ensemble of nets
if isfile('workspace\ensembleOfNets.mat')
    load('workspace\ensembleOfNets.mat', 'nets');
else
    % Train a new ensemble of nets
    nets = cell(1, numberOfNets);
    
    for i = 1:numberOfNets
        [imdsTrain, imdsValidation] = splitEachLabel(imds, quotaForEachLabel, 'randomize');
        augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'DataAugmentation', aug);
        augImdsValidation = augmentedImageDatastore(imageSize, imdsValidation);

        options = trainingOptions('sgdm',...
                                  'InitialLearnRate', 0.01, ...
                                  'MaxEpochs', 30, ...
                                  'MiniBatchSize', miniBatchSize, ...
                                  'Shuffle', 'every-epoch', ...
                                  'ValidationData', augImdsValidation, ...
                                  'ValidationFrequency', valFrequency, ...
                                  'ValidationPatience', 4, ...
                                  'ExecutionEnvironment', 'parallel', ...
                                  'Plots', 'training-progress');
        
        % Store the network in a list
        nets{1,i} = trainNetwork(augImdsTrain, layers, options);
    end

    % Save the new ensemble
    save('workspace\ensembleOfNets.mat', 'nets')
end

% Get the prediction values for each net of the ensemble
netsYPredicted = cell(1, numberOfNets);
for i = 1:numberOfNets
    netsYPredicted{1,i} = classify(nets{1,i}, augImdsTest);
end

% Apply majority voting to create a new set average predicted categories
majorityYPredicted = mode(cat(2, netsYPredicted{:}), 2);

% Overall accuracy
accuracy = sum(majorityYPredicted == YTest)/numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
plotconfusion(YTest, majorityYPredicted)

%% Data augmentation with random cropping
% Task 4 (optional)
% If preferred, launch this section insted of the previous one to perform
% further augmentations such as small rotations, scaling and cropping in 
% addition to horizontal reflection.
aug = imageDataAugmenter('RandXReflection', true, ...
                         'RandRotation', [-15 15], ...
                         'RandScale', [0.5 2]);

augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
                                       'DataAugmentation', aug, ...
                                       'OutputSizeMode', 'randcrop');

augImdsValidation = augmentedImageDatastore(imageSize, imdsValidation);

%% ConvNet with more layers
% Task 5 (optional)
% Add a convolutional layer of size 9x9 with 64 filters, a batch
% normalization layer and ReLU activation layer in addition to another
% fully connected layer in the end
layers = [
    imageInputLayer([64 64 1], 'Name', 'input', 'Normalization', 'rescale-zero-one')
    convolution2dLayer(3, 8, 'Stride', 1, 'Padding', 'same', ...
                        'Weights', 0.01.*randn(3,3,1,8), 'Bias', zeros(1,1,8), ...
                        'Name', 'conv_1')
    batchNormalizationLayer('Name', 'BN_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

    convolution2dLayer(5, 16, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'BN_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')

    dropoutLayer(0.25, 'Name', 'dropout_1')

    convolution2dLayer(7, 32, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'BN_3')
    reluLayer('Name', 'relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')

    convolution2dLayer(9, 64, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_4')
    batchNormalizationLayer('Name', 'BN_4')
    reluLayer('Name', 'relu_4')

    dropoutLayer(0.5, 'Name', 'dropout_2')

    fullyConnectedLayer(15, 'Name', 'FC_2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

lgraph = layerGraph(layers);
analyzeNetwork(lgraph);

% Training options
miniBatchSize = 64;
valFrequency = floor(numel(augImdsTrain.Files)/(2*miniBatchSize));
options = trainingOptions('sgdm',...
                          'InitialLearnRate', 0.01, ...
                          'MaxEpochs', 30, ...
                          'MiniBatchSize', miniBatchSize, ...
                          'Shuffle', 'every-epoch', ...
                          'ValidationData', augImdsValidation, ...
                          'ValidationFrequency', valFrequency, ...
                          'ValidationPatience', 4, ...
                          'ExecutionEnvironment', 'parallel', ...
                          'Plots', 'training-progress');

% Train the net
net = trainNetwork(augImdsTrain, layers, options);

% Evaluate performance on test set
testDatasetPath  = fullfile('test');

imdsTest = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
augImdsTest = augmentedImageDatastore(imageSize, imdsTest);

% Apply the network to the test set
YPredicted = classify(net, augImdsTest);
YTest = imdsTest.Labels;

% Overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
plotconfusion(YTest, YPredicted)
