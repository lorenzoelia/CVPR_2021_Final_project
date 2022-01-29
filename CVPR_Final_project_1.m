clear
close all
clc
%% The dataset
% The project consists in an implementation of a an image classifier  based
% on convolutional neural networks. The dataset used is the 'fifteen scene
% categories' also used in Lazebnik et al., 2006.

% Import the dataset, already splitted in train and test sets, in an
% imageDatastore
trainDatasetPath = fullfile('train');
imds = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Show all the labels with relative amounts
labelCount = countEachLabel(imds);
disp(labelCount)

% Automatic anisotropic resizing to 64x64 and normalization rescale in 0-1
imageSize = [64 64];
divideBy = 255;
imds.ReadFcn = @(x)double(imresize(imread(x), imageSize))/divideBy;

% Split the provided training set in 85% for actual training set and 15%
% to be used as validation set
quotaForEachLabel = 0.85;

%% Network design

layers = [
    imageInputLayer([64 64 1], 'Name', 'input')

    convolution2dLayer(3, 8, 'Stride', 1, 'Padding', 'same', ...
                        'Weights', 0.01.*randn(3,3,1,8), 'Bias', zeros(1,1,8), ...
                        'Name', 'conv_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')

    convolution2dLayer(3, 16, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')

    convolution2dLayer(3, 32, 'Stride', 1, 'Padding', 'same', 'Name', 'conv_3')
    reluLayer('Name', 'relu_3')

    fullyConnectedLayer(15, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

lgraph = layerGraph(layers);
analyzeNetwork(lgraph)
%% Best initial learn rate
% This section exploits the steps to find the possible best values for the
% initial learning rate and validation patience as stopping criterion.

% First of all, we try to find the best initial learning rate. Consider a
% pre-fixed number for epochs so we can notice any change in fixed
% time. Possibily, the best rate will be the one with the lower average
% final validation loss. We carry out the results by a psuedo cross
% validation approach, in fact through a repeated validation set approach.
initialLearnRates = [1 0.1 0.01 0.001 0.0001];

% Check if the file containing the average final validation losses for
% choosing the best initial learn rate exists. If not, proceed to compute
% the best initial learn rate
if isfile('workspace\initialLearnRate.mat')
    load('workspace\initialLearnRate.mat', 'learnRateLoss')
else
    cvIterations = 10;
    learnRateLoss = zeros(cvIterations, length(initialLearnRates));
    
    for i = 1:length(initialLearnRates)
        for j = 1:cvIterations
            % Split the data
            [imdsTrain, imdsValidation] = splitEachLabel(imds, quotaForEachLabel, 'randomize');
    
            % Training options
            miniBatchSize = 32;
            valFrequency = floor(numel(imdsTrain.Files)/(2*miniBatchSize)); % Two validations per epoch
            options = trainingOptions('sgdm', ...
                                      'InitialLearnRate', initialLearnRates(i), ...
                                      'MaxEpochs', 10, ...
                                      'MiniBatchSize', miniBatchSize, ...
                                      'ValidationData', imdsValidation, ...
                                      'ValidationFrequency', valFrequency, ...
                                      'ExecutionEnvironment', 'parallel', ...
                                      'Plots', 'training-progress');
    
            % Train the net
            [net, info] = trainNetwork(imdsTrain, layers, options);
            
            % Save the final validation loss
            learnRateLoss(j,i) = info.FinalValidationLoss;
        end
    end
    
    % Save the variable 'learnRateLoss' from the workspace in a file, since
    % it's quite infeasible to compute it each time
    save('workspace\initialLearnRate.mat', 'learnRateLoss');
end

% Compute mean and std deviation of each observation (#cvIterations) for
% every initial learning rate
mu = mean(learnRateLoss);
sigma = std(learnRateLoss);

% Plot the average final validation losses for each initial learn rate with
% error bars on a semi-logarithmic scale
figure
hAx = axes;
hAx.XScale = 'log';
hAx.XDir = 'reverse';
xlim([min(initialLearnRates), max(initialLearnRates)]);
hold all
errorbar(initialLearnRates, mu, sigma, '-o')
grid on
grid minor
title('Average final validation losses for different initial learn rates')
xlabel('Initial learn rate')
ylabel('Average final validation loss')

% Choose the best initial learning rate, namely the one that minimizes the
% average validation loss and display it in command window
[~, i] = min(mu);
bestInitialLearnRate = initialLearnRates(i);
disp(['Best initial learn rate: ', num2str(bestInitialLearnRate)])

%% Best patience as stopping criterion
% Perform in a similar manner to the best initial learn rate the process to
% pick the best value for the validation patience. This time use the best
% initial learn rate previously obtained
patience = [1 2 3 4 5];

if isfile('workspace\validationPatience.mat')
    load('workspace\validationPatience.mat', 'patienceLoss')
else
    cvIterations = 10;
    patienceLoss = zeros(cvIterations, length(patience));

    for i = 1:length(patience)
        for j = 1:cvIterations
            % Split the data
            [imdsTrain, imdsValidation] = splitEachLabel(imds, quotaForEachLabel, 'randomize');
    
            % Training options
            miniBatchSize = 32;
            valFrequency = floor(numel(imdsTrain.Files)/(2*miniBatchSize));
            options = trainingOptions('sgdm', ...
                                      'InitialLearnRate', bestInitialLearnRate, ...
                                      'MaxEpochs', 30, ...
                                      'MiniBatchSize', miniBatchSize, ...
                                      'ValidationData', imdsValidation, ...
                                      'ValidationFrequency', valFrequency, ...
                                      'ValidationPatience', patience(i), ...
                                      'ExecutionEnvironment', 'parallel', ...
                                      'Plots', 'training-progress');
    
            % Train the net
            [net, info] = trainNetwork(imdsTrain, layers, options);
            
            % Save the final validation loss
            patienceLoss(j,i) = info.FinalValidationLoss;
        end
    end

    % Save the variable 'patienceLoss' from the workspace in a file, since
    % it's quite infeasible to compute it each time
    save('workspace\validationPatience.mat', 'patienceLoss');
end

% Compute mean and std deviation of each observation (#cvIterations) for
% every initial learning rate
mu = mean(patienceLoss);
sigma = std(patienceLoss);

% Plot the average final validation losses for each patience with
% error bars
figure
xlim([min(patience), max(patience)]);
hold all
errorbar(patience, mu, sigma, '-o')
grid on
grid minor
title('Average final validation losses for different patiences')
xlabel('Patience')
ylabel('Average final validation loss')

% Choose the best validation patience, namely the one that minimizes the
% average validation loss and display it in command window
[minmu, i] = min(mu);
bestPatience = patience(i);
disp(['Best patience: ', num2str(bestPatience)])

%% Evaluate performance on test set
% Once we have obtained the best hyperparameters for initial learn rate and
% validation patience, train a final convnet with the best params.

% Split the data
[imdsTrain, imdsValidation] = splitEachLabel(imds, quotaForEachLabel, 'randomize');

% Training options
miniBatchSize = 32;
valFrequency = floor(numel(imdsTrain.Files)/(2*miniBatchSize));
bestOptions = trainingOptions('sgdm', ...
                          'InitialLearnRate', bestInitialLearnRate, ...
                          'MaxEpochs', 30, ...
                          'MiniBatchSize', miniBatchSize, ...
                          'ValidationData', imdsValidation, ...
                          'ValidationFrequency', valFrequency, ...
                          'ValidationPatience', bestPatience, ...
                          'ExecutionEnvironment', 'parallel', ...
                          'Plots', 'training-progress');

% Train the net
bestNet = trainNetwork(imdsTrain, layers, bestOptions);

% Test set
testDatasetPath = fullfile('test');

imdsTest = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest.ReadFcn = @(x)double(imresize(imread(x), imageSize))/divideBy;

% Apply the network to the test set
YPredicted = classify(bestNet, imdsTest);
YTest = imdsTest.Labels;

% Overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
plotconfusion(YTest, YPredicted)

