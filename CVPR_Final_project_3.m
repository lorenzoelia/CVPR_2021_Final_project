clear
close all
clc
%% The dataset
trainDatasetPath = fullfile('train');
imds = imageDatastore(trainDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% AlexNet accepts as inputs RGB images, thus we 'trick' the net by giving
% in input a grayscale image for each color channel
imds.ReadFcn = @(x)cat(3, imread(x), imread(x), imread(x));

% Split the provided training set in 85% for actual training set and 15%
% to be used as validation set
quotaForEachLabel = 0.85;
[imdsTrain, imdsValidation] = splitEachLabel(imds, quotaForEachLabel, 'randomize');

%% Transfer learning
% Load pretrained network from the Computer Vision Toolbox
net = alexnet;
analyzeNetwork(net);

% Get the input size used by AlexNet
inputSize = net.Layers(1).InputSize;

% Replace final layers. Save all the original layers up to the last fully
% connected layers (excluded)
layersTransfer = net.Layers(1:end-3);

% Freeze the weights of AlexNet
for ii = 1:size(layersTransfer, 1)
    props = properties(layersTransfer(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layersTransfer(ii).(propName) = 0;
        end 
    end
end

% Append custom layers givin a heavier learning rate for weights and biases
% to speed up the evolution of just our new layers
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 10, ...
                        'BiasLearnRateFactor', 10, 'Name', 'myFC')
    softmaxLayer('Name', 'mySoftmax')
    classificationLayer('Name', 'myOutput')
];

lgraph = layerGraph(layers);
analyzeNetwork(lgraph);

% Train the network
aug = imageDataAugmenter('RandXReflection', true);
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', aug);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

miniBatchSize = 64;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
                        'MiniBatchSize', miniBatchSize, ...
                        'MaxEpochs', 10, ...
                        'InitialLearnRate', 1e-3, ...
                        'Shuffle','every-epoch', ...
                        'ValidationData', augimdsValidation, ...
                        'ValidationFrequency', valFrequency, ...
                        'ExecutionEnvironment', 'parallel', ...
                        'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain, layers, options);

% Evaluation on test set
% Test set
testDatasetPath  = fullfile('test');

imdsTest = imageDatastore(testDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imdsTest.ReadFcn = @(x)cat(3, imread(x), imread(x), imread(x));
augImdsTest = augmentedImageDatastore(inputSize, imdsTest);

% Apply the network to the test set
YPredicted = classify(netTransfer, augImdsTest);
YTest = imdsTest.Labels;

% Overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);
disp(['Test accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
plotconfusion(YTest, YPredicted)

%% Extract activation maps as features to train a multiclass 1vsAll SVM
% Extract the activation map from the last convolutional layer
layer = 'myFC';
featuresTrain = activations(netTransfer, augimdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(netTransfer, augImdsTest, layer, 'OutputAs', 'rows');

whos featuresTrain

% One-against-all approach
classes = unique(imds.Labels);
SVMs1vsAll = cell(1, numClasses);
YPred = cell(1, numClasses);

% Use the features extracted from the training images as predictor 
% variables and fit a multiclass support vector machine (SVM) with
% one-vs-all approach. Thus, train 15 classifiers
for i = 1:length(classes)
    trainingSet = copy(imdsTrain);
    trainingSet.Labels(find(trainingSet.Labels ~= classes(i))) = 'Other';

    YTrain = trainingSet.Labels;

    SVMs1vsAll{1,i} = fitcsvm(featuresTrain, YTrain, 'KernelFunction', 'linear');
    
    % Classify the test images using the trained SVM model using the features 
    % extracted from the test images.
    [YPred{1,i}, YPred{2,i}] = predict(SVMs1vsAll{1,i}, featuresTest);
end

% Employ a majority voting approach to determine a single output predicted
% vector. To sort the outcome of multiple different predicted classes for a
% single test observation, the final resulting label will be obtained by a
% look at the prediction score for every classifier and then taking the max

% generalYPred = cat(2, YPred{1,:});    % Uncomment this line to view all the results aligned column-wise
temp = cat(2, YPred{2,:});
generalScores = temp(:, 1:2:end);
clear temp

[~, i] = max(generalScores, [], 2);

% Final predicted values
finalYPred = classes(i);

% Calculate the classification accuracy on the test set. Accuracy is the 
% fraction of labels that the network predicts correctly.
YTest = imdsTest.Labels;
accuracy = mean(finalYPred == YTest);
disp(['Test accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
plotconfusion(YTest, finalYPred)

%% Multiclass SVM classification using non-linear kernel
% Task 6 (optional)
SVMs1vsAll = cell(1, numClasses);
YPred = cell(1, numClasses);

for i = 1:length(classes)
    trainingSet = copy(imdsTrain);
    trainingSet.Labels(find(trainingSet.Labels ~= classes(i))) = 'Other';

    YTrain = trainingSet.Labels;

    SVMs1vsAll{1,i} = fitcsvm(featuresTrain, YTrain, 'KernelFunction', 'gaussian'); 

    [YPred{1,i}, YPred{2,i}] = predict(SVMs1vsAll{1,i}, featuresTest);
end

% generalYPred = cat(2, YPred{1,:});
temp = cat(2, YPred{2,:});
generalScores = temp(:, 1:2:end);
clear temp

[~, i] = max(generalScores, [], 2);

finalYPred = classes(i);

% Accuracy
YTest = imdsTest.Labels;
accuracy = mean(finalYPred == YTest);
disp(['Test accuracy: ', num2str(accuracy)])

% Confusion matrix
figure
plotconfusion(YTest, finalYPred)
%% Multiclass SVM classification using the Error Correcting Output Code
% Task 7 (optional)

trainingSet = imdsTrain.Labels;
YTest = imdsTest.Labels;

classifier = fitcecoc(featuresTrain, trainingSet);

YPred = predict(classifier, featuresTest);

accuracy = mean(YPred == YTest);
disp(['Test accuracy: ', num2str(accuracy)])

%% Data augmentation with random cropping
% Task 4 (optional)
% If preferred, launch this section insted of the previous one to perform
% further augmentations such as small rotations, scaling and cropping in 
% addition to horizontal reflection.
aug = imageDataAugmenter('RandXReflection', true, ...
                         'RandRotation', [-15 15], ...
                         'RandScale', [0.5 1.5]);

augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
                                       'DataAugmentation', aug, ...
                                       'OutputSizeMode', 'randcrop');
augImdsValidation = augmentedImageDatastore(imageSize, imdsValidation);

