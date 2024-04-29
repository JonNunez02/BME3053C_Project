% BME3053C Final Project
% Author: Team 4
% Group Members:Jonathan Mendoza, Jon Nunez, Sebin George, Chandan Modem 
% Course: BME 3053C Computer Applications for BME
% Term: Spring 2024
% J. Crayton Pruitt Family Department of Biomedical Engineering
% University of Florida
% Email: mendoza.j@ufl.edu
% April 28, 2024

input_directory = "Add Pathway to input_Directory";

sampling_frequency = 360; 
gain = 200;
fragment_duration = 10; 
ecg_samples = [];
labels = {};

subfolders = {'SVTA', 'AFL', 'AFIB','NSR','APB'};

for i = 1:length(subfolders)
    current_subfolder = subfolders{i};
    wildcard_pattern = fullfile(input_directory, current_subfolder, '*.mat');
    file_list = dir(wildcard_pattern);
    for j = 1:length(file_list)
        file_name = fullfile(input_directory, current_subfolder, file_list(j).name);
        data = load(file_name);
        val = data.val; 
        num_fragments = floor(length(val) / (sampling_frequency * fragment_duration));
        start_indices = randperm(num_fragments, num_fragments);
        for k = 1:num_fragments
            start_index = (start_indices(k) - 1) * sampling_frequency * fragment_duration + 1;
            end_index = start_index + sampling_frequency * fragment_duration - 1;
            fragment = val(start_index:end_index);
            ecg_samples = [ecg_samples; fragment];
            labels = [labels; current_subfolder];
        end
    end
end

save('preprocessed_data.mat', 'ecg_samples', 'labels');
disp('Preprocessing completed.');
disp(['Total number of fragments: ', num2str(size(ecg_samples, 1))]);
disp('Data saved to preprocessed_data.mat');
snapshot_duration = 10; 
sampling_frequency = 360;
num_samples_per_snapshot = snapshot_duration * sampling_frequency;
snapshots = [];
snapshot_labels = {};

for i = 1:size(ecg_samples, 1)
    fragment = ecg_samples(i, :);
    num_snapshots = floor(length(fragment) / num_samples_per_snapshot);
    for j = 1:num_snapshots
        start_index = (j - 1) * num_samples_per_snapshot + 1;
        end_index = start_index + num_samples_per_snapshot - 1;
        snapshot = fragment(start_index:end_index);
        snapshots = [snapshots; snapshot];
        snapshot_labels = [snapshot_labels; labels(i)];
    end
end

disp('Snapshot generation completed.');
disp(['Total number of snapshots: ', num2str(size(snapshots, 1))]);

save('snapshots_data.mat', 'snapshots', 'snapshot_labels');
disp('Snapshots data saved to snapshots_data.mat');

load('snapshots_data.mat');  

sampling_frequency = 360; 
num_samples_per_snapshot = 3600; 
time_window = 10; 
output_directory = "Add the pathway to output_directory"; 
class_subfolders = {'SVTA', 'AFL','AFIB','NSR','APB'};

for i = 1:size(snapshots, 1)
    snapshot = snapshots(i, :);
    label = snapshot_labels{i};
    class_index = find(strcmp(label, class_subfolders));
    if isempty(class_index)
        error(['Invalid label: ', label]);
    end
  
    class_directory = fullfile(output_directory, 'scalograms', class_subfolders{class_index});
    
    [S, F, T] = spectrogram(snapshot, hann(512), 256, 512, sampling_frequency);
    S_db = 10*log10(abs(S));
    resized_spectrogram = imresize(S_db, [224, 224]);
    resized_spectrogram_rgb = ind2rgb(gray2ind(mat2gray(resized_spectrogram), 256), jet(256));
    spectrogram_filename = fullfile(class_directory, ['spectrogram_snapshot_', num2str(i), '.png']);
    imwrite(resized_spectrogram_rgb, spectrogram_filename);
end

disp('Spectrograms generation completed.');
disp(['Spectrograms saved to ', output_directory]);

training_directory = "Add the path to output_directory";
testing_directory = "Add the path to output_directory";
trainingDatastore = imageDatastore(fullfile(training_directory, 'Scalograms'),'IncludeSubfolders', true,'LabelSource', 'foldernames');
testingDatastore = imageDatastore(fullfile(testing_directory, 'Scalograms'),'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[trainData, valData] = splitEachLabel(trainingDatastore, 0.8, 'randomized');
disp("Number of training images: " + num2str(numel(trainData.Files)));
disp("Number of validation images: " + num2str(numel(valData.Files)));
disp("Number of testing images: " + num2str(numel(testingDatastore.Files)));
net = googlenet;
lgraph = layerGraph(net);
numClasses = numel(categories(TrainData.Labels));
newDropoutLayer = dropoutLayer(0.6, 'Name', 'new_Dropout');
newConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 5, 'BiasLearnRateFactor', 5);
newClassLayer = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, 'pool5-drop_7x7_s1', newDropoutLayer);
lgraph = replaceLayer(lgraph, 'output', newClassLayer);
lgraph = replaceLayer(lgraph, 'loss3-classifier', newConnectedLayer);
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'ValidationData', valData, ... 
    'ValidationFrequency', 140, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');


trainedNet = trainNetwork(TrainData, lgraph, options);
predictedLabels = classify(trainedNet, valData);
actualLabels = valData.Labels;

figure;
plotconfusion(actualLabels, predictedLabels);
title('Confusion Matrix on Validation Set');
saveas(gcf, 'confusion_matrix.png');
accuracy = mean(predictedLabels == actualLabels);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);
save('trained_net.mat', 'trainedNet');
test_directory = "Add the pathway to test_directory";

output_directory = fullfile(test_directory, 'testp');
if ~isfolder(output_directory)
    mkdir(output_directory);
end

file_list = dir(fullfile(test_directory, '*.mat'));
for i = 1:numel(file_list)
    file_name = fullfile(test_directory, file_list(i).name);
    data = load(file_name);
    ecg_signal = data.val; 
    
    [S, F, T] = spectrogram(ecg_signal, hann(512), 256, 512, 360);
    S_db = 10 * log10(abs(S));
    
    resized_spectrogram = imresize(S_db, [224, 224]);
    resized_spectrogram_rgb = ind2rgb(gray2ind(mat2gray(resized_spectrogram), 256), jet(256));
    [~, filename, ~] = fileparts(file_list(i).name);
    image_filename = fullfile(output_directory, [filename, '.png']);
    imwrite(resized_spectrogram_rgb, image_filename);
end

load('trained_net.mat');

test_directory2 = "Add the pathway to testp";
[imageData, imageLabels] = loadImageDataAndLabels(test_directory2, preprocessImage, extractLabel);
predictedLabels = classify(trainedNet, imageData);
accuracy = mean(predictedLabels == imageLabels);
disp(['Testing Accuracy: ', num2str(accuracy * 100), '%']);

predictedLabels = categorical(predictedLabels);
figure;
confusionchart(imageLabels, predictedLabels);
save('trained_net.mat', 'trainedNet');
[file, path] = uigetfile({'*.mat', 'MAT Files'}, 'Select a .mat ECG');
while isequal(file, 0)
    disp('No ECG file selected. Retry...');
    [file, path] = uigetfile({'*.mat', 'MAT Files'}, 'Select a .mat ECG');
end

load(fullfile(path, file));
ecg_signal = val;
snapshot_duration = 10;
sampling_frequency = 360;
num_samples_per_snapshot = snapshot_duration * sampling_frequency;
start_index = randi([1, length(ecg_signal) - num_samples_per_snapshot]);
end_index = start_index + num_samples_per_snapshot - 1;
snapshot = ecg_signal(start_index:end_index);
fb = cwtfilterbank('SignalLength', num_samples_per_snapshot, 'SamplingFrequency', sampling_frequency, 'VoicesPerOctave', 12);
[~, frq, cfs] = wt(fb, snapshot);
YPred = classify(trainedNet, abs(cfs));
disp("Predicted Class: " + YPred);

Classified_directory = "Add the pathway to Classified_Directory";
if ~isfolder(Classified_directory)
    mkdir(Classified_directory);
end

unique_labels = unique(predictedLabels);
for i = 1:numel(unique_labels)
    class_label = unique_labels(i);
    class_subfolder = fullfile(Classified_directory, char(class_label));
    if ~isfolder(class_subfolder)
        mkdir(class_subfolder);
    end
end

signal_filename = fullfile(class_subfolder, 'processed_signal.mat');
save(signal_filename, 'ecg_signal');
disp("Processed ECG signal saved to: " + signal_filename);