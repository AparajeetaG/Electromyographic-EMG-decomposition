
  
    function originalEMGDecomposition(filename, fs)
    % EMG Decomposition Project

% Load EMG data from a CSV file

filename = 'EMG_example_20s_2000Hz-2023.csv'; % Change this for different files
data = csvread(filename, 1, 0); % Assuming the first row is data 

% Randomly select a channel (column)
num_channels = size(data, 2);
selected_channel = randi(num_channels); % Randomly select a channel
emg_signal = data(:, selected_channel);

% Sample rate (fs) can be set manually for different files
fs = 2000; % Sample rate in Hz (change this based on the file used)

% --- Filter the EMG Signal ---
% Bandpass filter to remove noise
[b, a] = butter(4, [20, 450] / (fs / 2), 'bandpass'); % 4th order bandpass filter
filtered_emg = filtfilt(b, a, emg_signal);

% --- Detect Spikes in the EMG Signal ---
% Spike detection using a threshold
threshold = mean(filtered_emg) + 3 * std(filtered_emg); 
spikes = filtered_emg > threshold;

% --- Align Spikes for Analysis ---
% Finding indices of spikes and defining a window around each spike
spike_indices = find(spikes);
window = 40; % Define the window size
aligned_spikes = arrayfun(@(x) filtered_emg(max(1, x-window):min(length(filtered_emg), x+window)), spike_indices, 'UniformOutput', false);

% --- Extract Features from Aligned Spikes ---
% Converting cell array to matrix for PCA
spike_matrix = cell2mat(cellfun(@(x) x', aligned_spikes, 'UniformOutput', false));
% Feature extraction using PCA
[coeff, score, ~, explained] = pca(spike_matrix);
% Additional feature extraction methods
additional_features = cellfun(@(x) [max(x) - min(x), trapz(abs(x))], aligned_spikes, 'UniformOutput', false);
additional_features = cell2mat(additional_features);
% Combine PCA features with additional features
features = [score(:, 1:2), additional_features]; % Using first two PCA components

% K-means Clustering
num_clusters = 3; % Number of clusters
[idx, centroids_kmeans] = kmeans(features, num_clusters);

% Gaussian Mixture Model (GMM) Clustering
gmm = fitgmdist(features, num_clusters);
idx_gmm = cluster(gmm, features);

% Visualizing Clusters (optional)
figure;
subplot(1, 2, 1);
scatter(features(:, 1), features(:, 2), 100, idx, 'filled');
title('K-Means Clustering');
subplot(1, 2, 2);
scatter(features(:, 1), features(:, 2), 100, idx_gmm, 'filled');
title('GMM Clustering');

% Classification based on clustering results
% For K-Means
classified_spikes_kmeans = idx;



% 1. Input File (Data Loading)
figure;
plot(emg_signal); % Plot the entire signal
title('1. Raw EMG Signal');

% 2. Filter Signal
figure;
plot(filtered_emg);
title('2. Filtered EMG Signal');

% 3. Detect Spikes
figure;
plot(filtered_emg);
hold on;
plot(find(spikes), filtered_emg(spikes), 'r*');
title('3. Spike Detection');
hold off;

% 4. Align Spikes
figure;
for i = 1:min(length(aligned_spikes), 10)
    plot(aligned_spikes{i});
    hold on;
end
title('4. Aligned Spikes');

% 5. Extract Features (Assuming PCA and other methods used)
figure;
scatter(features(:, 1), features(:, 2)); % Assuming first two columns are principal components
title('5. Feature Scatter Plot');

% 6. Cluster Spikes (Assuming k-means clustering)
figure;
scatter(features(:, 1), features(:, 2), 100, idx, 'filled');
title('6. K-Means Clustering Results');



% Classification based on K-Means clustering results
classified_spikes_kmeans = idx;
figure;
scatter(features(:, 1), features(:, 2), 100, classified_spikes_kmeans, 'filled');
title('Classification Results - K-Means');

% Classification based on GMM clustering results
classified_spikes_gmm = idx_gmm;
figure;
scatter(features(:, 1), features(:, 2), 100, classified_spikes_gmm, 'filled');
title('Classification Results - GMM');

% Analyze Clustering Results
% For K-Means Clustering
analyze_clusters('K-Means', features, classified_spikes_kmeans, centroids_kmeans);

% For GMM Clustering
analyze_clusters('GMM', features, classified_spikes_gmm, gmm.mu);


  % --- Time-Frequency Analysis using STFT ---
    % Perform STFT on the filtered EMG signal
    stft_window = hamming(128); % Window for STFT. Adjust size as needed.
    stft_overlap = 64; % Overlap between windows. Adjust as needed.
    [s, f, t] = spectrogram(filtered_emg, stft_window, stft_overlap, [], fs);
    figure;
    surf(t, f, 10*log10(abs(s)), 'EdgeColor', 'none');
    axis tight;
    view(0, 90);
    xlabel('Time (Seconds)');
    ylabel('Frequency (Hz)');
    title('Time-Frequency Analysis of Filtered EMG Signal');


% Calculate silhouette scores for the clustering results
%silhouette_scores = silhouette(features, idx, 'Euclidean');

% Visualize the silhouette scores
%figure;
%silhouette(features, idx, 'Euclidean');
%xlabel('Silhouette Value');
%ylabel('Cluster');
%title('Silhouette Scores for Clustering Quality Assessment');



% Calculate silhouette scores for K-means clustering results
silhouette_scores_kmeans = silhouette(features, idx, 'Euclidean');

% Visualize the silhouette scores for K-means
figure;
silhouette(features, idx, 'Euclidean');
title('Silhouette Scores for K-means Clustering');

% Calculate silhouette scores for GMM clustering results
silhouette_scores_gmm = silhouette(features, idx_gmm, 'Euclidean');

% Visualize the silhouette scores for GMM
figure;
silhouette(features, idx_gmm, 'Euclidean');
title('Silhouette Scores for GMM Clustering');

% Function to Analyze and Visualize Clustering Results
function analyze_clusters(method_name, features, cluster_indices, centroids)
    fprintf('\nAnalyzing Clusters using %s\n', method_name);
    num_clusters = size(centroids, 1);

    % Intra- and Inter-cluster distances
    intra_cluster_distances = zeros(num_clusters, 1);
    inter_cluster_distances = zeros(num_clusters, num_clusters);

    for i = 1:num_clusters
        intra_cluster_distances(i) = mean(pdist2(features(cluster_indices == i, :), centroids(i, :)));
        for j = 1:num_clusters
            inter_cluster_distances(i, j) = pdist2(centroids(i, :), centroids(j, :));
        end
    end

    % Visualizing distances
    figure;
    subplot(1, 2, 1);
    bar(intra_cluster_distances);
    title([method_name ' - Intra-Cluster Distances']);
    xlabel('Cluster'); ylabel('Average Distance');

    subplot(1, 2, 2);
    imagesc(inter_cluster_distances);
    colorbar;
    title([method_name ' - Inter-Cluster Distances']);
    xlabel('Cluster'); ylabel('Cluster');
end

  
    end