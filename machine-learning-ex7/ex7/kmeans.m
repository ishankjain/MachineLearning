% load('ex7data2.mat');
function idx=kmeans(X)
    K = 2;
    max_iters = 20;
    initial_centroids = kMeansInitCentroids(X, K);
    [centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
end
