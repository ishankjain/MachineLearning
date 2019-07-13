X=cell2mat(featureTrain(1));
[U, S] = pca(X);
K = 480;
Z = projectData(X, U, K);