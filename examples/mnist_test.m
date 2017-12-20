addpath /Users/george/Downloads/mnistHelper/
subplot = @(m,n,p) subtightplot(m,n,p,[0.05, 0]);

%%
train_images_ = loadMNISTImages('/Users/george/Downloads/train-images-idx3-ubyte');
col_images_ = loadMNISTLabels('/Users/george/Downloads/train-labels-idx1-ubyte');


%%
train_images = train_images_(:,col_images_ ==1 | col_images_ ==2 | col_images_ ==3| col_images_ ==4 );
col_images = col_images_(col_images_ ==1 | col_images_ ==2 | col_images_ ==3| col_images_ ==4 );

%%
train_images = train_images_;
col_images = col_images_;
%%
addpath /Users/george/Research_Local/pcafast/src

rng(23422)
randsample = randi(size(train_images,2),1E4,1);
randsample = 1:size(train_images,2);
X_clusters = train_images(:,randsample)';
col_clusters = col_images(randsample);

[X_clusters_PCs, ~,~] = pcafast (X_clusters, 30);
figure(5555); scatter(X_clusters_PCs(:,3),X_clusters_PCs(:,4), 20, col_clusters, 'filled'); 


[N_clusters, dim ]= size(X_clusters);
plot_subset = randi(N_clusters,5E3,1);


%%

cd /Users/george/Research_Local/fmmtsne
clear opts
opts.stop_lying_iter = 200;
opts.compexagcoef = 12;
opts.no_dims =   2;
opts.max_iter = 1E3;
opts.n_trees = 50;
opts.start_late_exag_iter = 400;
opts.late_exag_coeff = 2;
opts.no_momentum_during_exag =0;
opts.perplexity = 30;
opts.nbody_algo = 'bh';
opts.search_k = 3*opts.perplexity*opts.n_trees;
tic
cluster_firstphase = fast_tsne(X_clusters_PCs,opts);
toc

figure(433)
scatter(cluster_firstphase(:,1), cluster_firstphase(:,2),  10, col_clusters, 'filled');
title('BH t-SNE')
colormap(jet)
%%
cd /Users/george/Research_Local/fmm_tsne/bh_tsne_original

 X_clusters_original = fast_tsne_original(X_clusters_PCs,2,dim);
figure(434)
scatter(X_clusters_original(:,1), X_clusters_original(:,2),  10, col_clusters, 'filled');
title('BH t-SNE')
colormap(lines(max(col_images)))