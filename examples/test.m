dim =3;
sigma_0 = .0001;
cluster_num = 2;
cluster_size = 1E3;


X_clusters = [];
col_clusters = repmat(1:cluster_num,cluster_size,1);
col_clusters = col_clusters(:);

for i =1:cluster_num,
    X_clusters = [X_clusters; mvnrnd(rand(dim,1)', diag(sigma_0*ones(dim,1)), cluster_size)];
end

N_clusters = size(X_clusters,1)


figure(19988)
scatter3(X_clusters(:,1), X_clusters(:,2), X_clusters(:,3),10, col_clusters, 'filled');
title('Original Data')



 %%

clear opts
opts.stop_lying_iter = 2E2;
opts.compexagcoef = 12;
opts.no_dims =   2;
opts.max_iter = 400;

opts.n_trees = 500;
opts.start_late_exag_iter = 2E3;
opts.late_exag_coeff = 2;
opts.no_momentum_during_exag =0;
opts.perplexity = 30;
%opts.nbody_algo = '';
opts.search_k = 3*opts.perplexity*opts.n_trees*5;
opts.theta = 0.5;

cluster_firstphase = fast_tsne(X_clusters,opts);

figure(433)
scatter(cluster_firstphase(:,1), cluster_firstphase(:,2),  10, col_clusters, 'filled');
title('After t-SNE')
colormap(lines(cluster_num))

%%
tic
cluster_firstphase = fast_tsne_original(X_clusters);
toc
%%

clear opts
opts.stop_lying_iter = 200;
opts.compexagcoef = 12;
opts.no_dims =   1;
opts.max_iter = 1E3;

opts.n_trees = 50;
opts.start_late_exag_iter = 800;
opts.late_exag_coeff = 2;
opts.no_momentum_during_exag =0;
opts.perplexity = 90;
%opts.nbody_algo = 'fmm';
opts.search_k = 3*opts.perplexity*opts.n_trees;
opts.theta = 0.5;
tic
cluster_firstphase = fast_tsne(X_clusters,opts);
toc


 figure(2)
scatter(cluster_firstphase(:,1), rand(N_clusters,1),  10, col_clusters, 'filled');
title('After t-SNE')
colormap(lines(cluster_num))
