#!/usr/bin/env Rscript

#' R script that obtains BDgraph and GeneNet estimates to covariance selection problem 

# Load data
n_nodes <- 30
n_obs <- 20
gname <- 'circle'

data <- as.matrix(read.csv(paste('data/', gname, '_data_', n_nodes, '_', n_obs, '.csv', sep=''), header=F))
n_iter <- 1e4L

# BDgraph
library('BDgraph')
print("Running BDgraph...")
bdgraph.obj <- bdgraph(data, iter=n_iter, save=TRUE, g.prior = 0.5) # such that E(size) = 1 just like geometric prior

# Writing as csv  
write.table(bdgraph.obj$sample_graphs, file=paste('res/bd_graphs_',gname,'_',n_nodes,'_',n_obs,'.csv', sep=''), row.names=F, col.names=F)
write.table(bdgraph.obj$graph_weights, file=paste('res/bd_graphs_weight_',gname,'_',n_nodes,'_',n_obs,'.csv', sep=''), row.names=F, col.names=F)
write.table(plinks(bdgraph.obj), file=paste('res/bd_adjm_',gname,'_',n_nodes,'_',n_obs,'.csv', sep=''), row.names=F, col.names=F)

# GeneNet
library('GeneNet')
print("Running GeneNet...")
infer.pcor = ggm.estimate.pcor(data)
test.results <- network.test.edges(infer.pcor, plot=FALSE)

signif.lvl <- c(0.001, 0.01, 0.05)

to_adjm <- function(n, df){
  v1 = df$node1
  v2 = df$node2

  A = matrix(0, nrow=n, ncol=n)
  for (i in 1:length(v1)){
    A[v1[i], v2[i]] <- 1
    A[v2[i], v1[i]] <- 1
  }
  
  return(A)
}

for (i in 1:3){
  temp <- test.results[ test.results$pval < signif.lvl[i], ]
  A = to_adjm(n_nodes, temp)
  write.table(A, file=paste('res/gene_net_pval_',i,'_',gname,'_',n_nodes,'_',n_obs,'.csv', sep=''),
            row.names=F, col.names=F)
}

