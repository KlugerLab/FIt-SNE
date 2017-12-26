#' Compute t-SNE Heatmap
#'
#' Experimental version of t-SNE heatmaps codes
#'
#' @param expression_matrix Full expression matrix genes (rows) vs. cells (columns) 
#' @param goi goi Genes of Interest
#' @param tsne_embedding 1D t-SNE embedding
#' @param cell_labels Labels for each cell. Typically these are the cluster assignments for each cell, as obtained by 
#'                    dbscan on the 2D t-SNE. This way, the columns of the heatmap will have a color assigned to each of them,
#'                    and they can be mapped to a corresponding location on the 2D t-SNE
#' @param enrich For every gene in goi, find enrich number of genes that are close to that gene in the distance induced by the 1D t-SNE
require(gplots)
library(pdist)
library(plyr)
library(RColorBrewer)
library(heatmaply)

tsnehm <- function(expression_matrix, goi, tsne_embedding, cell_labels, enrich=0){
  
  notinrows <- !(goi%in% rownames(expression_matrix));
  
  print(sprintf("The following rows are not in the matrix: %s", paste(goi[notinrows])))
  goi <- goi[!notinrows]
  if (enrich ==0 ) {
    goidf <- data.frame(x=tsne_embedding, t(expression_matrix[goi,]))
  }else{
    goidf <- data.frame(x=tsne_embedding, t(expression_matrix))
  }
  group <- split (goidf, cut(goidf$x,breaks = 100))
  bin_counts <- ldply(lapply(group,function(x) t(as.data.frame(colSums((x[,-1]))))), data.frame)
  bin_counts_s  <- sweep(bin_counts[,-1],2,colSums(bin_counts[,-1]), '/')
  if (enrich >0 ) {
    print("Now enriching")
    pdisttest <- pdist(t(as.matrix(bin_counts_s)),indices.A = goi, indices.B=1:ncol(bin_counts_s))
    sortedpdist <- t(apply(as.matrix(pdisttest), 1, order,decreasing=FALSE))
    enriched_genes <- unique(as.vector(sortedpdist[,1:(enrich +1)]))
    bin_counts_s <- bin_counts_s[,enriched_genes]
    
  }
  
  #assign a label to each column based on which of the cell_labels is the most common
  dbscan_tsne <- data.frame(x=tsne_embedding, y=cell_labels)
  dbscan_group <- split (dbscan_tsne, cut(dbscan_tsne$x,breaks = 100))
  Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  group_labels <-unlist(lapply(dbscan_group, function(x) Mode(x[,2])))
  rownames(bin_counts_s) <- sprintf("bin:%s, label: %d", rownames(bin_counts_s), group_labels)
  group_labels[is.na(group_labels)] <- NA
  if (enrich>0){
  gene_colors <- rep(0, length(enriched_genes));
  }else{
    gene_colors <-c();
  }
  gene_colors[1:length(goi)] <- 1
  
  my_palette <- colorRampPalette(c("white", "red"))(n = 1000)
  row_color_palette <- colorRampPalette(c("white", "blue"))
  col_color_palette <- colorRampPalette(brewer.pal(n = max(group_labels+1,na.rm=TRUE),"Spectral"))
  bin_count_s
  toplot <- t(as.matrix(bin_counts_s>0.05,2) + 0)
  heatmaply(toplot,col_side_colors = group_labels, row_side_colors=gene_colors, 
            row_side_palette=row_color_palette, showticklabels = c(FALSE,TRUE),  
                  col=my_palette,  dendrogram='row', titleX =FALSE, RowV=FALSE,
                 show_legend=FALSE,hide_colorbar = TRUE,fontsize_row = 5,margins = c(70,50,NA,0),
            )
}