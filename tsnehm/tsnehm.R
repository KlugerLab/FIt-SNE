#' Compute t-SNE Heatmap
#'
#' Alpha version of t-SNE heatmaps codes
#'
#' @param A_norm Full expression matrix genes (rows) vs. cells (columns) 
#' @param tsne_embedding2 1D t-SNE embedding
require(gplots)
library(pdist)
library(plyr)
library(RColorBrewer)
library(heatmaply)


hmtsne <- function(A_norm, goi, tsne_embedding2,  enrich=0){


	#Identify missing genes
	notinrows <- !(goi%in% rownames(A_norm));
	print(sprintf("The following rows are not in the matrix: %s", paste(goi[notinrows])))

	goi <- goi[!notinrows]
	if (enrich ==0 ) {
		goidf <- data.frame(x=tsne_embedding2, t(A_norm[goi,]))
	}else{
		goidf <- data.frame(x=tsne_embedding2, t(A_norm))
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

	rownames(bin_counts_s) <- sprintf("bin:%s", rownames(bin_counts_s))
	if (enrich>0){
		gene_colors <- rep(0, length(enriched_genes));
	}else{
		gene_colors <-c();
	}
	gene_colors[1:length(goi)] <- 1

	my_palette <- colorRampPalette(c("white", "red"))(n = 1000)
	row_color_palette <- colorRampPalette(c("white", "blue"))
	col_color_palette <- colorRampPalette(brewer.pal(n = max(group_labels+1,na.rm=TRUE),"Spectral"))
	heatmaply(t(as.matrix(signif(bin_counts_s,2))) , row_side_colors=gene_colors,
		  row_side_palette=row_color_palette, showticklabels = c(FALSE,TRUE),
		  col=my_palette,  dendrogram='row', titleX =FALSE, RowV=FALSE,
		  show_legend=FALSE,hide_colorbar = TRUE,fontsize_row = 5,margins = c(70,50,NA,0),
		  )
}
