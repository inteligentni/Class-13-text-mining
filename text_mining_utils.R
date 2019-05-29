# a utility function for printing a sample of tweets from the given corpus
print.tweets <- function(corpus, from = 1, to = 10) {
  for(i in from : to) {
    cat( paste("[[", i, "]] ", sep = ""))
    writeLines(text = strwrap(corpus[[i]]$content, width = 80))
    cat("\n")
  }
}


# a utility function for computing evaluation measures for a binary classifier
# cm is a confusion matrix
compute.eval.measures <- function(cm) {
  TP <- cm[1,1]
  FP <- cm[2,1]
  FN <- cm[1,2]
  acc <- sum(diag(cm))/sum(cm)
  prec <- TP / (TP + FP) 
  rec <- TP / (TP + FN)
  f1 <- (2*prec*rec)/(prec+rec)
  # return the performance metrics
  c(accuracy = round(acc, digits = 4), precision = round(prec, 4), 
    recall = round(rec, 4), F1= round(f1, 4))  
}


# a utility function for printing a sample of items from each cluster
# - clustering is a named vector of cluster assignments
# - n.items is the number of items to print from each cluster
print.clusters <- function(clustering, n.items = 5) {
  clust.dist <- as.matrix(table(clustering))
  k <- length(unique(clustering))
  for(i in 1:k) {
    cat(paste("\nCLUSTER", i, ":\n"))
    cl.size <- clust.dist[i]
    # if the number of items in the cluster is <=10, print them all
    if(cl.size <= n.items) 
      print(names(clustering)[clustering==i])
    # otherwise, take a random sample of 10 items
    else { indices <- sample(x = 1:cl.size, size = n.items, replace = F)
    clust.items <- names(clustering)[clustering==i]
    print(clust.items[indices]) }
  }
}

# an auxiliary f. for reading text from a file
read.text = function(pathname) {
  return (paste(readLines(pathname), collapse="\n"))
}
