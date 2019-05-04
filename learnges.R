library(pcalg)

d = data.frame()
for (filepath in c("~/keras-rl/saves2/", "~/keras-rl/saves/")) {
  for (file in list.files(filepath)){
    d1 = read.csv(paste(filepath,file, sep=""))
    counter = 0
    for (i in colnames(d1)){
      if (grepl("obs", i, fixed = TRUE))
        counter = counter+1
    }
    dtmp = d1[2:nrow(d1),1:counter]
    names(dtmp) = paste(names(dtmp),"_n",sep="")
    
  
    d1 = cbind(d1[1:nrow(d1)-1,],dtmp)
  
    d = rbind(d,d1)
  }
}

d1 = subset(d,do_bool==0)
p = pc(list(C=cor(d1), n=nrow(d1)),pcalg::gaussCItest,alpha=.05,colnames(d1)); plot(p, main="PC OUTPUT")

indexes = as.integer(d$do_bool==1)+1
targets = list(integer(0), as.integer(c(9,11)))
score = new("GaussL0penIntScore", d, targets, indexes)
ti.lb <- c(sapply(seq_along(targets), function(i) match(i, indexes)),
           length(indexes) + 1)
p = gies(score); as(p$essgraph,"matrix")+0; plot(p$essgraph)
