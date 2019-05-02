library(pcalg)

d = data.frame()
for (file in c("~/keras-rl/dqn185.csv","~/keras-rl/saves/dqn99.csv",
               "~/keras-rl/saves/dqn98.csv","~/keras-rl/saves/dqn97.csv",
               "~/keras-rl/saves/dqn96.csv","~/keras-rl/saves/dqn95.csv","~/keras-rl/dqn149do.csv")){
  d1 = read.csv(file)
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

indexes = as.integer(d$do_bool==1)+1
targets = list(integer(0), as.integer(c(9,11)))
score = new("GaussL0penIntScore", d, targets, indexes)
ti.lb <- c(sapply(seq_along(targets), function(i) match(i, indexes)),
           length(indexes) + 1)
p = gies(score); as(p$essgraph,"matrix")+0; plot(p$essgraph)
