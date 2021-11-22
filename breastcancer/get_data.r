# Example from chapter 7 of Graphical Models with R by 
# Søren Højsgaard, David Edwards, Steffen Lauritzen, 2012 

library(gRbase)
library(gRapHD)

# Loading dataset
data(breastcancer)

# Run Chow-Liu Algorithm to find the minimal spanning forest
bF <- minForest(breastcancer)
nby <- neighbourhood(bF, orig=1001, rad=4)$v[,1]

# Subsetting into one community
bc.marg <- breastcancer[,nby]
bc.marg$code <- NULL
mbF <- minForest(bc.marg)

# Running stepwise addition of edges that increases the BIC measure
mbG <- stepw(model=mbF, data= bc.marg)

# Writing as csv
write.table(adjMat(mbF), 'data/mbF.csv', sep=',', row.names=F, col.names=F)
write.table(adjMat(mbG), 'data/mbG.csv', sep=',', row.names=F, col.names=F)
write.table(bc.marg, 'data/data.csv', sep=',', row.names=F, col.names=F)
