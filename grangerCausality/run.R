source('R/bivariateGC_analyzer.R')
source('R/conditionalGC_analyzer.R')
source('R/groupLassoGC_analyzer.R')

oneFileProcess <- function(d, N, D){
  file_name <- paste("data/",d,"/system_", D, "_", N,".csv",sep="")
  S <- read.csv(file_name)
  
  print (paste("***---", file_name, "---***"))
  print ("Bivariate Granger causality analysis")
  print ("analyzing.....")

  bivariateGC_analyzer(S,d)
  
  print ("Conditional Granger causality analysis")
  print ("analyzing.....")
  conditionalGC_analyzer(S,d)
  
  print ("Group Lasso Granger causality analysis")
  print ("analyzing.....")
  groupLassoGC_analyzer(S,d)
  
  print (paste("File",file_name, "processing complete."))
}


dirs = c('var_system','henon_system')
D_set = c('5', '30')
N_set = c('1000')

for (d in dirs){
  for (D in D_set){
    for (N in N_set){
      oneFileProcess(d, N, D)
    }
  }
}



