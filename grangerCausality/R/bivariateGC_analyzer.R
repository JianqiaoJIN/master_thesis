require("zoo")
require("vars")

# --- Biavariate Granger causality analyzer -------#
# input:
#   S: system (type: data frame)
#   d: directory name
#
# result:
#   A_: estimated adjacent matrix
#   W:  strength of causal influence 

bivariateGC_analyzer <- function(S, d){
  N <- dim(S)[1]
  D <- dim(S)[2]
  A_ <- matrix(0,D,D) # estimated connection
  W <- matrix(0,D,D) #causal influence 
  
  # normalize 
  S <- as.data.frame(scale(S, center = TRUE, scale = TRUE)) 
  
  # dataframe -> matrix
  S <- do.call("merge", lapply(1:D, function(k) as.zoo(S[k])))
  
  for ( i in 1:D){
    
    target <- S[,i]
    
    for (j in 1:D){
      if (j == i){
        next
      }
      
      candidate <- S[,j]
      
      # optimal lag selection 
      bi_system <- merge(target, candidate)
      L <- VARselect(bi_system, lag.max = 5)$selection[1] #AIC criteria
      
      # prepare train data
      candidate_X <- do.call("merge", lapply(1:L, function(k) lag(candidate, -k)))
      target_X <- do.call("merge", lapply(1:L, function(k) lag(target, -k)))
      
      all <- merge(target, candidate_X, target_X)
      colnames(all) <- c("target", paste("candiate", 1:L, sep = "_"), paste("target", 1:L, sep = "_"))
      all <- na.omit(all)
      target_Y <- as.vector(all[,1])
      candidate_X <- as.matrix(all[,(1:L+1)])
      target_X <- as.matrix(all[,(1:L + 1 + L)])
      
      # train data -> OLS regression
      U_model <- lm(formula = target_Y ~ target_X + candidate_X) # unstricted model
      R_model <- lm(formula = target_Y ~ target_X) # restricted model
      
      # F-test
      p <- anova(R_model,U_model)$`Pr(>F)`[2] #F-test and Pr(>F)
      
      if(p < 0.05){
        A_[i,j] <- 1

        sigma_U <- summary(U_model)$sigma
        sigma_R <- summary(R_model)$sigma
        W[i,j] <- log((sigma_R/sigma_U)^2)
      }
      
    }
  }
  
  # save the result
  A_ <- as.data.frame(A_)
  colnames(A_) <- colnames(S)
  W <- as.data.frame(W)
  colnames(W) <- colnames(S)

  write.csv(A_, row.names = FALSE, file = paste("results/",d,"/bivariateGC/A_est_",D, "_", N,".csv", sep=""))
  write.csv(W, row.names = FALSE, file = paste("results/",d,"/bivariateGC/W_est_",D, "_", N,".csv", sep=""))
}