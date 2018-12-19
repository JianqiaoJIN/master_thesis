require("zoo")
require("vars")

# --- Conditional Granger causality analyzer -------#
# input:
#   S: system (type: data frame)
#   d: directory name
#
# result:
#   A_: estimated adjacent matrix
#   W:  strength of causal influence 

conditionalGC_analyzer <- function(S, d){
  
  N <- dim(S)[1]
  D <- dim(S)[2]
  A_ <- matrix(0,D,D) # estimated connection
  W <- matrix(0,D,D) #causal influence 
  
  # normalize 
  S <- as.data.frame(scale(S, center = TRUE, scale = TRUE)) 
  
  # dataframe -> matrix
  S <- do.call("merge", lapply(1:D, function(k) as.zoo(S[k])))
  
  # optimal lag selection 
  L <- VARselect(S,lag.max = 5)$selection[1] #AIC criteria
  
  # prepare X_train data
  name_ <- paste(names(S)[1],1:L,sep='_')
  X_train <- do.call("merge", lapply(1:L, function(k) lag(S[,1], -k)))
  for (i in 2:D){
    name_ <- c(name_, paste(names(S)[i],1:L,sep='_'))
    Z_i_lag <- do.call("merge", lapply(1:L, function(k) lag(S[,i], -k)))
    X_train <- merge(X_train, Z_i_lag)
  }
  
  X_train <- na.omit(X_train)
  colnames(X_train) <- name_
  
  
  for ( i in 1:D){
    
    target_Y <- S[(L+1):N, i]
    
    for (j in 1:D){
      if (j == i){
        next
      }
      
      candidate_X <- X_train[,((j-1)*L+1):(j*L)]
      condition_X <- X_train[,-(((j-1)*L+1):(j*L))]
      
      # OLS regression 
      U_model <- lm(formula = target_Y ~ condition_X + candidate_X) # unstricted model
      R_model <- lm(formula = target_Y ~ condition_X) # restricted model
      
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
  
  write.csv(A_, row.names = FALSE, file = paste("results/",d,"/conditionalGC/A_est_",D,"_",N,".csv", sep=""))
  write.csv(W, row.names = FALSE, file = paste("results/",d,"/conditionalGC/W_est_",D,"_",N,".csv", sep=""))

}