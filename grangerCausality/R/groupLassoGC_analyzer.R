require("gglasso")
require("zoo")
require("vars")

# --- Group Lasso Granger causality analyzer -------#
# input:
#   S: system (type: data frame)
#   d: directory name
#
# result:
#   A_: estimated adjacent matrix
#   W:  strength of causal influence 


groupLassoGC_analyzer <- function(S, d){
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
  X_train <- as.matrix(X_train)
  
  for (i in 1:D){
    target_Y <- as.matrix(S[(L+1):N, i])
    
    # OLS regression with group Lasso penalty
    group <- rep(1:(dim(X_train)[2]/L), each = L)
    cv <- cv.gglasso(x=X_train, y=target_Y, group = group, loss = "ls", pred.loss = "L1",lambda.factor=0.05, nfolds=5)
    pre <- coef(cv$gglasso.fit, s = cv$lambda.1se)
    
    # select coefficients whose value is not equal to 0
    pre <- pre[-1] # remove the intercept 
    names(pre) <- name_
    pre <- pre[pre != 0] 
    if (length(pre) == 0){
      next
    }
    
    # get the group index 
    pre_index <- do.call("c",lapply(1:length(pre), function(k) strsplit(names(pre)[k], split = '_')) )
    pre_index <- do.call("c", lapply(1:length(pre_index), function(k) as.integer(pre_index[[k]][2])))
    names(pre) <- pre_index
    
    causes <- pre[names(pre) != i]
    if (length(causes) == 0){
      next
    }
    causes_name <- unique(names(causes))
    
    # record the connectivity and causal influence 
    for (j in 1:length(causes_name)){
      A_[i, as.integer(causes_name[j])] <- 1
      W[i, as.integer(causes_name[j])] <- sum(abs(causes[names(causes) == causes_name[j]]))
    }
  }
  

  # save the result
  A_ <- as.data.frame(A_)
  colnames(A_) <- colnames(S)
  W <- as.data.frame(W)
  colnames(W) <- colnames(S)
  
  write.csv(A_, row.names = FALSE, file = paste("results/",d,"/groupLassoGC/A_est_",D,"_",N,".csv", sep=""))
  write.csv(W, row.names = FALSE, file = paste("results/",d,"/groupLassoGC/W_est_",D,"_",N,".csv", sep=""))

}
