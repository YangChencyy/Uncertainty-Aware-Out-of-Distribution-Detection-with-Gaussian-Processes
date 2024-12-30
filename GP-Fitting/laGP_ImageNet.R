library(reticulate)
library(ramify)
library(wordspace)
library(MASS)
library(laGP)


eps <- sqrt(.Machine$double.eps)
nonug <- TRUE

sqrt.noneg <- function(x) {
  x[x<0] <- eps
  return(sqrt(x))
}


KL <- function(u1, u2, std1, std2){
  kl_divergence <-  (std2^2 + ((u1 - u2)^2)) / (2*std1^2) + log(std1^2)-log(std2^2)
  return(kl_divergence)
}

KL_all <- function(u1_list, u2, std1_list, std2){
  KLs <- KL(u1_list, u2, std1_list, std2)
  return(mean(KLs, na.rm = TRUE))
}

model_fit_test <- function(trainset, testsets, n_tr = 10000, n_ts = 1600, n_v = 1500, f = 32, classes = 10){
  data_dir = trainset
  
  train.df = read.csv(paste0("data/", data_dir, "/train_features_logits_labels.csv"))
  test.df = read.csv(paste0("data/", data_dir, "/test_features_logits_labels.csv"))
  
  set.seed(430)
  
  # Train (use 10000 samples for training)
  train.df <- train.df[1:n_tr, ]
  
  CNN_train_score <- train.df[, (f+1):(f+classes)]
  CNN_train_score <- normalize.rows(as.matrix(exp(CNN_train_score)), method = "manhattan") #normalize
  
  models <- vector("list", classes)
  
  # some of input variables are constant; need to remove them 
  column.idx <- matrix(TRUE, ncol = classes, nrow = f)
  
  # scale of X 
  scale.X <- matrix(1, ncol = classes, nrow = f)
  
  # min of X 
  min.X <- matrix(0, ncol = classes, nrow = f)
  
  # mean of Y 
  mean.y <- rep(0, classes)
  
  # normalize to [0,1]
  normalize <- function(X, min, scale){
    t((t(X) - min)/scale)
  }
  
  print("Start Model fitting on ImageNet ...")
  for(i in 1:classes){
    print(paste0("class", i))
    X <- train.df[train.df[,"label"]==i-1, 1:f]
    y <- train.df[train.df[,"label"]==i-1, (f+i)] # only look at the scores of correct label
    
    range.X <- apply(X, 2, range)
    scale.X[,i] <- range.X[2,] - range.X[1,]
    min.X[,i] <- range.X[1,]
    scale.X[scale.X[,i] == 0,i] <- 1
    
    # normalize X to [0,1]
    X.norm <- normalize(X, min = min.X[,i], scale = scale.X[,i])
    
    ## build GP and jointly optimize via profile mehtods
    gpisep <- newGPsep(X.norm, y, d=rep(0.01, ncol(X.norm)),
                       g=eps, dK=TRUE)
    
    mleGPsep(gpisep, param="d", tmin=0.001, tmax=0.8)
    
    models[[i]] = gpisep
    
    rm(gpisep)
  } 
  print("Train Finished")
  
  directory_path <- paste0("Rdata_ckpt/", data_dir)
  
  if (!dir.exists(directory_path)) {
    if (dir.create(directory_path)) {
      cat("Directory created successfully:", directory_path, "\n")
    } else {
      cat("Failed to create the directory:", directory_path, "\n")
    }
  } else {
    cat("Directory already exists:", directory_path, "\n")
  }
  
  print("Start Validation ...")
  # Use first 2000 InD data for validation
  valid.df <- test.df[1:n_v, ]    
  valid.X <- valid.df[,1:f]
  results_valid = vector("list", classes) 
  for(i in 1:classes){  # 10 different clusters 
    gpisep = models[[i]]
    results_valid[[i]] <- predGPsep(gpisep, normalize(valid.X, min = min.X[,i], scale = scale.X[,i]), lite=TRUE, nonug = nonug)
  }
    
  print("Testing on InD data ...")
  # Use second 2000 InD data for testing InD
  ind.df = test.df[(n_v+1):(n_ts+n_v),]
  InD.X <- ind.df[, 1:f]  
  results_ind = vector("list", classes)
  for(i in 1:classes){  # 10 different clusters 
    gpisep = models[[i]]
    results_ind[[i]] <- predGPsep(gpisep, normalize(InD.X, min = min.X[,i], scale = scale.X[,i]), lite=TRUE, nonug = nonug)
  }

  for (testset in testsets){
    print(paste0("Test on OOD data: ", testset))

    # Test OOD
    results_ood = vector("list", classes)
    
    # Read data from NN
    t.df = read.csv(paste0("data/", data_dir, "/", testset, "_test.csv"))
      

    # Use 2000 data for testing OOD
    ood.df = t.df[t.df[,"class"]=='OOD', ] 
    ood.df = ood.df[(n_v+1):(n_ts+n_v),]
    OOD.X <- ood.df[, 1:f]
    
    
    for(i in 1:classes){  # 10 different clusters 
      gpisep = models[[i]]
      results_ood[[i]] <- predGPsep(gpisep, normalize(OOD.X, min = min.X[,i], scale = scale.X[,i]), lite=TRUE, nonug = nonug)
    }
    
    save(results_ind, ind.df, results_ood, ood.df, results_valid, valid.df, file=paste0(directory_path, "/", trainset, "_", testset, ".RData"))
  }
  
  print("Test Finished")
  
  deleteGPseps()
}  




get_argmax <- function(x) {
  return(which.max(x) - 1)
}


score_function <- function(trainset, testset, q_ = 0.95, f = 32, n_tr = 10000, n_ts = 1600, n_v = 1500, classes = 10){
  directory_path <- paste0("Rdata_ckpt/", trainset)
  
  load(file=paste0(directory_path, "/", trainset, "_", testset, ".RData")) 
  
  ind.df$predictions = apply(ind.df[, (f+1):(f+classes)], 1, get_argmax) + 1
  ood.df$predictions = apply(ood.df[, (f+1):(f+classes)], 1, get_argmax) + 1
  valid.df$predictions = apply(valid.df[, (f+1):(f+classes)], 1, get_argmax) + 1
  
  
  ind.df$KL = 0
  ood.df$KL = 0
  valid.df$KL = 0
  ind.df$mean = 0
  ood.df$mean = 0
  valid.df$mean = 0
  ind.df$std = 0
  ood.df$std = 0
  valid.df$std = 0
  
  
  for (i in (1:nrow(valid.df))){
    class = valid.df$predictions[i]
    valid.df$KL[i] = KL_all(results_valid[[class]]$mean, results_valid[[class]]$mean[i], # mean
                            sqrt.noneg(results_valid[[class]]$s2), sqrt.noneg(results_valid[[class]]$s2[i])) # SD
  }  
  
  for (i in (1:nrow(ind.df))){
    class = ind.df$predictions[i]
    ind.df$KL[i] = KL_all(results_valid[[class]]$mean, results_ind[[class]]$mean[i], # mean
                           sqrt.noneg(results_valid[[class]]$s2), sqrt.noneg(results_ind[[class]]$s2[i])) # SD
  }
  
  
  
  for (i in (1:nrow(ood.df))){
    class = ood.df$predictions[i]
    ood.df$KL[i] = KL_all(results_valid[[class]]$mean, results_ood[[class]]$mean[i], # mean
                          sqrt.noneg(results_valid[[class]]$s2), sqrt.noneg(results_ood[[class]]$s2[i])) # SD
  }
  
  
  # Calculate KL for validation data, set the threshold based on validation data only (do not need test data for setting threshold)
  KL_list = c()
  for (i in (1:classes)){
    KL_list = c(KL_list, quantile(valid.df[valid.df$predictions == i, ]$KL, q_, na.rm = TRUE))
  }
  
  ID_acc_list = c()
  OOD_acc_list = c()
  ID_sum = 0
  OOD_sum = 0
  columns = 0
  
  for (i in 1:classes){
    ID_acc = mean(ind.df[ind.df$predictions == i, ]$KL < KL_list[i])
    ID_sum = ID_sum + sum(ind.df[ind.df$predictions == i, ]$KL < KL_list[i])
    columns = columns + nrow(ind.df[ind.df$predictions == i, ])
    
    ID_acc_list = c(ID_acc_list, ID_acc)
    OOD_acc = mean(ood.df[ood.df$predictions == i, ]$KL > KL_list[i])
    OOD_sum = OOD_sum + sum(ood.df[ood.df$predictions == i, ]$KL > KL_list[i])
    OOD_acc_list = c(OOD_acc_list, OOD_acc)
  }
  
  # Create a list to store the dataframes
  result <- list(ind.df = ind.df, ood.df = ood.df, 
                 ID_acc = ID_acc_list, OOD_acc = OOD_acc_list, ID_all = ID_sum/n_ts, OOD_all = OOD_sum/n_ts, 
                 AUROC = 1/2 - (1-OOD_sum/n_ts)/2 + ID_sum/n_ts/2)
  
  return(result)
}


######## Call above Functions #######

values <- c(0)   # The best group for 32&64, change to other groups or add more groups here
for (i in values){
  test_name = i
  
  f = 32
  InD_Dataset = paste0("imagenet10-",f, "-", test_name, "-o1")
  OOD_Datasets = c("DTD", "iSUN", "LSUN-C", "LSUN-R", "Places365", "SVHN")
  n_tr = 10000
  n_ts = 1600
  n_v = 1500
  classes = 10
  
  #model_fit_test(trainset = InD_Dataset, testsets = OOD_Datasets, n_tr = n_tr, n_ts = n_ts, n_v = n_v, f = f)  # Run only once
}

# Write to a log file
log_file <- "results_imagenet.txt"
sink(log_file, append = TRUE)
cat("==== Log Start ====\n")
cat(Sys.time(), " - Starting the script\n")

for (i in values){
  test_name = i
  f = 32
  InD_Dataset = paste0("imagenet10-",f, "-", test_name, "-o1")
  OOD_Datasets = c("DTD", "iSUN", "LSUN-C", "LSUN-R", "Places365", "SVHN") 
  n_tr = 10000 
  n_ts = 1600
  n_v = 1500
  classes = 10
  
  list0.95_InD = c()
  list0.95_OOD = c()
  list0.95_AUROC = c()
  for (OOD_Dataset in OOD_Datasets){
    pred = score_function(InD_Dataset, OOD_Dataset, q_ = 0.95, f = f, n_tr = n_tr, n_ts = n_ts, n_v = n_v, classes = classes)
    list0.95_InD = c(list0.95_InD, pred$ID_all)
    list0.95_OOD = c(list0.95_OOD, pred$OOD_all)
    list0.95_AUROC = c(list0.95_AUROC, pred$AUROC)
  }
  
  list0.9_InD = c()
  list0.9_OOD = c()
  list0.9_AUROC = c()
  for (OOD_Dataset in OOD_Datasets){
    pred = score_function(InD_Dataset, OOD_Dataset, q = 0.9, f = f, n_tr = n_tr, n_ts = n_ts, n_v = n_v, classes = classes)
    list0.9_InD = c(list0.9_InD, pred$ID_all)
    list0.9_OOD = c(list0.9_OOD, pred$OOD_all)
    list0.9_AUROC = c(list0.9_AUROC, pred$AUROC)
  }
  
  list0.8_InD = c()
  list0.8_OOD = c()
  list0.8_AUROC = c()
  for (OOD_Dataset in OOD_Datasets){
    pred = score_function(InD_Dataset, OOD_Dataset, q = 0.8, f = f, n_tr = n_tr, n_ts = n_ts, n_v = n_v, classes = classes)
    list0.8_InD = c(list0.8_InD, pred$ID_all)
    list0.8_OOD = c(list0.8_OOD, pred$OOD_all)
    list0.8_AUROC = c(list0.8_AUROC, pred$AUROC)
  }
  
  df = data.frame(InD_0.95 = list0.95_InD,
                  OOD_0.95 = list0.95_OOD,
                  AUROC_0.95 = list0.95_AUROC,
                  InD_0.9 = list0.9_InD,
                  OOD_0.9 = list0.9_OOD,
                  AUROC_0.9 = list0.9_AUROC,
                  InD_0.8 = list0.8_InD,
                  OOD_0.8 = list0.8_OOD,
                  AUROC_0.8 = list0.8_AUROC)
  rownames(df) <- OOD_Datasets
  # Console output
  sink() 
  cat("InD Dataset: ", InD_Dataset, "\n")
  cat("Features: ", f, "\n")
  cat("Training Samples: ", n_tr, "\n")
  cat("Results DataFrame:\n")
  print(df)

  # Write to a log file
  sink(log_file, append = TRUE)
  cat("InD Dataset: ", InD_Dataset, "\n")
  cat("Features: ", f, "\n")
  cat("Training Samples: ", n_tr, "\n")
  cat("Results DataFrame:\n")
  print(df)
  
}

sink() 
cat("Logs have been written to", log_file, "\n")