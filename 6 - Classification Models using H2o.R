# Install packages
  #install.packages("randomForest")
  #install.packages("mlbench")
  #install.packages("caret")
  #install.packages("e1071")
  #install.packages("data.table")
  #install.packages("lime")
  #install.packages("gridExtra")
  #install.packages("summarytools")

  # Installing h20 package
  # The following two commands remove any previously installed H2O packages for R.
    #if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
    #if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
  
  # Next, we download, install and initialize the H2O package for R.
    #install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1497/R", getOption("repos"))))

# Load libraries
  library(foreign)
  library(randomForest)
  library(mlbench)
  library(caret)
  library(e1071)
  library(data.table)
  library(lime)
  library(gridExtra)
  library(summarytools)

  library(h2o)
    localH2O = h2o.init()
  
  
# Set woking directory
  setwd("R:/PRISM_DataSci/FR17 FAMCATML/Data Files")

# Load training and test datasets
  training <- fread("fr17_train.csv", stringsAsFactors = T)
  validation <- fread("fr17_test.csv", stringsAsFactors = T)
  
  # Summary descriptive
    #view(dfSummary(training))
    #view(dfSummary(validation))
  
  # Exclude variables not needed for randon forest modelling
    train <- as.data.frame(training[,3:43])
    test <- as.data.frame(validation[,3:43])
    
  # Convert predictor variables to factor
    cols = c(1:7,9:17,24,25,28,31:40)
    train[,cols] <- lapply(train[,cols],as.factor)
    test[,cols] = lapply(test[,cols], as.factor) 
    
    # Summary descriptive
      #view(dfSummary(train))
      #view(dfSummary(test))
    
    
# Convert data to h20 objects
  train <- as.h2o(train)
  test <- as.h2o(test)
  
  # Columns
    names(train)
  
# Identify the response column
  ycol <- "FH"
  
# Identify the predictor columns
  xcols <- setdiff(names(train), ycol)
  
# Convert response to factor (required for randomForest)
  train[,ycol] <- as.factor(train[,ycol])
  test[,ycol] <-as.factor(test[,ycol])
    

##-------------------------------------------------------------
# LOGISTIC REGRESSION WITH RANDOM HYPERPARAMETER SEARCH

  hyper_params <- list(alpha = seq(from = 0, to = 1, by = 0.001),
                       lambda = seq(from = 0, to = 1, by = 0.000001))
  
  search_criteria <- list(strategy = "RandomDiscrete",
                          max_runtime_secs = 10*3600,
                          max_models = 100,
                          stopping_metric = "AUC", 
                          stopping_tolerance = 0.00001, 
                          stopping_rounds = 5, 
                          seed = 1234)
  
  models_glm <- h2o.grid(algorithm = "glm", grid_id = "grd_glm", x = xcols, 
                         y = ycol, training_frame = train, 
                         validation_frame = test, nfolds = 0, 
                         family = "binomial", hyper_params = hyper_params, 
                         search_criteria = search_criteria, stopping_metric = "AUC", 
                         stopping_tolerance = 1e-5, stopping_rounds = 5, seed = 1234)
  
  models_glm_sort <- h2o.getGrid(grid_id = "grd_glm", sort_by = "auc", decreasing = TRUE)
  models_glm_best <- h2o.getModel(models_glm_sort@model_ids[[1]])
  
  
  # Parameter values for the best model
    glm_paramaters <- models_glm_best@allparameters
  
  # AUC value for the validation dataset
    glm_auc <- models_glm_best@model$validation_metrics@metrics$AUC
    
  # Variable importance
    glm_imp <- h2o.varimp(models_glm_best)

    


##-------------------------------------------------------------
  # RANDOM FOREST WITH RANDOM HYPERPARAMETER SEARCH
    
    hyper_params <- list(ntrees = 10000,  ## early stopping
                         max_depth = 5:15, 
                         min_rows = c(1,5,10,20,50,100),
                         nbins = c(30,100,300),
                         nbins_cats = c(64,256,1024),
                         sample_rate = c(0.7,1),
                         mtries = c(-1,2,6))
    
    search_criteria <- list(strategy = "RandomDiscrete",
                            max_runtime_secs = 10*3600,
                            max_models = 100,
                            stopping_metric = "AUC", 
                            stopping_tolerance = 0.00001, 
                            stopping_rounds = 5, 
                            seed = 1234)
    
    models_rf <- h2o.grid(algorithm = "randomForest", grid_id = "grd_rf", x = xcols, 
                          y = ycol, training_frame = train, 
                          validation_frame = test, nfolds = 0, hyper_params = hyper_params, 
                          search_criteria = search_criteria, stopping_metric = "AUC", 
                          stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)
    
    models_rf_sort <- h2o.getGrid(grid_id = "grd_rf", sort_by = "auc", decreasing = TRUE)
    models_rf_best <- h2o.getModel(models_rf_sort@model_ids[[1]])
    
    
    # Parameter values for the best model
      rf_parameters <- models_rf_best@allparameters
      
    # AUC value for the validation dataset
      rd_auc <- models_rf_best@model$validation_metrics@metrics$AUC
      
    # Variable importance
      rf_imp <- h2o.varimp(models_rf_best)
      
    
    
     
##-------------------------------------------------------------
  # GRADIENT BOOSTING MACHINE WITH RANDOM HYPERPARAMETER SEARCH
      
    hyper_params <- list(ntrees = 10000,  ## early stopping
                         max_depth = 5:15, 
                         min_rows = c(1,5,10,20,50,100),
                         learn_rate = c(0.001,0.01,0.1),  
                         learn_rate_annealing = c(0.99,0.999,1),
                         sample_rate = c(0.7,1),
                         col_sample_rate = c(0.7,1),
                         nbins = c(30,100,300),
                         nbins_cats = c(64,256,1024))
      
    search_criteria <- list(strategy = "RandomDiscrete",
                            max_runtime_secs = 10*3600,
                            max_models = 100,
                            stopping_metric = "AUC", 
                            stopping_tolerance = 0.00001, 
                            stopping_rounds = 5, 
                            seed = 1234)
      
    models_gbm <- h2o.grid(algorithm = "gbm", grid_id = "grd_gbm", x = xcols,
                           y = ycol, training_frame = train, 
                           validation_frame = test, nfolds = 0, hyper_params = hyper_params, 
                           search_criteria = search_criteria, stopping_metric = "AUC", 
                           stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)
      
    models_gbm_sort <- h2o.getGrid(grid_id = "grd_gbm", sort_by = "auc", decreasing = TRUE)
    models_gbm_best <- h2o.getModel(models_gbm_sort@model_ids[[1]])
    
    # Parameter values for the best model
      gbm_parameters <- models_gbm_best@allparameters
      
    # AUC value for the validation dataset
      gbm_auc <- models_gbm_best@model$validation_metrics@metrics$AUC
      
    # Variable importance
      gbm_imp <- h2o.varimp(models_gbm_best)   
      
      
      
      
##-------------------------------------------------------------
  # NEURAL NETWORK MACHINE WITH RANDOM HYPERPARAMETER SEARCH
      
    hyper_params <- list(activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", 
                                        "MaxoutWithDropout", "TanhWithDropout"), 
                         hidden = list(c(50, 50, 50, 50), c(200, 200), c(200, 200, 200), c(200, 200, 200, 200)), 
                         epochs = c(50, 100, 200), 
                         l1 = c(0, 0.00001, 0.0001), 
                         l2 = c(0, 0.00001, 0.0001), 
                         adaptive_rate = c(TRUE, FALSE), 
                         rate = c(0, 0.1, 0.005, 0.001), 
                         rate_annealing = c(1e-8, 1e-7, 1e-6), 
                         rho = c(0.9, 0.95, 0.99, 0.999), 
                         epsilon = c(1e-10, 1e-8, 1e-6, 1e-4), 
                         momentum_start = c(0, 0.5),
                         momentum_stable = c(0.99, 0.5, 0), 
                         input_dropout_ratio = c(0, 0.1, 0.2))
      
     search_criteria <- list(strategy = "RandomDiscrete",
                             max_runtime_secs = 10*3600,
                             max_models = 100,
                             stopping_metric = "AUC", 
                             stopping_tolerance = 0.00001, 
                             stopping_rounds = 5, 
                             seed = 1234)
      
    models_dl <- h2o.grid(algorithm = "deeplearning", grid_id = "grd_dl", x = xcols, 
                          y = ycol, training_frame = train, 
                          validation_frame = test, nfolds = 0, hyper_params = hyper_params, 
                          search_criteria = search_criteria, stopping_metric = "AUC", 
                          stopping_tolerance = 1e-3, stopping_rounds = 2, seed = 1234)
      
    models_dl_sort <- h2o.getGrid(grid_id = "grd_dl", sort_by = "auc", decreasing = TRUE)
    models_dl_best <- h2o.getModel(models_dl_sort@model_ids[[1]])  
    
    
    # Parameter values for the best model
      dl_parameters <- models_dl_best@allparameters
      
    # AUC value for the validation dataset
      dl_auc <- models_dl_best@model$validation_metrics@metrics$AUC
      
    # Variable importance
      dl_imp <- h2o.varimp(models_dl_best)  
      
      
      
      
##-------------------------------------------------------------
  # EMSEMBLE MODEL
    # a stacking ensemble model was also created by combining four base models including Logistic Regression, 
    # Random Forest, Gradient Boosting Machine, and Neural Network.
    # The value of hyperparameters in the four base models are the same as the hyperparameters 
    # in the best models selected before.
      
      md_lr <- h2o.glm(x = xcols, y = ycol, training_frame = train, 
                       nfolds = 10, fold_assignment = "AUTO", keep_cross_validation_predictions = TRUE,
                       seed = 1234, family = "binomial", alpha = 0.144, lambda = 0.005041)
      
      md_rf <- h2o.randomForest(x = xcols, y = ycol, training_frame = train,
                                nfolds = 10, fold_assignment = "AUTO", 
                                keep_cross_validation_predictions = TRUE, ntrees = 10000, max_depth = 13, 
                                min_rows = 10, nbins = 30, nbins_cats = 256, mtries = -1, sample_rate =0.7, 
                                stopping_metric = "AUC", stopping_tolerance = 0.001, stopping_rounds = 2, 
                                seed = 1234 )
      
      md_gbm <- h2o.gbm(x = xcols, y = ycol, training_frame = train,
                        nfolds = 10, fold_assignment = "AUTO", 
                        keep_cross_validation_predictions = TRUE, ntrees = 10000, 
                        max_depth = 11, min_rows = 100, nbins = 300, nbins_cats = 256, 
                        learn_rate = 0.01, learn_rate_annealing =0.999, sample_rate = 0.7, 
                        col_sample_rate = 0.7, stopping_metric = "AUC", stopping_tolerance = 0.001, 
                        stopping_rounds = 2, seed = 1234 )
      
      md_dl <- h2o.deeplearning(x = xcols, y = ycol, training_frame = train,
                                nfolds = 10, fold_assignment = "AUTO", 
                                keep_cross_validation_predictions = TRUE, activation = "Maxout", 
                                hidden = c(200, 200, 200), epochs = 200, adaptive_rate = FALSE, rho = 0.999, 
                                epsilon = 1e-06, rate = 0.001, rate_annealing = 1e-6, momentum_start = 0, 
                                momentum_stable = 0.99, input_dropout_ratio = 0.2, l1 = 1e-5, l2 = 1e-4, 
                                stopping_metric = "AUC", stopping_tolerance = 0.001, stopping_rounds = 2, 
                                seed = 1234 )
      
      md_ens <- h2o.stackedEnsemble(x = xcols, y = ycol, training_frame = train,
                                    base_models = list(md_lr@model_id, md_rf@model_id, md_gbm@model_id, 
                                                       md_dl@model_id))
      
    
  
      
##-------------------------------------------------------------
  # MODEL COMPARISON
    h2o.auc(h2o.performance(md_lr, test))
    h2o.auc(h2o.performance(md_rf, test))
    h2o.auc(h2o.performance(md_gbm, test))
    h2o.auc(h2o.performance(md_dl, test))
    h2o.auc(h2o.performance(md_ens, test))
    
    
    ##  To get model co-efficients
        h2o.getModel(md_ens@model$metalearner$name)@model$coefficients_table
        
        
        
##----------------------------------------------------------------------------
  # EXPORT VARIABLE IMPORTANCE 
    write.dta(glm_imp, "R:/PRISM_DataSci/FR17 FAMCATML/Results/glm_imp.dta")
    write.dta(dl_imp, "R:/PRISM_DataSci/FR17 FAMCATML/Results/dl_imp.dta")
        

        
## Validation of models and exporting probabilities
  # Make predictions using logistic regression model
    lr_model <- as.data.frame(h2o.predict(object = md_lr, newdata = test))
    
    # Combine patid with predictions from dl2
      lrm_valid <- cbind(validation$patid, validation$FH, lr_model$p1)
      lrm_valid <- as.data.frame(lrm_valid)
      colnames(lrm_valid) <- c("patid", "FH", "lrm_prob")
    
    # Export into stata file
      library(foreign)
      write.dta(lrm_valid, "R:/PRISM_DataSci/FR17 FAMCATML/Data Files/output/lrm_valid.dta")
     
       
      
  # Make predictions using random forest model
    rf_model <- as.data.frame(h2o.predict(object = md_rf, newdata = test))
      
    # Combine patid with predictions from dl2
      rfm_valid <- cbind(validation$patid, validation$FH, rf_model$p1)
      rfm_valid <- as.data.frame(rfm_valid)
      colnames(rfm_valid) <- c("patid", "FH", "rfm_prob")
      
      # Export into stata file
      library(foreign)
      write.dta(rfm_valid, "R:/PRISM_DataSci/FR17 FAMCATML/Data Files/output/rfm_valid.dta") 
      
      
      
  # Make predictions using gradient boost model
    gbm_model <- as.data.frame(h2o.predict(object = md_gbm, newdata = test))
      
      # Combine patid with predictions from dl2
      gbm_valid <- cbind(validation$patid, validation$FH, gbm_model$p1)
      gbm_valid <- as.data.frame(gbm_valid)
      colnames(gbm_valid) <- c("patid", "FH", "gbm_prob")
      
      # Export into stata file
      library(foreign)
      write.dta(gbm_valid, "R:/PRISM_DataSci/FR17 FAMCATML/Data Files/output/gbm_valid.dta") 
 
      
      
  # Make predictions using deep-learning model
    dl_model <- as.data.frame(h2o.predict(object = md_dl, newdata = test))
      
      # Combine patid with predictions from dl2
      dlm_valid <- cbind(validation$patid, validation$FH, dl_model$p1)
      dlm_valid <- as.data.frame(dlm_valid)
      colnames(dlm_valid) <- c("patid", "FH", "dlm_prob")
      
      # Export into stata file
      library(foreign)
      write.dta(dlm_valid, "R:/PRISM_DataSci/FR17 FAMCATML/Data Files/output/dlm_valid.dta") 

    
       
  # Make predictions using deep-learning model
    ens_model <- as.data.frame(h2o.predict(object = md_ens, newdata = test))
      
      # Combine patid with predictions from dl2
      ensm_valid <- cbind(validation$patid, validation$FH, ens_model$p1)
      ensm_valid <- as.data.frame(ensm_valid)
      colnames(ensm_valid) <- c("patid", "FH", "ensm_prob")
      
      # Export into stata file
      library(foreign)
      write.dta(ensm_valid, "R:/PRISM_DataSci/FR17 FAMCATML/Data Files/output/ensm_valid.dta")  
        
  
 
# good practice to shut down h2o environment
  h2o.shutdown(prompt = FALSE)
  
  
  
  