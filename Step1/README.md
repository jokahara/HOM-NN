Adds data from new .pkl files to training to retrain pretrained_model.
Initially half of each dataset is used for 8-fold cross validation, 
and the rest are kept as a test set. Test data not meeting the error limit
is added to training data iteratively, until accuracy is within desired range.
