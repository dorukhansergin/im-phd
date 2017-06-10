# import required packages
import os
import pickle
from mts_dataset_parser import parse_mts_dataset

"""
You can change the "file" var to try a different dataset in the selection
Or import your own dataset externally
Do not change the "dirname" var to work locally
""" 
dirname = "../data"
file = "scaled_Uwave_mts.p"

with open( os.path.join( dirname , file ) , 'rb' ) as infile:
    data = pickle.load( infile )
( mts_list , dmts_list , train_index , test_index , labels_list , num_attributes) = parse_mts_dataset( data )
labels_train = labels_list[train_index]; labels_test = labels_list[test_index]

from im_phd import IMPHD
list_lambdas = list( range(7,21,3) )
list_betas = [0, 0.025 , 0.05, 0.075, 0.1]
imphd = IMPHD()
imphd.extract_features( mts_list=mts_list, num_attributes=num_attributes , list_lambdas=list_lambdas
                       , list_betas=list_betas, dmts_list=dmts_list )
			
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

# Initialize random forest parameters and the classifier
n_trees=100; n_jobs=-1; random_seed=0;
rf = RandomForestClassifier( oob_score = True , n_estimators = n_trees , n_jobs = n_jobs , random_state = random_seed )

# Create a parameter grid based on the list of lambdas and betas provided, this step can be adjusted 
# as long as the features are readily extracted in the IMPHD class
pg = ParameterGrid( {'num_intervals':list_lambdas , 'radius_cut':list_betas} )
best_score = -1
for params in list( pg ):
    # Use get_feature_set method to retrieve specific feature set per parameter tuple
    feature_set = imphd.get_feature_set( **params )[train_index]
    rf.fit( feature_set , labels_train )
    # Update the best model if the incoming score is better
    if rf.oob_score_ > best_score:
        rf_best = copy.copy(rf)
        params_best = params
        best_score = rf.oob_score_

features_test = feature_set = imphd.get_feature_set( **params_best )[test_index]
predicted_labels = rf_best.predict( features_test )

from sklearn.metrics import accuracy_score, classification_report
print( classification_report( labels_test, predicted_labels ) )
print("The overall accuracy is:")
print( accuracy_score( labels_test , predicted_labels ) )