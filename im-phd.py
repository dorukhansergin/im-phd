class PHDIM:
    def __init__( self, randomSeed = 42 , n_trees = 100 , n_jobs = -1 ):
        self._param_dict = None
        self._level_features = dict()
        self._difference_features = dict()
        self._rf_best = None
        self._params_best = None

    def _saveLevelFeatures( self , mts_list ):
        for seg_count in self._param_dict['seg_count']:
            self._level_features[str(seg_count)] = np.vstack( list( map( lambda mts: self._extractLevelFeatures( mts, seg_count ) , mts_list ) ) )
            
    def _extractLevelFeatures( self , mts , seg_count ):
        # Initialize feature vector and segmeng cut points
        feat_vec = np.array([])        
        segment_cuts = np.linspace( num= seg_count+1 , dtype=np.int , start=0 , stop=mts.shape[0], endpoint=True )
        # Summarize segments by mean
        for idx_cut in range( segment_cuts.shape[0] )[:-1]:               
            if segment_cuts[idx_cut] == segment_cuts[idx_cut + 1 ]:
                feat_vec = np.append( feat_vec , mts[ segment_cuts[idx_cut] , : ] )  
            else:
                feat_vec = np.append( feat_vec , mts[ segment_cuts[idx_cut]:segment_cuts[idx_cut + 1 ] , : ].mean(axis=0) )
        return feat_vec
    
    def _savePolarFeatures( self , mts_list , reduced_comb ):
        from functools import partial
        for bin_count in self._param_dict['bin_count']:
            f = partial( self._extractPolarFeatures , bin_count = bin_count , reduced_comb = reduced_comb )
            self._difference_features[str(bin_count)] = np.vstack( list( map( lambda mts: self._extractPolarFeatures( mts , bin_count , reduced_comb ) , mts_list ) ) )

    def _extractPolarFeatures( self , dmts , radius_cut , reduced_comb ):
        feat_vec = np.array([])
        outside_limit = 7*np.pi/8
        import itertools
        if reduced_comb is None:
            cols_list = list( itertools.combinations( range( self.col_dim_ ) , 2 ) )
        else:
            import random
            random.seed( a = self._randomSeed )
            cols_list = []
            cols_set = set( range( self.col_dim_ ) )
            while len( cols_set ) >= reduced_comb:
                next_sample = random.sample( cols_set , reduced_comb )
                cols_list.extend( itertools.combinations( next_sample , 2 ) )
                cols_set = cols_set - set( next_sample )
            if len( cols_set ) > 1:
                cols_list.extend( itertools.combinations( list( cols_set ) , 2 ) ) 
        interval_cuts = np.linspace( num=2 , dtype=np.int , start=0 , stop=dmts.shape[0] , endpoint=True )
        for cut_idx in np.arange(interval_cuts.shape[0])[1:]:
            for cols in cols_list:
                dmts_cols = dmts[ interval_cuts[(cut_idx-1)]:interval_cuts[cut_idx] ,cols]
                nonzero_dmts = dmts_cols[ np.where( np.linalg.norm( dmts_cols , axis=1 ) > radius_cut ) ]
                if nonzero_dmts.shape[0] == 0:
                    # Assume uniformity
                    new_feat = np.repeat(0.125,8).reshape(-1)
                else:    
#                     stability_ratio = 1 - ( nonzero_dmts.shape[0] / dmts_cols.shape[0] )
                    polars = np.arctan2( nonzero_dmts[:,1] , nonzero_dmts[:,0] )
                    fifth_bin_mask = np.where( np.logical_or( polars < -outside_limit , polars > outside_limit ) )[0]
                    fifth_bin = fifth_bin_mask.shape[0]                     
                    new_feat = np.append( np.histogram( polars[ np.setdiff1d( np.arange(polars.shape[0]) , fifth_bin_mask)] , 
                                bins = 7 , range = (-outside_limit,outside_limit), density=False)[0] , fifth_bin )
                feat_vec = np.append( feat_vec , new_feat )
        return feat_vec

    def extract_features( self , mts_list , num_attributes , list_lambdas , list_betas , dmts_list = None , reduced_comb = None ):
        assert self._param_dict is not None , "Parameter dictionary is not defined"
        self.col_dim_ = num_attributes
        self._saveLevelFeatures( mts_list )
        self._savePolarFeatures( dmts_list , reduced_comb )     
        self._labels = np.vstack( labels_list ).reshape(-1)
    
    def _getFeatureSet( self , seg_count , bin_count): 
        return np.append( self._level_features[str(seg_count)] , self._difference_features[str(bin_count)] , axis = 1 )

    def _findBestParamSet( self , train_index ):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import ParameterGrid
        import copy
        pg = ParameterGrid( self._param_dict )
        best_score = -1
        for kargs in list( pg ):
            feature_set = self._getFeatureSet( **kargs )[train_index]
            rf = RandomForestClassifier( n_estimators = self._n_trees , oob_score = True , n_jobs = self._n_jobs , random_state = self._randomSeed )
            rf.fit( feature_set , self._labels[train_index] )
            if rf.oob_score_ > best_score:
                rf_best = copy.copy(rf)
                kargs_best = kargs
                best_score = rf.oob_score_
        return ( kargs_best , rf_best )
            
    def KFoldTest( self , k = 10 ):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score
        skf = StratifiedKFold( n_splits = k , random_state = self._randomSeed )
        self._kfold_params = []
        self._kfold_scores = []
        for train_index , test_index in skf.split( np.zeros( self._labels.shape[0] ) , self._labels ):
            ( kargs_best , rf_best ) = self._findBestParamSet( train_index )
            predicted_labels = rf_best.predict( self._getFeatureSet( **kargs_best )[test_index] )
            self._kfold_params.append( kargs_best )
            self._kfold_scores.append( accuracy_score( self._labels[test_index] , predicted_labels ) )
        return ( self._kfold_scores , self._kfold_params )
    
    def TestTrainAccuracy( self , test_index , train_index ):
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        ( kargs_best , rf_best ) = self._findBestParamSet( train_index )
        predicted_labels = rf_best.predict( self._getFeatureSet( **kargs_best )[test_index] )
        return ( accuracy_score( self._labels[test_index] , predicted_labels ) , kargs_best )
    
    def train( self ):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import ParameterGrid
        import copy
        pg = ParameterGrid( self._param_dict )
        best_score = -1
        for kwargs in list( pg ):
            feature_set = self._getFeatureSet( **kwargs )
            rf = RandomForestClassifier( n_estimators = self._n_trees , oob_score = True , n_jobs = self._n_jobs , random_state = self._randomSeed )
            rf.fit( feature_set , self._labels )
            if rf.oob_score_ > best_score:
                self._rf_best  = copy.copy(rf)
                self._params_best = kwargs
                best_score = rf.oob_score_
    
    def predict( self , mts_test , labels_test , reduced_comb ):
        dmts_test = [ np.diff( mts , axis=0) for mts in mts_test ]
        features_test = np.append( 
            np.vstack( list( map( lambda mts: self._extractLevelFeatures( mts , self._params_best['seg_count'] ) , mts_test ) ) ) ,
            np.vstack( list( map( lambda mts: self._extractPolarFeatures( mts , self._params_best['bin_count'] , reduced_comb ) , dmts_test ) ) )    
            , axis=1 )
        return self._rf_best.predict( features_test )
    

    

