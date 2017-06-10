def parse_mts_dataset( data ):
    mts_list = data['mts_list']
    dmts_list = data['dmts_list']
    train_index = data['train_index']
    test_index = data['test_index']
    labels_list = data['labels_list']
    num_attributes = mts_list[0].shape[1]    
    return( mts_list , dmts_list , train_index , test_index , labels_list , num_attributes)