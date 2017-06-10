import numpy as np


class IMPHD:
    def __init__(self, randomSeed=42, n_trees=100, n_jobs=-1):
        self._param_dict = dict()
        self._level_features = dict()
        self._difference_features = dict()
        self._rf_best = None
        self._params_best = None
        self._cols_list = []

    def _save_level_features(self, mts_list):
        for num_intervals in self._param_dict['num_intervals']:
            self._level_features[str(num_intervals)] = np.vstack(
                list(map(lambda mts: self._extract_level_features(mts, num_intervals), mts_list)))

    def _extract_level_features(self, mts, num_intervals):
        feat_vec = np.array([])
        interval_cuts = np.linspace(num=num_intervals + 1, dtype=np.int, start=0, stop=mts.shape[0], endpoint=True)
        # Summarize intervals by mean, update feat_vec for each summarizaton of each interval
        for idx_cut in range(interval_cuts.shape[0])[:-1]:
            # The case where the time series are upsampled because length < lambda or sometimes the cuts cannot be evenly spaced
            if interval_cuts[idx_cut] == interval_cuts[idx_cut + 1]:
                feat_vec = np.append(feat_vec, mts[interval_cuts[idx_cut], :])
            # The usual case, take the mean between the current cut point and the next
            else:
                feat_vec = np.append(feat_vec, mts[interval_cuts[idx_cut]:interval_cuts[idx_cut + 1], :].mean(axis=0))
        return feat_vec

    def _save_polar_features(self, mts_list):
        # Extract features for each beta provided by the user
        for radius_cut in self._param_dict['radius_cut']:
            self._difference_features[str(radius_cut)] = np.vstack(
                list(map(lambda mts: self._extract_polar_features(mts, radius_cut), mts_list)))

    def _extract_polar_features(self, dmts, radius_cut):
        feat_vec = np.array([])
        outside_limit = 7 * np.pi / 8
        # For each combination in the combination set, extract polar features
        for cols in self._cols_list:
            dmts_cols = dmts[:, cols]
            # Beta parameter filters the polar points at this point
            nonzero_dmts = dmts_cols[np.where(np.linalg.norm(dmts_cols, axis=1) > radius_cut)]
            if nonzero_dmts.shape[0] == 0:
                # Assumes uniformity when all the polar points are below
                new_feat = np.repeat(0.125, 8).reshape(-1)
            else:
                # Fifth bin is handled seperately as np.histogram does not handle circular intervals
                polars = np.arctan2(nonzero_dmts[:, 1], nonzero_dmts[:, 0])
                fifth_bin_mask = np.where(np.logical_or(polars < -outside_limit, polars > outside_limit))[0]
                fifth_bin = fifth_bin_mask.shape[0]
                new_feat = np.append(np.histogram(polars[np.setdiff1d(np.arange(polars.shape[0]), fifth_bin_mask)],
                                                  bins=7, range=(-outside_limit, outside_limit), density=False)[0],
                                     fifth_bin)
                # Note the normalization of the histogram before appending
            feat_vec = np.append(feat_vec, new_feat / np.sum(new_feat))
        return feat_vec

    def extract_features(self, mts_list, num_attributes, list_lambdas, list_betas, dmts_list=None, reduced_comb=None):
        if dmts_list is None:
            print("First difference series not found, will be calculated...");  # self._print_line();
        else:
            assert len(mts_list) == len(dmts_list), "Error: Given mts_list and dmts_list are not of same size"
        self._col_dim = num_attributes
        # Check whether a combiation set is provided by the user, if not, use the full combination set
        if reduced_comb is None:
            print("Full combination set is being used");
            self._print_line();
            import itertools
            self._cols_list = list(itertools.combinations(range(self._col_dim), 2))
        else:
            print("User defined combination set is being used")
            self._cols_list = reduced_comb

        self._param_dict['num_intervals'] = list_lambdas
        self._param_dict['radius_cut'] = list_betas
        print("IM features are being computed for the following lambda values:")
        print(list_lambdas);
        self._print_line();
        self._save_level_features(mts_list)
        print("PHD features are being computed for the following beta values:")
        print(list_betas);
        self._print_line();
        self._save_polar_features(dmts_list)

    def get_feature_set(self, num_intervals, radius_cut):
        return np.append(self._level_features[str(num_intervals)], self._difference_features[str(radius_cut)], axis=1)

    def random_reduced_combination_generator(self, num_attributes, num_reduced_subsets, random_seed=42):
        import random
        import itertools
        random.seed(a=random_seed)
        cols_list = []
        cols_set = set(range(num_attributes))
        while len(cols_set) >= num_reduced_subsets:
            next_sample = random.sample(cols_set, num_reduced_subsets)
            cols_list.extend(itertools.combinations(next_sample, 2))
            cols_set = cols_set - set(next_sample)
        if len(cols_set) > 1:
            cols_list.extend(itertools.combinations(list(cols_set), 2))
        return cols_list

    def extract_features_for_single_instance(self, mts, num_intervals, radius_cut):
        dmts = np.diff(mts, axis=0)
        return np.append(self._extract_level_features(mts, num_intervals),
                         self._extract_polar_features(dmts, radius_cut), axis=0)

    def _print_line(self):
        print("**** -------------------------------------- ****")
