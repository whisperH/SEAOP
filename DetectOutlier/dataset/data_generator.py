import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import platform
from copy import copy
from DetectOutlier.utils.data import get_distance, get_scale, get_global

class CancerDataGenerator(object):
    def __init__(self, args):

        if platform.system().lower() == 'windows':
            self.R_USER = args.Windows_R_USER
            self.R_HOME = None
        else:
            self.R_USER = args.Linux_R_USER
            self.R_HOME = '/usr/lib/R'
        self.use_cross_valid = args.use_cross_valid
        self.sample_ratio = args.sample_ratio
        self.generate_duplicates = args.generate_duplicates
        self.input_matrix = args.input_matrix

        self.joint_data = args.joint_data
        self.fill_strategy = args.fill_strategy
        self.scale_strategy = args.scale_strategy

        self.prepare_data_dir = args.prepare_data_dir
        self.dataset_list = args.dataset_list_raw

    def preprocessor(self, data_dir=None, dataset_file=None, raw_data=None, save_scale_data=True):
        if self.joint_data == 'original':
            if (raw_data is None) and (data_dir is not None) and (dataset_file is not None):
                raw_data = pd.read_csv(os.path.join(data_dir, dataset_file), index_col=0)
            assert raw_data is not None, f"no data to process, check {data_dir} and {dataset_file}"
            scale_data = get_scale(raw_data, self.scale_strategy, r_user=self.R_USER, r_home=self.R_HOME)
            index_list = raw_data.index.tolist()
            column_list = raw_data.columns.tolist()

            scale_data = pd.DataFrame(
                data=scale_data,
                index=index_list,
                columns=column_list
            )
            scale_data = get_global(scale_data, strategy=self.fill_strategy)

            if save_scale_data:
                scale_data.to_csv(os.path.join(self.prepare_data_dir, dataset_file))

            return scale_data
        else:
            raise Exception("unknown joint_data in configs")

    def generator(self, dataset, kf_num, saved_dir, dist_type='l2', **kwargs):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        '''

        # load dataset
        if type(dataset) == str :
            assert dataset in self.dataset_list, "data file not in dataset_list_raw"
            # pair_data = np.load(os.path.join(self.data_dir, dataset + '.npy'), allow_pickle=True)
            feature_data = pd.read_csv(
                os.path.join(self.prepare_data_dir, dataset), index_col=0
            )
            dataname=dataset.split(".")[0]
        else:
            feature_data = dataset
            dataname=kwargs.get("dataset_file", "infer_data.csv").split(".")[0]

        # spliting the current data to the training set and testing set
        index_list = feature_data.index.tolist()

        # ============================================================= #

        # generate the correlation matrix of each samples
        # if not os.path.exists(os.path.join(self.prepare_data_dir, f'dataset_list_{dataname}_corr.csv')):
        #     corr_data = pd.DataFrame(
        #         data=np.zeros((len(index_list), len(index_list))),
        #         index=index_list,
        #         columns=index_list
        #     )
        #
        #     all_sample_num = len(index_list)
        #     for sample_id in range(all_sample_num-1):
        #         for other_sample_id in range(sample_id+1, all_sample_num):
        #             _sample = feature_data.iloc[[sample_id, other_sample_id], :]
        #             id_dist = get_distance(_sample.values, dist_type)
        #             corr_data.loc[index_list[sample_id], index_list[other_sample_id]] = id_dist
        #             corr_data.loc[index_list[other_sample_id], index_list[sample_id]] = id_dist
        #
        #
        #     corr_data.to_csv(os.path.join(self.prepare_data_dir, f'dataset_list_{dataname}_corr.csv'))

        if self.input_matrix == 'feature_matrix':
            in_data = feature_data
        else:
            raise Exception("Unknown input matrix")

        min_feature_value = in_data.min().min()
        validate_num = (in_data > min_feature_value).astype(int).sum(axis=1)
        sample_range = validate_num.sort_values(ascending=False).head(int(validate_num.shape[0]*self.sample_ratio)).index
        train_sample = in_data.loc[sample_range]
        test_sample = pd.concat([in_data, train_sample]).drop_duplicates(keep=False)
        X_test_2 = test_sample.values
        y_test_2 = np.array(test_sample.index.tolist())
        print(f"select samples from {sample_range}")

        if self.use_cross_valid:
            kf = KFold(n_splits=kf_num)
            for kf_id, (train_index, test_index) in tqdm(enumerate(kf.split(train_sample))):
                prefix = f"seed_list-{kf_id+1}-dataset_list-{dataname}"
                X_train = train_sample.values[train_index]
                y_train = np.array(index_list)[train_index]
                X_test = train_sample.values[test_index]
                y_test = np.array(index_list)[test_index]
                data_dict = {
                    'X_train':X_train, 'y_train':y_train,
                    'X_test':np.concatenate([X_test, X_test_2]), 'y_test':np.concatenate([y_test, y_test_2]),
                }
                np.save(
                    os.path.join(saved_dir, f'{prefix}.npy'),
                    data_dict
                )
        else:
            # rand_seed_list = [random.randint(1, 100) for i in range(kf_num)]
            rand_seed_list = [i for i in range(kf_num)]
            for kf_id, seed_value in tqdm(enumerate(rand_seed_list)):
                prefix = f"seed_list-{kf_id+1}-dataset_list-{dataname}"
                X_train, X_test, y_train, y_test = train_test_split(
                    train_sample, sample_range,
                    test_size=0.1, shuffle=True,
                    random_state=seed_value
                )
                data_dict = {
                    'X_train':X_train, 'y_train':y_train,
                    'X_test':np.concatenate([X_test, X_test_2]), 'y_test':np.concatenate([y_test, y_test_2]),
                }
                np.save(
                    os.path.join(saved_dir, f'{prefix}.npy'),
                    data_dict
                )


def random_shuffle_all_features(samples):
    feature_index = samples.columns.tolist()

    random.shuffle(feature_index)
    fake_samples=samples[feature_index]

    return fake_samples

def random_shuffle_features(samples, topk):
    feature_name = samples.columns.tolist()
    raw_feature_name = copy(feature_name)
    no_nan_name = samples.dropna(axis=1, how='all').columns.tolist()

    raw_topk_no_nan_name = no_nan_name[topk[0]: topk[1]]
    shuffle_topk_no_nan_name = copy(raw_topk_no_nan_name)
    random.shuffle(shuffle_topk_no_nan_name)

    for i in range(len(raw_topk_no_nan_name)):
        raw_index = feature_name.index(raw_topk_no_nan_name[i])
        feature_name[raw_index] = shuffle_topk_no_nan_name[i]
    fake_samples = samples[feature_name]
    fake_samples.columns = raw_feature_name
    return fake_samples

def generate_fake_data(data_name, fake_data, shuffle_ratio, repeat_time):
    print(f"process {data_name}-{shuffle_ratio}-{repeat_time}")
    feat_num = fake_data.shape[1]
    # selected_files = os.path.join(fake_data_dir, f'raw-shuffle_ratio-{str(shuffle_ratio)}-repeat_time-{str(repeat_time)}.csv')
    # selected_values.to_csv(selected_files)

    C = random.randint(0, int(feat_num * (1-shuffle_ratio)))
    random_feat_index = (C, C+int(feat_num * shuffle_ratio)-1)

    fake_values = random_shuffle_features(fake_data, random_feat_index)
    # fake_values = random_shuffle_all_features(selected_values)

    print(f"process {data_name}-{repeat_time} done")
    return fake_values, f"fake-shuffle_ratio-{str(shuffle_ratio)}-repeat_time-{str(repeat_time)}"