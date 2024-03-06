import pandas as pd
import os
import sys
import random
import numpy as np
from copy import copy
sys.path.append("run_OD")
sys.path.append("./")
from DetectOutlier.utils.file_io import PathManager

random_seed = 42
repeat_times = 10


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


# if __name__ == '__main__':
#     random.seed(random_seed)
#     from DetectOutlier.utils.parameters import get_hyper_para, setup
#     command_line_args = get_hyper_para()
#     print("Command Line Args:", command_line_args)
#     args = setup(command_line_args)
#     data_name = args.dataset_list_raw[0].split(".")[0]
#
#     format_data_dir = args.logs_dir
#     fake_data_dir=args.logs_dir
#
#     if not PathManager.exists(fake_data_dir):
#         PathManager.mkdirs(fake_data_dir)
#
#     clean_data = pd.read_csv(
#         os.path.join("./datasets/Format", f'{data_name}_clean_result.csv'),
#         index_col=0
#     )
#
#     for repeat_time in range(repeat_times):
#         generate_fake_data(
#             data_name,
#             clean_data,
#             fake_ratio,
#             shuffle_ratio,
#             repeat_time,
#             fake_data_dir
#         )