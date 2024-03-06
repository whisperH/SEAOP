import pandas as pd
import numpy as np
import os, sys
sys.path.append("run_OD")
sys.path.append("../")
sys.path.append("./")

from DetectOutlier.utils.file_io import PathManager

def remove_header(dataset_name, column_list=None):
    if 'HCC' in dataset_name:
        remove_list = ["ID","Protein.IDs","Majority.protein.IDs","Protein.names","Gene.names", "iBAQ"]
    elif 'LADC' in dataset_name:
        remove_list = ["Protein.IDs","Majority.protein.IDs","Protein.names","Gene.names", "iBAQ"]
    elif 'HelaGroups' in dataset_name:
        remove_list = ["iBAQ peptides", "iBAQ"]
        for i in column_list:
            if 'iBAQ' not in i:
                remove_list.append(i)
    elif 'Hela.train' in dataset_name:
        remove_list = []
    else:
        raise NotImplementedError
    return remove_list


def data_description(X, save_dir=None):
    import matplotlib.pyplot as plt
    des_dict = {}

    # des_dict['Samples'] = X.shape[0]
    des_dict['Features'] = X.shape[1]
    # missing_info = X.isna().sum(axis=1)/X.shape[1]
    missing_info = (X == 0).astype(int).sum(axis=1)/X.shape[1]
    des_dict.update(
        {"missing_"+k: v for k, v in missing_info.describe().to_dict().items()}
    )
    plot_data = {k: v for k, v in missing_info.items() if k not in ['count', 'missing_count']}
    plot_data = pd.Series(plot_data)
    plot_data.plot.box()
    if save_dir is not None:
        plt.savefig(f"{save_dir} missing_info.png", dpi=300)
        plt.close()

    print("="*100)
    print(des_dict)
    print("="*100)

if __name__ == '__main__':
    # dataset_list_raw = ['HCC.csv', 'HelaGroups.txt', 'LADC.csv', 'Hela.train.csv']
    dataset_list_raw = ['HCC.csv', 'LADC.csv']
    raw_data_dir = "../datasets/Raw"
    check_data_dir = "../datasets/Format"
    if not PathManager.exists(check_data_dir):
        PathManager.mkdirs(check_data_dir)

    for dataset in dataset_list_raw:
        print(f"processing {dataset}")
        if 'csv' in dataset:
            raw_data = pd.read_csv(os.path.join(raw_data_dir, dataset), index_col=0)
        elif 'txt' in dataset:
            raw_data = pd.read_csv(os.path.join(raw_data_dir, dataset),
                                   index_col=0, sep='\t', low_memory=False)
        else:
            raise Exception('Unknown file type, please transform them to txt or csv')

        remove_feature_name = remove_header(dataset, raw_data.columns.tolist())

        Protein_ID = raw_data['Protein.IDs']
        raw_data = raw_data.drop(columns=remove_feature_name)
        raw_data.replace(0,np.nan, inplace=True)
        # data_description(raw_data, check_data_dir)
        save_data = raw_data.T
        save_data.columns = Protein_ID
        save_data.to_csv(os.path.join(check_data_dir, dataset.split(".")[0] + '.csv'))