import pandas as pd
import os
from visualization.visulize import anno_scatter

cwd = os.getcwd()
print(cwd)

def compare_results(models_name, data_name):
    marker = [">", "v", "1", "s", "+", "o", "^", "h", "x", "d", "D", "H", "*"]


    x = []
    names = []
    y = []
    label = []
    shape = []
    for idx, imodel_res_name in enumerate(models_name):
        if "Hela" in data_name:
            single_outliers_file = os.path.join('../Result', "Hela" + "_" + imodel_res_name, data_name + '_All_result.csv')
        else:
            single_outliers_file = os.path.join('../Result', data_name + "_" + imodel_res_name, data_name + '_All_result.csv')
        single_outliers_info = pd.read_csv(single_outliers_file, index_col=0)
        if single_outliers_info.empty:
            continue
        single_outliers_name = single_outliers_info.loc['name'].values.tolist()
        single_outliers_pValue = single_outliers_info.loc['p_value'].values.tolist()

        if idx == 0:
            if "Hela" in data_name:
                inliers_file = os.path.join('../Result', "Hela_" + imodel_res_name, data_name + '_clean_result.csv')
            else:
                inliers_file = os.path.join('../Result', data_name + "_" + imodel_res_name, data_name + '_clean_result.csv')
            inliers_info = pd.read_csv(inliers_file)
            all_datas = inliers_info.iloc[:, 0].values.tolist()
            all_datas.extend(single_outliers_name)

        x.append([all_datas.index(_) for _ in single_outliers_name])
        names.append(single_outliers_name)
        y.append([1-round(float(_), 3) for _ in single_outliers_pValue])
        label.append(imodel_res_name)
        shape.append(marker[idx])
    save_path = os.path.join("../Result", f"{data_name}_compare_res.pdf")
    print(x)
    print(names)
    print(y)
    print(label)

    anno_scatter(x, y, label, shape, save_path)
    print(f"Saving compare_results in Result/{data_name}_compare_res.png")

if __name__ == '__main__':

    models_name = [
        'ABOD', 'CBLOF',
        'FeatureBagging', 'IFOREST',
        'LOF', 'LSCP',
        'KNN', 'AvgKNN', 'ECOD'
    ]
    data_name = "HCC"
    # data_name = "HCC_T"
    # data_name = "LADC_N"
    # models_name = [
    #     'LADC_ABOD',  'LADC_CBLOF', 'LADC_FeatureBagging', 'LADC_IFOREST',
    #     'LADC_LOF', 'LADC_LSCP'
    # ]
    # data_name = "LADC_N"

    # models_name = [
    #     'Hela_LOF','Hela_CBLOF','Hela_IFOREST','Hela_ABOD',  'Hela_FeatureBagging',
    #      'Hela_LSCP'
    # ]
    # data_name = "HelaGroups"

    compare_results(models_name, data_name)

