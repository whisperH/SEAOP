import pandas as pd
import numpy as np
import os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
sys.path.append("SEAOP")
sys.path.append("./")
sys.path.append("../")
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(current_path, "../")
from visualization.visulize import plot_model_acc_trend, plot_model_accsd_trend
from DetectOutlier.utils.parameters import get_hyper_para, setup, check_hyper_para

repeat_times = 10




def plot_contribution(model_para_name, save_path):
    fig, ax = plt.subplots(figsize=(20, 5))

    x_label = []
    y_value = []
    error_bar = []
    colors = []
    preserve_model = {}

    for id, (k, v) in enumerate(model_para_name.items()):
        if k == 'Boost':
            colors.append("#d47d6c")
            x_label.append("Ours")
        else:
            colors.append("#a9aaad")
            x_label.append(str(id))
        mean_acc = np.mean(v['acc'])

        y_value.append(mean_acc)
        error_bar.append(np.std(v['acc'], ddof=1) / np.sqrt(np.size(v['acc'])))
    ax.errorbar(range(len(x_label)), y_value, yerr=error_bar)
    # ax.bar(range(len(x_label)), y_value, align='center', color=colors)
    # for pos, y, err in zip(x_label, y_value, error_bar):
    #     ax.errorbar(pos, y, err, lw=0.2, capsize=0.5, capthick=0.1, color="#555555")


    # ax.legend(loc="best")
    ax.set_xticks(range(len(x_label)), x_label, rotation=90)
    ax.set_xlabel('Model Index NO.')
    ax.set_ylabel('Accuracy (%)')
    plt.subplots_adjust()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.show()
    return preserve_model



if __name__ == '__main__':
    command_line_args = get_hyper_para()
    print("Command Line Args:", command_line_args)
    args = setup(command_line_args)
    check_hyper_para(args)
    logs_dir = os.path.join(root_path, args.logs_dir)
    model_type_list = []
    for _ in args.models:
        _model_name = list(_.keys())[0]
        if _model_name == "KNN":
            model_type_list += ['KNN', 'AvgKNN']
        else:
            model_type_list.append(_model_name)


    result_info = np.load(os.path.join(logs_dir, "outliers", "model_result_stat.npy"), allow_pickle=True).item()
    for iindex in ["mean_acc", "sd_acc", "cv_acc"]:
        plot_model_acc_trend(
            model_type_list, result_info,
            iindex,
            os.path.join(logs_dir, "outliers", f'contribute_{iindex}.png')
        )
