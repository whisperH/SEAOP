import os
import logging

logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import sys

sys.path.append("SEAOP")
sys.path.append("/")


from DetectOutlier.dataset.data_generator import CancerDataGenerator
from DetectOutlier.utils.parameters import get_hyper_para, setup, prefix_str2para
from multiprocessing import Pool
import multiprocessing
from train_SEAOP import RunPipeline

def main():
    # run the above pipeline for reproducing the results in the paper
    command_line_args = get_hyper_para()
    print("Command Line Args:", command_line_args)
    args = setup(command_line_args)

    # reset log_dir and dataset_list
    pipeline = RunPipeline(args, infer_flag=True)
    data_generator = CancerDataGenerator(args)

    # Step 1: train selected models contained in SEAOP
    model_select_file = os.path.join(args.logs_dir, 'outliers', 'model_selection.npy')
    model_mask = np.load(model_select_file, allow_pickle=True).item()
    experiment_params = []
    model_list = set()
    for ipara_str in model_mask[args.validateModelonFake]:
        para = prefix_str2para(ipara_str, dataset_list=args.infer_datafile.split(".csv")[0])
        experiment_params.append(para)
        model_list.add(para['model_name'])
    # Step 2: train selected single model in SEAOP
    if args.runInferSingle:
        # S2.1 scale data and split into train and test set
        scale_data = data_generator.preprocessor(data_dir=args.data_dir, dataset_file=args.infer_datafile, save_scale_data=False)
        data_generator.generator(
            dataset=scale_data,
            kf_num=args.seed_nums,
            saved_dir=args.prepare_infer_data_dir,
            dist_type=args.dist_type,
            dataset_file=args.infer_datafile
        )
        # S2.2 train models with different parameters in SEAOP
        if args.multiple_thread:
            num_cpu_using = int(multiprocessing.cpu_count() * 0.5)
            print("num cores using:", num_cpu_using)
            pool = Pool(processes=num_cpu_using)
            start_time = time.time()
            for i, ipara in enumerate(experiment_params):
                pool.apply_async(pipeline.runTrain, args=(ipara,))
            print('Waiting for all subprocesses done...')
            pool.close()
            pool.join()
            end_time = time.time()
            print(
                f"while multiple_thread is {args.multiple_thread}, "
                f"outliers detection finish with {end_time-start_time}"
            )
        else:
            cost_time = {
                "model_index": [],
                "model_name": [],
                "cost": [],
            }
            for i, ipara in tqdm(enumerate(experiment_params)):
                start_time = time.time()
                cost_time["model_name"].append(ipara["model_name"])
                try:
                    pipeline.runTrain(ipara)
                except Exception as e:
                    print(e)
                end_time = time.time()
                print(
                    f"while multiple_thread is {args.multiple_thread}, "
                    f"{ipara} outliers detection finish with {end_time-start_time}"
                )

                cost_time["model_index"].append(i)
                cost_time["cost"].append(end_time-start_time)
            pd.DataFrame(cost_time).to_csv(
                os.path.join(pipeline.logs_dir, 'single_model_cost.csv')
            )

    if args.validateFakeData:
        fake_data = pd.read_csv(os.path.join(args.simulated_data_dir, args.infer_datafile))
        gt_sample_name = list(fake_data.values[:, 0])
        # select fake data from args.simulated_data_dir
        pipeline.runEnsembleTest(
            args, args.infer_datafile, model_list,
            validateFakeData=True,
            model_name_str=model_mask[args.validateModelonFake],
            gt_sample_name=gt_sample_name,
        )


    if args.runBoostTest:
        raw_data = pd.read_csv(os.path.join(args.data_dir, args.infer_datafile), index_col=0)
        idata_name=args.infer_datafile.split(".")[0]
        print(f"model list is {model_list}")
        pipeline.runEnsembleTest(args, idata_name, model_list)

        sub_outliers = pd.read_csv(
            os.path.join(pipeline.logs_dir, f"{idata_name}_final_result.csv"), index_col=0
        )
        if sub_outliers.shape[1] == 0:
            clean_data = raw_data
        else:
            mask = ~raw_data.index.isin(sub_outliers.values[0])
            clean_data = raw_data[mask]
        clean_data.to_csv(os.path.join(pipeline.logs_dir, f'{idata_name}_clean_result.csv'))
        print(f"clean data is saved in {pipeline.logs_dir}")

if __name__ == '__main__':
    main()