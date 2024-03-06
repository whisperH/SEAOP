import os
import logging;

logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd

import gc
from tqdm import tqdm
import sys
sys.path.append("SEAOP")
sys.path.append("/")

import joblib

from DetectOutlier.dataset.data_generator import CancerDataGenerator, generate_fake_data
from DetectOutlier.model.PyOD import ModelFactory
from DetectOutlier.utils.utility import dict_combinations, normalization
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from DetectOutlier.utils.file_io import PathManager, load_all_outlierfiles, load_outliervalues_from_npy, save_results
from DetectOutlier.utils.parameters import get_hyper_para, setup, check_hyper_para, para2prefix_str, prefix_str2para, get_fake_dataname, get_fake_suffix
from scipy.stats import chi2_contingency, chisquare
from visualization.visulize import plot_model_acc_trend, plot_model_trend_pdf, get_combination_result, visualize_num_bar, visualize_all_single_bar
from itertools import combinations
from multiprocessing import Process, Manager, Pool
import multiprocessing
import time
import pymannkendall as mk
from collections import Counter
import random
from copy import copy

class RunPipeline():
    def __init__(self, args, infer_flag=False):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        '''

        self.save_model = args.save_model
        self.input_matrix = args.input_matrix
        # self.outliers_fraction = args.outliers_fraction

        # global parameters
        self.generate_duplicates = True

        self.seed_list = list(np.arange(args.seed_nums) + 1)

        if infer_flag:
            self.train_logs_dir = args.logs_dir
            self.logs_dir = args.infer_log_dir
            self.prepare_data_dir = args.prepare_infer_data_dir
            self.dataset_list = args.infer_datafile
            # args.runInferSingle = True
            # args.runBoostTest = True
        else:
            self.logs_dir = args.logs_dir
            self.prepare_data_dir = args.prepare_data_dir
            self.dataset_list = args.dataset_list_raw




    # model fitting function
    def model_fit(self, clf, X_train, X_test, y_train, y_test, corr_data=None, para=None, **kwargs):
        idataset_name = kwargs.get('dataset_name', '')
        prefix = para2prefix_str(para)
        start_time = time.time()

        # try:
        clf = clf.fit(X=np.array(X_train))
        end_time = time.time()
        time_fit = end_time - start_time

        if self.save_model:
            if not PathManager.exists(os.path.join(self.logs_dir, "model_weights")):
                PathManager.mkdirs(os.path.join(self.logs_dir, "model_weights"))
            joblib.dump(
                clf, os.path.join(
                    self.logs_dir, "model_weights", f"{prefix}.joblib"
                )
            )

        # predicting score (inference)
        start_time = time.time()
        X = np.r_[X_train, X_test]
        Y = np.r_[y_train, y_test]
        y_pred, confidence = clf.predict(X, return_confidence=True)
        scores_pred = clf.decision_function(X)
        outlier_thre = clf.threshold_
        end_time = time.time()
        time_inference = end_time - start_time
        print(f"time fit is {time_fit}")
        print(f"time inference is {time_inference}")

        norm_scores_pred = normalization(scores_pred)
        save_results(X, Y, y_pred, scores_pred, norm_scores_pred, confidence,
                          prefix, outlier_thre, para, logs_dir=self.logs_dir, dataset_name=idataset_name)
        del clf
        gc.collect()

        # except Exception as error:
        #     print(f'Error in model fitting. Model:{prefix}, Error: {error}')
        #     exit(0)
        return 0

    # run the experiment
    def runTrain(self, model_para):
        # for i, model_para in tqdm(enumerate(self.experiment_params)):
        model_name = model_para['model_name']
        dataset_name = model_para['dataset_list'].split(".")[0]

        para_bak = model_para.copy()
        prefix = f"seed_list-{model_para['seed_list']}-dataset_list-{dataset_name}"

        del model_para['model_name']
        del model_para['seed_list']
        del model_para['dataset_list']
        # del model_para['dist_list']
        # load dataset
        # try:
        # corr_data_file = os.path.join(self.prepare_data_dir, f"dataset_list_{dataset_name}_corr.csv")
        # corr_data = pd.read_csv(corr_data_file, index_col=0)

        data_pair_file = os.path.join(self.prepare_data_dir, f"{prefix}.npy")
        data = np.load(data_pair_file, allow_pickle=True).item()

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        # except Exception as error:
        #     print(f'Error when generating data: {error}')
        #     exit(-1)
        self.model_fit(
            clf=ModelFactory(model_name, model_para),
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            # corr_data=corr_data,
            para=para_bak, dataset_name=dataset_name
        )

    def runInfer(self, model_weights_dir, model_file, infer_data, train_dataname, infer_name, gt_label):
        '''
        单模型做推理
        Args:
            model_weights_dir: 模型的权重文件路径
            model_file: 模型权重文件
            infer_data: 需要做推理的数据
            train_dataname:
            infer_name:

        Returns:

        '''
        para= prefix_str2para(model_file)
        para['dataset_list']=infer_name
        model = joblib.load(os.path.join(model_weights_dir, model_file))
        prefix = model_file.replace(train_dataname, infer_name).replace(".joblib", "")
        X = infer_data.values
        Y = np.array(infer_data.index.tolist())
        outlier_thre = model.threshold_
        y_pred, confidence = model.predict(X, return_confidence=True)
        scores_pred = model.decision_function(X)

        norm_scores_pred = normalization(scores_pred)
        save_results(
            X, Y, y_pred, scores_pred, norm_scores_pred, confidence,
            prefix, outlier_thre, para, logs_dir=self.logs_dir, dataset_name=infer_name, gt_label=gt_label
        )

    def runEnsembleTest(self, args, datafile, model_list, gt_sample_name=(), validateFakeData=False, **kwargs):
        '''
        所有单模型的结果在做集成并经过chi square检验
        Args:
            datafile:
            gt_sample_name:

        Returns:

        '''
        dataname = datafile.split(".csv")[0]
        if validateFakeData:
            assert "model_name_str" in kwargs, "please provide selected model_name_str when validating fake data"
            model_name_str = kwargs['model_name_str']
            model_name_str_list = []
            for i in list(model_name_str):
                model_name_str_list.append(f"{i+get_fake_suffix(dataname)}.npy")
            npy_list = load_all_outlierfiles(self.train_logs_dir, dataname, model_name_str_list)
        else:
            npy_list = load_all_outlierfiles(self.logs_dir, dataname)

        outlier_ids = {}
        outlier_scores = {_:[] for _ in model_list}
        all_ids = {
            _: {} for _ in model_list
        }
        contribute_list = {
            "model_name": [],
            "hyper_list": [],
            "sample_name": [],
            "classify": [],

        }
        print("***" * 30)
        print("loading outliers detection result")
        # dict{
        #   dataset_name: {
        #       outlier_ids: {
        #           model_name: {
        #               outlier: num,
        #               normal: num,
        #           }
        #       }
        #   }
        # }
        for outliers_ifile in tqdm(npy_list):
            outliers_res_dict = load_outliervalues_from_npy(outliers_ifile, dataname)
            normal_idlist = outliers_res_dict['normal_idlist']
            outlier_idlist = outliers_res_dict['outlier_idlist']
            norm_normal_scorelist = outliers_res_dict['norm_normal_scorelist']
            norm_outlier_scorelist = outliers_res_dict['norm_outlier_scorelist']
            model_name = outliers_res_dict['model_name']

            if dataname not in outlier_ids:
                outlier_ids[dataname] = {}

            for idname, iscore in zip(
                    outlier_idlist + normal_idlist,
                    norm_outlier_scorelist + norm_normal_scorelist
            ):
                if model_name not in model_list:
                    continue
                if idname not in all_ids[model_name]:
                    all_ids[model_name][idname] ={
                            "outlier": 0,
                            "normal": 0,
                            "outlier_score": [],
                            "inlier_score": []
                        }

                if idname not in outlier_ids[dataname]:
                    outlier_ids[dataname][idname] = {_: {
                        "outlier": 0,
                        "normal": 0
                    } for _ in model_list
                    }

                if idname in outlier_idlist:
                    outlier_ids[dataname][idname][model_name]['outlier'] += 1
                    all_ids[model_name][idname]['outlier'] += 1
                    all_ids[model_name][idname]['outlier_score'].append(iscore)
                    outlier_scores[model_name].append(iscore)
                    contribute_list['classify'].append('outlier')
                else:
                    outlier_ids[dataname][idname][model_name]['normal'] += 1
                    all_ids[model_name][idname]['normal'] += 1
                    all_ids[model_name][idname]['inlier_score'].append(iscore)
                    contribute_list['classify'].append('inlier')
                contribute_list['model_name'].append(model_name)
                contribute_list['hyper_list'].append(outliers_ifile)
                contribute_list['sample_name'].append(idname)


        np.save(
            os.path.join(self.logs_dir, f'{dataname}_outliers_stat.npy'),
            all_ids
        )

        print("***" * 30)
        print("integrate all outliers from subsampling dataset")
        outlier_integrated_id = {}
        all_integrated_id = {}

        # 以sample name为key统计每个样本出现异常和正常的次数
        start_time = time.time()
        for idataset_name, sample_info in outlier_ids.items():
            # all_data = self.data_generator.generator(dataset=idataset_name, seed=0)
            # id_pair_list = all_data['id_pair_list']
            print("=" * 50)
            print(f"In dataset: {idataset_name}")
            print("=" * 50)

            if idataset_name not in outlier_integrated_id:
                outlier_integrated_id[idataset_name] = []
                all_integrated_id[idataset_name] = []

            for sample_name, outlier_info in sample_info.items():
                sample_df = pd.DataFrame(outlier_info).T
                sample_sum_df = sample_df.sum().values.tolist()

                if isinstance(sample_name, str):
                    sample_name = [sample_name]
                for ip_name in sample_name:
                    # if ip_name not in outlier_integrated_id[idataset_name]:
                    #     outlier_integrated_id[idataset_name][ip_name] = {'num': 0}
                    try:
                        # https://zhuanlan.zhihu.com/p/56423752
                        # 求所有模型中outlier和normal的总和
                        chi2, p = chisquare(sample_sum_df)
                        outlier_res_dict = {
                            'name': ip_name,
                            'outlier_num': sample_df['outlier'].sum(),
                            'normal_num': sample_df['normal'].sum(),
                            'p_value': p,
                        }
                        # 原假设：The two categorical variables have relationship
                        if sample_df['normal'].sum() < sample_df['outlier'].sum():
                            # percent = sample_df['outlier'].sum() / (sample_df['normal'].sum()+sample_df['outlier'].sum())
                            # 如果异常比例大于设定的阈值
                            # if percent >= self.accept_percent:
                            # print(percent)
                            all_integrated_id[idataset_name].append(outlier_res_dict)
                            if p < 0.05:
                                outlier_integrated_id[idataset_name].append(outlier_res_dict)
                                conclusion = "reject the null Hypothesis."
                            else:
                                conclusion = "Accept the null Hypothesis."
                            # print(sample_name)
                            # print(sample_df)
                            # print(conclusion)
                        else:
                            continue

                            # print(sample_name)
                            # print(sample_df)
                    except Exception as e:
                        if (sample_df['outlier'] == 0).all():
                            # print(f"in {sample_name}: outlier array is empty")
                            continue
                        elif (sample_df['normal'] == 0).all():
                            print(f"in {sample_name}: normal array is empty")
                            outlier_res_dict = {
                                'name': ip_name,
                                'outlier_num': sample_df['outlier'].sum(),
                                'normal_num': 0,
                                'p_value': 0,
                            }
                            outlier_integrated_id[idataset_name].append(outlier_res_dict)
                            all_integrated_id[idataset_name].append(outlier_res_dict)
                        else:
                            print(e)
                            # print(sample_df)
                            # print(sample_name)
                            exit(3)
            if not outlier_integrated_id[idataset_name]:
                print(f"not outliers in {idataset_name}")

            print("show normal\\abnormal ratio of each candidates")

            # =================================================================== #
        end_time = time.time()
        print(f"ensemble test is finished with time: {end_time-start_time}")


        _all_integrated_id = []
        _outlier_integrated_id = []
        for idataset_name in all_integrated_id.keys():
            _all_integrated_id.extend(all_integrated_id[idataset_name])
            _outlier_integrated_id.extend(outlier_integrated_id[idataset_name])

        df_outlier_integrated_id = pd.DataFrame(_outlier_integrated_id)
        df_outlier_integrated_id.T.to_csv(os.path.join(self.logs_dir, f'{dataname}_final_result.csv'))
        if len(gt_sample_name) > 0:
            print(f"Accuracy on {dataname} is {df_outlier_integrated_id.shape[0]/len(gt_sample_name)}")
        if len(_outlier_integrated_id) > 0:
            contribute_df = pd.DataFrame(contribute_list)
            pseudo_error_idx = contribute_df['sample_name'].isin(df_outlier_integrated_id['name'])
            pseudo_error_df = contribute_df[pseudo_error_idx]
            visualize_all_single_bar(pseudo_error_df, os.path.join(self.logs_dir, f'{dataname}_single_model_res.png'))
            pseudo_error_df.to_csv(os.path.join(self.logs_dir, f'{dataname}_pseudo_error.csv'))

        df_all_integrated_id = pd.DataFrame(_all_integrated_id).T
        df_all_integrated_id.to_csv(os.path.join(self.logs_dir, f'{dataname}_All_result.csv'))

        print(f"final_result is saved in {self.logs_dir}")
        print("***" * 30)
        print("visualizing the results...")
        if args.visualize_stat:
            PathManager.mkdirs(os.path.join(self.logs_dir, 'decision', dataname))
            visualize_num_bar(
                _all_integrated_id,
                os.path.join(self.logs_dir, f'{dataname}_single_model_Chi_Test.png'),
                mark_list_info=_outlier_integrated_id
            )

            if not args.multiple_thread:
                # for outliers_ifile in npy_list:
                #     draw_raidal_Tree(
                #         outliers_ifile, _outlier_integrated_id
                #     )
                combination_list = list(combinations(model_list, 2))

                for model_name1, model_name2 in combination_list:
                    save_path = os.path.join(self.logs_dir, 'decision', dataname, f'{model_name1}_{model_name2}_join.pdf')
                    get_combination_result(
                        all_ids, [model_name1, model_name2], outlier_scores,
                        _outlier_integrated_id, save_path, gt_sample_name=gt_sample_name
                    )
                # draw_3DSurface(outliers_ifile, _outlier_integrated_id)
            else:
                num_cpu_using = int(multiprocessing.cpu_count() * 0.7)
                print("num cores using:", num_cpu_using)
                vis_pool = Pool(processes=num_cpu_using)
                # for outliers_ifile in npy_list:
                # vis_pool.apply_async(
                #     draw_raidal_Tree, args=(
                #         outliers_ifile, _outlier_integrated_id
                #     )
                # )
                combination_list = list(combinations(model_list, 2))
                for idx, (model_name1, model_name2) in enumerate(combination_list):
                    save_path = os.path.join(self.logs_dir, 'decision', dataname, f'{model_name1}_{model_name2}_join.pdf')
                    vis_pool.apply_async(
                        get_combination_result, args=(
                            all_ids, [model_name1, model_name2], outlier_scores,
                            _outlier_integrated_id, save_path, gt_sample_name
                        )
                    )
                print('Waiting for all subprocesses done...')
                vis_pool.close()
                vis_pool.join()
        return 0


def set_model_para(args, seed_list, dataset_list):
    # from pyod
    experiment_params = []
    for model_settings in args.models:
        combine_value_list = []
        combine_name_list = []

        model_name = list(model_settings.keys())[0]
        model_para = list(model_settings.values())[0]
        for ipara in model_para:
            combine_name_list.append(list(ipara.keys())[0])
            combine_value_list.append(list(ipara.values())[0])
        # combine_name_list.append('dist_list')
        # combine_value_list.append(args.dist_list)
        combine_name_list.append('seed_list')
        combine_value_list.append(seed_list)
        combine_name_list.append('dataset_list')
        combine_value_list.append(dataset_list)
        retList = dict_combinations(combine_value_list)
        for icombine in retList:
            temp_ = {"model_name": model_name}
            for para_name, para_value in zip(combine_name_list, icombine):
                temp_.update({para_name: para_value})
            experiment_params.append(temp_)
    return experiment_params

def main():
    # run the above pipeline for reproducing the results in the paper
    command_line_args = get_hyper_para()
    print("Command Line Args:", command_line_args)
    args = setup(command_line_args)
    check_hyper_para(args)

    pipeline = RunPipeline(args)
    data_generator = CancerDataGenerator(args)
    dataname = args.dataset_list_raw[0].split(".csv")[0]
    model_type_list = []
    for _ in args.models:
        _model_name = list(_.keys())[0]
        if _model_name == "KNN":
            model_type_list += ['KNN', 'AvgKNN']
        else:
            model_type_list.append(_model_name)
    # S0.1 scale data and split into train and test set
    if args.data_prepare:
        for dataset_file in args.dataset_list_raw:
            data_generator.preprocessor(data_dir=args.data_dir, dataset_file=dataset_file)
            data_generator.generator(
                dataset=dataset_file,
                kf_num=args.seed_nums,
                saved_dir=args.prepare_data_dir,
                dist_type=args.dist_type
            )
            print(f"save pair data {dataset_file} with {args.seed_nums} folder in {args.prepare_data_dir}")
    # S0.2 combine all parameters
    experiment_params = set_model_para(
        args, pipeline.seed_list,
        pipeline.dataset_list
    )
    print(f"ensemble model nums:{len(experiment_params)}")

    # Step 1: train all models contained in SEAOP
    if args.runSingleOD:
        print("==================================== step1: training single model ====================================")
        # S1.1 train models with different parameters in SEAOP
        if args.multiple_thread:
            num_cpu_using = int(multiprocessing.cpu_count() * 0.7)
            print("num cores using:", num_cpu_using)
            pool = Pool(processes=num_cpu_using)

            for i, ipara in enumerate(experiment_params):
                pool.apply_async(pipeline.runTrain, args=(ipara,))
            print('Waiting for all subprocesses done...')
            pool.close()
            pool.join()

        else:
            for i, ipara in tqdm(enumerate(experiment_params)):
                pipeline.runTrain(ipara)
        print("outliers detection finish")


    # Step 2: generate simulated data via feature shuffle with different threshold
    if args.runFakeData:
        print("==================================== step2: generate fake data ====================================")
        # s2.1 remove them from scaled dataset
        scale_data = pd.read_csv(
            os.path.join(args.prepare_data_dir, f"{dataname}.csv"), index_col=0
        )

        # # s2.2 get all candidate outliers
        # assert os.path.isdir(os.path.join(pipeline.logs_dir, "outliers", dataname)), 'please get results of single model'
        # npy_list = load_all_outlierfiles(pipeline.logs_dir, dataname)
        # outlier_candidate_pool = []
        # for outliers_ifile in tqdm(npy_list):
        #     outliers_res_dict = load_outliervalues_from_npy(outliers_ifile, dataname)
        #     outlier_idlist = outliers_res_dict['outlier_idlist']
        #     outlier_candidate_pool += outlier_idlist
        # outlier_candidate_pool = list(set(outlier_candidate_pool))

        print("load from external file")
        outlier_candidate_pool = []
        for i in args.external_outlier_candidates:
            outlier_candidate_pool += list(i.values())[0]
        outlier_candidate_pool = list(set(outlier_candidate_pool))

        if len(outlier_candidate_pool) == 0:
            candidate_clean_data = scale_data
        else:
            mask = ~scale_data.index.isin(outlier_candidate_pool)
            candidate_clean_data = scale_data[mask]
        candidate_clean_data.to_csv(os.path.join(args.simulated_data_dir, f'{dataname}_candidate_clean.csv'))
        # s2.2 simulated data with shuffle features
        fake_datalist = []
        fake_dataname_list = []
        fake_gt_labels = []

        fake_num = candidate_clean_data.shape[0]
        group_fake_num = fake_num // args.fake_groups

        for shuffle_ratio in args.feature_shuffle_ratio:
            for repeat_time in range(args.fake_repeats):
                if repeat_time % args.fake_groups == 0: # start from 0
                    # reset random_seed
                    sample_index = [_ for _ in range(fake_num)]
                    random.seed(repeat_time)
                    random.shuffle(sample_index)

                fake_data = copy(candidate_clean_data)
                selected_values = fake_data.iloc[sample_index][
                                  (repeat_time%args.fake_groups)*group_fake_num:
                                  (repeat_time%args.fake_groups+1)*group_fake_num
                                  ]
                fake_files = os.path.join(
                    args.simulated_data_dir,
                    f'{get_fake_dataname(shuffle_ratio, repeat_time)}.csv'
                )

                fake_data, fake_file_name = generate_fake_data(
                    dataname,
                    selected_values,
                    shuffle_ratio,
                    repeat_time
                )
                fake_data.to_csv(fake_files)

                fake_datalist.append(fake_data)
                fake_dataname_list.append(fake_file_name)
                fake_gt_labels.append([0]*fake_data.shape[0])

    # Step 3: test single model on fake data
    if args.runSingleInfer:
        print("==================================== step3: inferring single model on fake data ====================================")
        # s3.1 : load infer data: fake data, load data from Step 2 or from args.simulated_data_dir
        if not args.runFakeData:
            fake_datalist = []
            fake_dataname_list = []
            fake_gt_labels = []
            for ifake_file in os.listdir(args.simulated_data_dir):
                if "fake" in ifake_file:
                    fake_data = pd.read_csv(os.path.join(args.simulated_data_dir, ifake_file), index_col=0)
                    fake_datalist.append(fake_data)
                    fake_dataname_list.append(ifake_file)
                    fake_gt_labels.append([0]*fake_data.shape[0])

        assert len(fake_datalist) != 0, 'no fake data'
        assert args.save_model, 'save_model must be True'

        # s3.2 : load model parameters
        model_weights_dir = os.path.join(pipeline.logs_dir, "model_weights")
        model_weights_list = os.listdir(model_weights_dir)
        assert len(model_weights_list)>0, 'model weights are not exist'

        # s3.3 : scale and infer fake data
        for ifake_data, ifake_name, ifake_gt_label in zip(fake_datalist, fake_dataname_list, fake_gt_labels):
            # train_dataname = args.dataset_list_raw[0].split(".csv")[0]
            infer_dataname = ifake_name.split(".csv")[0]

            if args.multiple_thread:
                num_cpu_using = int(multiprocessing.cpu_count() * 0.7)
                print("num cores using:", num_cpu_using)
                pool = Pool(processes=num_cpu_using)

                for i, model_file in enumerate(model_weights_list):
                    pool.apply_async(pipeline.runInfer, args=(
                        model_weights_dir, model_file,
                        ifake_data, dataname,
                        infer_dataname, ifake_gt_label, ) # outliers results are saved in infer_dataname
                    )
                print('Waiting for all subprocesses done...')
                pool.close()
                pool.join()

            else:
                for i, model_file in tqdm(enumerate(model_weights_list)):
                    pipeline.runInfer(
                        model_weights_dir, model_file,
                        ifake_data, dataname,
                        infer_dataname, ifake_gt_label
                    )
        print("inferring outliers finished by single model")

    # Step 4: infer model on fake data
    if args.runFakeStat:
        print("==================================== step4: Statistic fake data ====================================")

        det_results = {}

        for shuffle_ratio in args.feature_shuffle_ratio:
            model_para_name = {}
            # load all outliers with repeat_times
            for repeat_time in tqdm(range(args.fake_repeats)):
                fake_dataname = get_fake_dataname(shuffle_ratio, repeat_time)
                npy_list = load_all_outlierfiles(pipeline.logs_dir, fake_dataname)
                # calculate all single model's accuracy
                for outliers_ifile in npy_list:
                    outliers_res_dict = load_outliervalues_from_npy(outliers_ifile, fake_dataname)
                    gt_label = outliers_res_dict['gt_label']
                    assert len(gt_label) > 0, "no gt_label with fake data"
                    # imodel_para_name should ignore the repeat times
                    imodel_para_name = outliers_res_dict['imodel_para_name'].split("-repeat_time")[0]
                    imodel_name = outliers_res_dict['imodel_para_name'].split("-shuffle_ratio-")[0]
                    outlier_idlist = outliers_res_dict['outlier_idlist']
                    normal_idlist = outliers_res_dict['normal_idlist']

                    if imodel_name not in det_results:
                        det_results[imodel_name] = {
                                'mean_acc': [],
                                'se_acc': [],
                                'sd_acc': [],
                                'cv_acc': [],
                                'shuffle_ratio': [],
                        }
                    if imodel_para_name not in model_para_name:
                        model_para_name[imodel_para_name] = {
                            'acc': [],
                        }
                    label_normal_idlist = [1] * len(normal_idlist)
                    label_outlier_idlist = [0] * len(outlier_idlist)
                    # roc = roc_auc_score(gt_label, norm_normal_scorelist+norm_outlier_scorelist)
                    acc = accuracy_score(gt_label, label_normal_idlist+label_outlier_idlist)
                    # model_para_name[imodel_para_name]['roc'].append(roc)
                    model_para_name[imodel_para_name]['acc'].append(acc)

            # calculate sd of each model with different parameters
            for id, (k, v) in enumerate(model_para_name.items()):
                imodel_name = k.split("-shuffle_ratio-")[0]
                mean_acc = np.mean(v['acc'])
                sd_acc = np.std(v['acc'])
                se = np.std(v['acc'], ddof=1) / np.sqrt(np.size(v['acc']))
                if sd_acc == 0:
                    cv = 0
                else:
                    cv = sd_acc / mean_acc
                det_results[imodel_name]['mean_acc'].append(mean_acc)
                det_results[imodel_name]['sd_acc'].append(sd_acc)
                det_results[imodel_name]['se_acc'].append(se)
                det_results[imodel_name]['cv_acc'].append(cv)
                det_results[imodel_name]['shuffle_ratio'].append(shuffle_ratio)
        np.save(
            os.path.join(pipeline.logs_dir, 'outliers', 'model_result_stat.npy'),
            det_results
        )

    # Step 5: select model and parameters
    if args.runModelSelect:
        print("==================================== step5: Model Parameters selection ====================================")
        if args.runFakeStat is False:
            det_results = np.load(
                os.path.join(args.logs_dir, "outliers", "model_result_stat.npy"),
                allow_pickle=True
            ).item()

        if args.visualize_stat:
            for iindex in ["cv_acc", "mean_acc"]:
                # plot_model_acc_trend(
                plot_model_trend_pdf(
                    model_type_list, det_results,
                    iindex,
                    os.path.join(args.logs_dir, "outliers", f'contribute_{iindex}.pdf')
                )

        model_dropped = []
        model_selected = []
        model_selected_p = []
        shuffle_ratio_selected = []
        cv_selected = []
        acc_selected = []
        for idx, (model_name_para, acc_info) in enumerate(det_results.items()):
            # para = prefix_str2para(model_name_para)
            cv_acc = acc_info["cv_acc"]
            mean_acc = acc_info["mean_acc"]
            shuffle_ratio = acc_info["shuffle_ratio"]

            # rule 1: shuffle_ratio as small as possible, cv as stable as possible
            # https://github.com/mmhs013/pyMannKendall
            idx_shuffle_ratio = -1
            for i in range(len(cv_acc)):
                result = mk.original_test(cv_acc[i:], alpha=0.05)
                if result.trend == "no trend":
                    idx_shuffle_ratio = i
                    p_value = result.p
                    break
                else:
                    idx_shuffle_ratio = -1

            if idx_shuffle_ratio != -1:
                # rule 2: mean_acc as big as possible
                if mean_acc[idx_shuffle_ratio] >= args.accept_acc:
                    model_selected.append(model_name_para)
                    model_selected_p.append(p_value)
                    acc_selected.append(mean_acc[idx_shuffle_ratio])
                    shuffle_ratio_selected.append(shuffle_ratio[idx_shuffle_ratio])
                    cv_selected.append(cv_acc[idx_shuffle_ratio])
                else:
                    model_dropped.append(model_name_para)
            else:
                model_dropped.append(model_name_para)
        most_counts = Counter(shuffle_ratio_selected)
        sort_values = sorted(most_counts.items(),key=lambda x:x[1],reverse=True)[0+ args.offset][0]
        print(f"Threshold value is {sort_values}")
        # try1：选择小于0.3的所有
        mask_ = np.array(shuffle_ratio_selected)<=sort_values
        drop_mask_ = np.array(shuffle_ratio_selected)>sort_values
        # mask_ = (np.array(shuffle_ratio_selected)<=sort_values) & (np.array(shuffle_ratio_selected)>0.1)

        selected_result = {
            "model_dropped": np.hstack((np.array(model_selected)[drop_mask_], np.array(model_dropped))),
            "model_selected": np.array(model_selected)[mask_],
            "acc_selected": np.array(acc_selected)[mask_],
            "shuffle_ratio_selected": np.array(shuffle_ratio_selected)[mask_],
            "cv_selected": np.array(cv_selected)[mask_],
        }
        print(f"selected model number is {len(np.array(model_selected)[mask_])}")
        print(f"dropped model number is {len(selected_result['model_dropped'])}")
        np.save(
            os.path.join(pipeline.logs_dir, 'outliers', 'model_selection.npy'),
            selected_result
        )


if __name__ == '__main__':
    main()
