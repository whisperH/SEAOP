import argparse
import re
import os.path as osp
from DetectOutlier.config import get_cfg
from DetectOutlier.utils.file_io import PathManager
from DetectOutlier.utils.data import data_config

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    opt = args.opts
    args = cfg.merge_from_file(args.config_file)

    # modify data path and fill data strategy


    assert len(opt)%2 == 0, 'opt must be pair'
    for i in range(len(opt)//2):
        key_name = opt[i*2]
        if opt[i*2+1] in ['true', 'false', "True", 'False']:
            if (opt[i*2+1] == 'true') or (opt[i*2+1] == 'True'):
                args[key_name] = True
            else:
                args[key_name] = False
        else:
            args[key_name] = opt[i*2+1]

    args = data_config(args)
    output_dir = args.logs_dir
    if output_dir:
        PathManager.mkdirs(output_dir)

    path = osp.join(output_dir, "config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(args.dump())
    print("Full config saved to {}".format(osp.abspath(path)))
    return args

def get_hyper_para():
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")

    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument(
        '-c', '--config_file', type=str,
        # default='/home/huangjinze/code/PeptideOD/configs/meta_boost.yaml'
        default='E:\\CodeBase\\SEAOP\\configs\\debug.yaml'
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    args.config_file = osp.join(working_dir, args.config_file)
    return args


def check_hyper_para(args):
    assert args.joint_data in ['original', 'separation', 'cross'], \
        'join_data must be one of original, separation, cross'
    if len(args.dataset_list_raw) == 1:
        assert args.joint_data in ['original', 'separation'], \
            'joint_data must be original or separation when using one dataset'
    if len(args.dataset_list_raw) > 1:
        assert args.joint_data in ['cross', 'separation'], \
            'joint_data must be cross or separation when using more than one dataset'

def para2prefix_str(para):
    if para is not None:
        para_str = []
        for key, value in para.items():
            if key == "dataset_list":
                value=value.split(".")[0]
            para_str.append(key)
            para_str.append(str(value))
        prefix = "-".join(para_str)
    else:
        prefix = ''
    print(f"running {prefix}")
    return prefix

def is_number(string):
    pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    return bool(pattern.match(string))

def get_fake_suffix(fake_dataname):
    return fake_dataname.split("fake")[1]

def get_fake_dataname(shuffle_ratio, repeat_time):
    return f"fake-shuffle_ratio-{str(shuffle_ratio)}-repeat_time-{str(repeat_time)}"

def prefix_str2para(prefix_str, **kwargs):
    prefix_str=prefix_str.split(".joblib")[0]
    para_list = prefix_str.split("-")
    para={}
    for i in range(0, len(para_list), 2):
        if para_list[i] in kwargs:
            para[para_list[i]] = kwargs[para_list[i]]
        else:
            if para_list[i] in ['repeat_times', 'shuffle_ratio']:
                continue
            if is_number(para_list[i+1]):
                if "." in para_list[i+1]:
                    para[para_list[i]] = float(para_list[i+1])
                else:
                    para[para_list[i]] = int(para_list[i+1])
            else:
                para[para_list[i]] = para_list[i+1]
    return para