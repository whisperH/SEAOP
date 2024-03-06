# from DetectOutlier.utlis.myutils import Utils
# import numpy as np
import json

#add the baselines from the pyod package
from .baseline.iforest import IForest
from .baseline.ocsvm import OCSVM
from .baseline.abod import ABOD
from .baseline.cblof import CBLOF
from .baseline.cof import COF
from .baseline.combination import aom
from .baseline.copod import COPOD
from .baseline.ecod import ECOD
from .baseline.feature_bagging import FeatureBagging
from .baseline.hbos import HBOS
from .baseline.knn import KNN
from .baseline.lmdd import LMDD
from .baseline.loda import LODA
from .baseline.lof import LOF
from .baseline.loci import LOCI
from .baseline.lscp import LSCP
from .baseline.mad import MAD
from .baseline.mcd import MCD
from .baseline.pca import PCA
from .baseline.rod import ROD
from .baseline.sod import SOD
from .baseline.sos import SOS
# from .baseline.vae import VAE
from .baseline.auto_encoder_torch import AutoEncoder
# from .baseline.so_gaal import SO_GAAL
# from .baseline.mo_gaal import MO_GAAL
from .baseline.xgbod import XGBOD
# from .baseline.deep_svdd import DeepSVDD


def ModelFactory(model_name, model_para):
    '''
    :param seed: seed for reproducible results
    :param model_name: model name
    :param tune: if necessary, tune the hyper-parameter based on the validation set constructed by the labeled anomalies
    '''

    model_dict = {
        'IForest':IForest,
        'OCSVM':OCSVM,
        'ABOD':ABOD,
        'CBLOF':CBLOF,
        'COF':COF,
        'AOM':aom,
        'COPOD':COPOD,
        'ECOD':ECOD,
        'FeatureBagging':FeatureBagging,
        'HBOS':HBOS,
        'KNN':KNN,
        'LMDD':LMDD,
        'LODA':LODA,
        'LOF':LOF,
        'LOCI':LOCI,
        'LSCP':LSCP,
        'MAD':MAD,
        'MCD':MCD,
        'PCA':PCA,
        'ROD':ROD,
        'SOD':SOD,
        'SOS':SOS,
        # 'VAE':VAE,
        # 'DeepSVDD': DeepSVDD,
        'AutoEncoder': AutoEncoder,
        # 'SOGAAL': SO_GAAL,
        # 'MOGAAL': MO_GAAL,
        'XGBOD': XGBOD
    }
    if model_name in model_dict:
        return model_dict[model_name](**model_para)
    else:
        raise "no model in model.PyOD file"
