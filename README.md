## News:
This is the official code for: [SEAOP: a statistical ensemble approach for outlier detection in quantitative proteomics data](https://academic.oup.com/bib/article/25/3/bbae129/7638267)
## Installation
Please install the anaconda firstly.
```shell
conda create -n ppOD python=3.7 
conda activate ppOD
pip install -r requirements.txt
```
## datasets
The datasets provided in this project is fake data, please contact the author if you need the true data.

The datasets provided in this project is fake data, please contact the author if you need the true data.

The datasets provided in this project is fake data, please contact the author if you need the true data.

### optional

if using **log2_qnorm** or **qnorm** in **scale_strategy**, please install R and
refer this [link](https://bioconductor.org/packages/release/bioc/html/limma.html) to install limma.

##### windows
- Install [R](https://cloud.r-project.org/bin/windows/base/) software.

- A work around is to download [rpy2](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2) from the Unofficial Windows Binaries for Python Extension Packages to the current working directory. Then use the following command to install rpy2 from the downloaded file:
    ```shell
    cd extension
    pip install rpy2-2.9.5-cp37-cp37m-win_amd64.whl
    ```
- please following [this blog](http://joonro.github.io/blog/posts/install-rpy2-windows-10/) to config the enviroment.

##### Linux
- Install R Software
check R version that the system can get, we need it >=4.2: 
    ```shell
    sudo apt-get remove r-base r-base-core r-base-dev r-recommended
    
    apt policy r-base
    ```
- install R
    ```shell
    # update indices
    sudo apt update -qq
    # install two helper packages we need
    sudo apt install --no-install-recommends software-properties-common dirmngr
    # add the signing key (by Michael Rutter) for these repos
    # To verify key, run gpg --show-keys /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    # Fingerprint: E298A3A825C0D65DFD57CBB651716619E084DAB9
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
    # add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
    sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
    sudo apt update
    sudo apt install --no-install-recommends r-base
    ```
- install limma by referring [this link](https://bioconductor.org/packages/limma/)

- install rpy2    

    ```shell
    pip install rpy2
    ```
# Run Outliers Detection

## train model
please make sure these yields are "True" value in XXX.yaml
```yaml
runSingleOD: true # step 1: train all single model. if false, load candidate pool from args.external_outlier_candidates
runFakeData: true # step 2: (see M4) generate simulated data via feature shuffle with different threshold
runSingleInfer: true # Step 3: test single model on fake data
runFakeStat: true # Step 4: run model on the fake data
runModelSelect: true # Step 4: run model on the fake data
```
```bash for select best parameters and models
python train_SEAOP.py -c ./configs/Hela/boost.yaml
```

## infer model
1. create new folder with path: 
```bash
{logs_dir}\outliers
```
where "logs_dir" is defined in XXXX.yaml with "logs_dir" 

2. putting [model_selection.npy](https://drive.google.com/file/d/1fP4tmS9iSBYGMhyP14ugkRyBZDmxVIlC/view?usp=drive_link) in .\Result\Hela_boost_repeat100\outliers

3. please make sure these yields as following in XXX.yaml
4. infer new dataset
```bash for infer new dataset
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HCC_P.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HCC_T.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile LADC_N.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile LADC_T.csv runInferSingle True runBoostTest True
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HelaGroups.csv runInferSingle True runBoostTest True
```
Note:
If your device is laptop or computer instead of Server, please set multiple_thread as False in the command line.:
```bash for infer new dataset
python infer_SEAOP.py -c ../../configs/Hela/boost_100.yaml infer_datafile HelaGroups.csv runInferSingle True runBoostTest True multiple_thread False
```
