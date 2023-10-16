## News:
We will release the code upon the publication.

## Installation
Please install the anaconda firstly.
```shell
conda create -n ppOD python=3.7 
conda activate ppOD
pip install -r requirements.txt
```


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
runSingleOD: True 
runBoostTest: True
```
```bash
python generate_unsupervised_results.py -c ./configs/Hela/Hela_int_all.yaml
```
or
```bash
python generate_unsupervised_results.py -c ./configs/Hela/Hela_int_all.yaml runSingleOD True runBoostTest True
```


## test model
please make sure these yields as following in XXX.yaml
```yaml
runSingleOD: False 
runBoostTest: True
```
```bash
python generate_unsupervised_results.py -c ./configs/Hela/Hela_int_all.yaml
```
or
```bash
python generate_unsupervised_results.py -c ./configs/Hela/Hela_int_all.yaml  runSingleOD False runBoostTest True
```