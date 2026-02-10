# FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

This repository is academic reproduction of the official GitHub repository associated with the paper "FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective".
You can find the paper [here](https://arxiv.org/pdf/2311.06190.pdf) and the official GitHub repository [here](https://github.com/aikunyi/FourierGNN).


### Environment Setup
You can setup either virtual environment or conda environment, and then install the required packages using the following command
`
pip install -r requirements.txt
`

### Data Preparation
The datasets used in the paper can be downloaded from the sources mentioned in the paper. After downloading place the datasets in the `./data/_RAW_DATASETS` directory. After this you can run the command `python data_preprocess.py` to perform complete data preprocessing and generate the processed datasets in the `./data` directory.

### Preliminary Experiments
To run the preliminary experiments, you can use the command `python run_script.py`