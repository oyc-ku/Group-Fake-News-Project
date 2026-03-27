# Group-Fake-News-Project
To reproduce the results presented in the report, ensure that all required libraries are installed. You can create our Conda environment, which includes all necessary dependencies, by running the following command from the main project folder:

```bash
conda env create -f environment.yml
```

Specific parts of our code utilize the GPU and require CUDA support. Some libraries, such as cuML (a GPU replacement for scikit-learn), require a Linux based system. Additionally, certain PyTorch performance optimizations also require features that are only available on specific GPUs.<br>
We ran our code in a Linux environment with 16 GB of memory and a NVIDIA A30 GPU on CUDA 12.8, provided by the [Electronic Research Data Archive](https://www.erda.dk/) at The University of Copenhagen.


## Repository structure
Our data pipeline is split into different modules, which roughly correspond to each of the folders in this repository.<br>
These folders relate to a specific part of the project e.g. `logisticregression/` for [part 2](#2-logistic-regression-model).

We also have folders that store the shared resource folders. The `data/` folder contains datasets and analytics files and the `model/` folder contains our machine learning models. 
Note that `dataprocessing/` is a slight outlier, since it contains dataprocessing both for [part 1](#1-data-processing-and-exploration) and [part 4](#4-evaluation-of-models-with-liar-dataset).

Running the scripts will generate files. We have already saved some files such as `liar_stemmed_data.csv` in the repository, but some files are too big for github such as the FakeNewsCorpus dataset, `995,000_rows.csv`, the word-embedding model and the neural network model.

The `temp/` folder contains files we don't use anymore, such as `split_data.py` which was used to split the data into train, validation and test. Later on, we decided to split the data when we read the data files in the model training code instead of storing extra data files.

---

## 1 Data processing and exploration
To process the 995K FakeNewsCorpus subset, place the `995,000_rows.csv` file in the `data/` folder. You can then run the scripts located in the `dataprocessing/` folder.

The `dataprocessing.ipynb` is the main dataprocessing notebook, that uses the corpus to create a cleaned and stemmed version called `stemmed_data.csv` in the `data/` folder.
The dataprocessing folder also contains the equivalent pipeline for the LIAR dataset.

The `fakenews_functions.py` script contains functions and regex patterns, that `project.ipynb` uses to process the data.

The `exploration_token_distribution.ipynb` notebook is used to explore the distribution of unique tokens.

The `exploration_specialtokens_and_domains.ipynb` notebook is used to explore our special tokens and domain distribution. The notebook also generates `topword10000.csv`.

The `exploration_vocab_size.ipynb` notebook is used to explore the vocabulary sizes during parts of the data processing.


## 2 Logistic regression model
To run the logistic regression model in the `logisticregression/` folder, ensure `stemmed_data.csv` and `topwords10000.csv` are in the `data/` folder.

The `logisticregression/` folder has 2 equivalent notebooks. The first `logistic_regression.ipynb` runs on the GPU using the cuML library. The second `logistic_regression_CPU.ipynb` runs on CPU using standard libriaries. Each of these files create the `logistic_model.joblib` file that contains the logistic regression model and `logistic_metadata_model.joblib` that contains the model that has also been trained on metadata.


## 3 Neural network model
To run the neural network model in the `neuralnetwork/` folder, the files `stemmed_data.csv`, `995,000_rows.csv` must be in the `data/` folder.

The `word_embeddings.ipynb` notebook uses `stemmed_data.csv` to make a word-embedding model, which creates the files `embedding_model.model`, `embedding_model.model.syn1neg.npy` and `embedding_model.model.wv.vectors.npy` in the `models/` folder.

The `nn_lstm.ipynb` notebook contains the LSTM model, including the training loop and test results on the FakeNewsCorpus. This notebook saves the neural network model as `nn_model.pth` in the `models/` folder.

The `nn_lstm_preprocessing.py` script contains functions used in `nn_lstm.ipynb` to encode the news articles to PyTorch tensors.

The `nn_model_classes.py` script defines classes used in `nn_lstm.ipynb`, including our LSTM model class and classes used for training and loading the data in PyTorch.

## 4 Evaluation of models with LIAR dataset
**The test results from our models on the FakeNewsCorpus is found in [part 2](#2-logistic-regression-model) and [part 3](#3-neural-network-model).**<br>
The evaluation code for the LIAR dataset is located in the `evaluation/` folder. This requires the pre-trained models (`logistic_model.joblib` and `nn_model.pth`) to be placed in the `models/` folder.

To obtain our results:<br>
Run `evaluation_benchmark_random.ipynb` for benchmark results (includes FakeNewsCorpus).

Run `evaluation_logistic_model.ipynb` for Logistic Regression results.

Run `evaluation_nn_model.ipynb` for Neural Network results.
