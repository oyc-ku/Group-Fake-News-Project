# Group-Fake-News-Project
To reproduce the results in the report, the same libraries are needed.
Our conda environment, including the libraries we used, can be created with the following command from the main project folder.

```bash
conda env create -f environment.yml
```

Specific parts of our code also run on the GPU, and require CUDA support. Some libraries (such as cuML, that works as a almost identical replacement to sklearn, but runs on the GPU) require a Linux based PC.
Some PyTorch performance optimizations also require features that are only available on specific GPUs. 

We ran our code in a Linux environment with 16 GB of memory and a NVIDIA A30 GPU on CUDA 12.8, provided by the [Electronic Research Data Archive](https://www.erda.dk/) at The University of Copenhagen.

## Repository structure
There are different kinds of folder: One of thoese is the one that are related to a dirrect part e.g. `logisticregression/`for [part 2](#2-logistic-regression-model), and the second type is which that contain infomation/files which are using in multiple partssuch as `data/`.An outlier of those two are `dataprocessing/`  since it contains all dataprocessing both for [part 1](#1-data-processing-and-exploration) and [part 4](#4-evaluation-of-models-with-liar-dataset).

When running the code there would be generated files some which either already exsists `liar_stemmed_data.csv` but also data which are too big for github such as `stemmed_data.csv`.
If something does not match the path or it gives an error becuase of path just change the path so that it matches with it, sometimes it might be necessary to use your global path.
Download the whole project before you not just the files that you need.

## 1 Data processing and exploration
To process the 995K FakeNewsCorpus subset, place the `995,000_rows.csv` file in the `data/` folder. Then, the dataprocessing scripts/notebooks located in the `dataprocessing/` folder can be run.

The `dataprocessing.ipynb` is the main dataprocessing notebook, that uses the corpus to create a cleaned and stemmed version called `stemmed_data.csv` in the `data/` folder.

The `fakenews_functions.py` script contains functions and regex patterns that `project.ipynb` uses to process the data.

The `exploration.ipynb` notebook is used to explore the distribution of unique tokens.

The `exploration_tokens_and_domains.ipynb` notebook is used to explore our special tokens and the distribution of unique domains in the dataset it is also this file which makes `topword10000.csv`.

The `Compute_vocab_size.ipynb` notebook is used to explore the vocabulary sizes during parts of the data processing.

The `split_data.py` file is used to split the data into train, validation and test, even though all the files use `stemmed_data.csv` and then apply the samme procces on it as `split_data.py`.


## 2 Logistic regression model
To run the logistic regression model in the `logisticregression/` folder, the files `stemmed_data.csv` and `topwords10000.csv` need to be in the `data/` folder.

The `logisticregression/` folder has 2 equivalent files that contain the logistic regression model. One runs on the GPU using the cuML library`logistic_regression.ipynb`, and the other runs on the CPU`logistic_regression_CPU.ipynb` . Each of these files create the `linreg.joblib` file that contains the logistic regression model.

When running either of those files it would also do it both with and without metadeta.


## 3 Neural network model
To run the neural network model in the `neuralnetwork/` folder, the files `stemmed_data.csv`, `995,000_rows.csv` need to be in the `data/` folder.

The `word_embeddings.ipynb` notebook uses `stemmed_data.csv` to make a word-embedding model, which creates the files `embedding_model.model`, `embedding_model.model.syn1neg.npy` and `embedding_model.model.wv.vectors.npy` in the `models/` folder.

The `nn_lstm.ipynb` notebook contains the LSTM model, including the training loop and test results on the FakeNewsCorpus. This notebook saves the neural network model as `nn_model2.pth` in the `models/` folder.

The `nn_lstm_preprocessing.py` script contains functions used in `nn_lstm.ipynb` to encode the news articles to PyTorch tensors.


## 4 Evaluation of models with LIAR dataset
**The test results from our models on the FakeNewsCorpus is found in [part 2](#2-logistic-regression-model) and [part 3](#3-neural-network-model).**<br>
The evaluation code for the LIAR dataset is located in the `evaluation/` folder, and requires the files for the models `linreg.joblib` and `nn_model2.pth` to be in the `models/` folder. Then to get the results for the different of LIAR for benchmark run `evaluation_benchmark_random.ipynb` for logistic run `evaluation_logistic_model.ipynb` for neurelnetwork `evaluation_nn_model.ipynb`.




