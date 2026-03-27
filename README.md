# Group-Fake-News-Project
First update your Python environment, so it matches with environment.yml, this is to make sure you have all the corrrect libraries and such.
## Part 1
### Task 1/cleaning

To reproduce the results in the report, first download the script called fakenews_function.py and the sample news file.
After those two files have been installed put them in a folder together with Compute_vocab_size.ipynb. Change path_of_sample to the path of the sample file(might also need to change the n_cpu in ParralellPandas). Run the cells in the file until you get to the cell where it would print the size of the vocab, and then run that cell.
### Task 2/cleaning the full data
When you're done with task 1, then download project.ipynb. Update both the sample path and the CSV file so they both point to the their respektive dataset(sample to sample and file_chunks to 995k rows). Now make sure that all file paths in the notebook match your setup, and then run the script to generate the file. Now open up Compute_vocab_size.ipynb and run the sections you didn't do in task 1. And again update the file so it points to the 995k rows dataset before running those parts.

### Task 3/Exploration of data
Now open up exploration_tokens_and_domains.ipynb here you will need both the orignal 995k dataset and the cleaned version. Make sure to update the file paths so that data points to the cleaned dataset and metadata points to the uncleaned dataset. Then run the notebook. Next open up exploration.ipynb and update the file paths to match your setup and run this notebook aswell.

### Task 4/Splitting the data
Get the cleaned data file and the spilt_data.py. Update the file path in split_data.py so it points to the cleaned data file, and run the script.


## Part 2
### Task 2/Logistic model
You need the cleaned data, the top 10k words file (which you should have from exploration.ipynb), the full data set, and the file logistic_regression_CPU.ipynb. Update the paths so that full_data points to the full dataset, vocab_list points to the top‑10k‑words file, and data points to the cleaned dataset. And then run the notebook down to the cell that outputs the confusion matrix.
### Task 3/Adding metadata to logistic model
Use the same files as in task 2, but now start from the confusion matrix and run the notebook down to the evaluation part.
## Part 3
To get the advanced model make sure to run word_embeddings.ipynb first, after you have run that go to the nn_lstm_preprocessing.py file and change the paths to match your paths. So EMBEDDING_MODEL_PATH should be the model that you got from word_embeddings.ipynb and METADATA_PATH should be the total 995k rows dataset, and STEMMED_DATA_PATH should be the stemmed one that you got from the part 1 task 2. After that is done now run nn_lstm.ipynb until testing and you will get a model saved.

## Part 4

### Task 1/Simple and advanced models
For the logistic_regression, repeat what you did for Part 2 Task 2, but this time run the file until the LIAR part.
Alternatively you can take the linreg.joblib file that you made from logistic_regression_CPU.ipynb, and in the same environment open the file log_rev_ev.ipynb and run the notebook to get the same result.

For the LSTM part get all the files from Part 3 and run the testing part.

### Task 2/Liar dataset
First you need to procces the LIAR dataset. So take the test split of the LIAR dataset and run proccees_lair_data_set.ipynp, whilst making sure the notebook has the correct path to the test file and that it can see fakenews_functions.py.

For the logistic_regression, follow the same procedure as in Task 1, but just run it for the LIAR dataset. You can do this by either the log_rev_ev.ipynb or the logistic_regression_CPU.ipynb by scrolling down to the LIAR part and running the correspondings cells. If doing this in log_rev_ev.ipynb then make sure to have the joblib file and the data_test both that you get from running logistic_regression_CPU.ipynb.

For the LSTM part get all the files from Part 3 and the fille called nn_modelevaluation.ipynb.

## Part 5
To get the data from part 5 run data_set_exp.ipynb and remember to set the path so they correctly point to the cleaned liar and the cleaned test set.

