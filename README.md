# AMLSassignment
ELEC0132: Applied Machine Learning Systems (18/19) Assignment

## Download Full repo with datasets available:
https://drive.google.com/open?id=1XODub1W-K3Z8mu4WHU5fix6asR0PK4xq

## Setup
- I highly suggest using a python3.5 conda environment (with tensorflow-gpu==1.4 if using gpu)
- `pip install -r requirements.txt`

## Original Dataset
All three necessary versions of the dataset - to be able to run the code) can be found on the drive link above, you can also download the original provided labeled dataset (with outliers) using this link: https://drive.google.com/drive/folders/1NgP2jQakFHibIhpevDLshodWw-L52yXi?usp=sharing

## Written Report
You can currently find the latest version of the report in the provided pdf: `deep_learning_dive.pdf`

## File Structure / Running
 - __Data Pre-Processing__
  - run `data_preprocessing.ipynb`, all results are displayed inside the python notebook
 - __Main folders to lookout for__
  - `binary_age` for all learning and test runs with SVM, Logistic Regression, CNN, VGG on the 'young vs old' binary problem.
  - `binary_emotion` for all learning and test runs with SVM, Logistic Regression, CNN, VGG on the 'smiling vs not smiling' binary problem.
  - `binary_glasses` for all learning and test runs with SVM, Logistic Regression, CNN, VGG on the 'glasses vs no glasses' binary problem.
  - `binary_human` for all learning and test runs with SVM, Logistic Regression, CNN, VGG on the 'human vs cartoon' binary problem.
  - `multiclass_hair` for all learning and test runs with SVM, Logistic Regression, CNN on the 'bold vs blond vs brown vs ginger vs dark vs grey' multiclass problem.
  - `preprocessing` for all pre-processing related files, PCA dataset conversion files (`PCA_binary.py` and `PCA_multiclass.py`), outlier detection in `data_preprocessing.ipynb`, different dlib '.dat' feature extractors and pre-trained models (ex: CNN pre-trained facial detector/feature extractor).
  - `pca_dataset` includes all the numpy saved datasets from the pixel to PCA feature conversion.
  - `inference` contains all our inference results on the test splits that have been made on the original dataset (in each case 20% of the shuffled dataset, the splits have been made with the same seed for each runs so should be similar, same goes for the shuffling for every different file) (also, this is not the none labeled inference set of 100 images which was given to us on the 5th of January 2019), it also contains the VGG16 runs of all the binary tasks for the none labeled, 100 example, inference set, given to us on the 5th of January 2019. There seems to be quite the over-fitting in some cases but the latter are still promising results.  
  - Each binary and multiclass problem folders contain a `result_logs` folder that contains all the logging outputs for all the different runs for the given problem (using different types of features for the data - augmented landmarks, 250 first PCA components, normal 128x128 RGB pixel data). The latter are clearly labeled with what type of run it was, for example, in `binary_age/result_logs/output_binary_age_LR_PCA_plotting.txt`, we will have the learning curves be printed based on the cross validated n-fold grid search and results in `binary_age/result_logs/output_binary_age_LR_PCA.txt`, cross-validation matrix and other useful metrics are printed such as f1-scores and inference accuracies, as well as model architectures in the case of CNN or VGG codes. In the `result_logs` folder we also have intuitively named `.png` files that show learning (training and validation accuracy) curves and plots for the different models being ran. The inference `.csv` files are also included here but more accessible previously described `inference` folder at the root of the project. a compressed zip file (for easy download) of the latter is also provided at the root.  
 - __TensorBoard__
  - In each classification problem folder a `tf_logs` folder can also be seen and includes all the CNN runs with different structures and parameters, the latter can be run using tensorboard in the following way: `tensorboar --logdir=tf_logs/1/train` for example. The numbers coincide with the txt files of the CNNs (sometimes shifted by one in the case of the multi-class problem).

# Common Pitfalls and gitignored folders

I have explicitly not included the `dataset`, `new_dataset` and `test_dataset` (5th of January 2019 none-labeled dataset) folders in my commits.
I have added the latter in the `.gitignore` file since they are heavy and don't want to put load on Github's poor data-centers.

# USEFUL LINKS (more to be added soon)
https://arxiv.org/pdf/1509.06451.pdf - paper on DCN for face-detection that handles occlusion really well.
https://github.com/opencv/opencv/tree/master/data/haarcascades - for the haarcascade pre-trained models
http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf - dropout
https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
https://stackoverflow.com/questions/33610825/normalization-in-image-processing
https://keras.io/applications/
https://www.deeplearningbook.org/contents/guidelines.html
https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751
https://stats.stackexchange.com/questions/233850/using-pca-on-an-image-dataset-prior-to-classification-with-a-neural-network
https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_to_Speed-up_Machine_Learning_Algorithms.ipynb
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once
https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
https://www.dataquest.io/blog/learning-curves-machine-learning/
https://medium.com/difference-engine-ai/keras-a-thing-you-should-know-about-keras-if-you-plan-to-train-a-deep-learning-model-on-a-large-fdd63ce66bd2

and many many more ...
