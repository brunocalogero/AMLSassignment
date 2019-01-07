# AMLSassignment
ELEC0132: Applied Machine Learning Systems (18/19) Assignment

## Setup
- I highly suggest using a python3.5 conda environment (with tensorflow-gpu==1.4 if using gpu)
- `pip install -r requirements.txt`

## File Structure / Running
 - __Data Pre-Processing__
  - run `data_preprocessing.ipynb`, all results are displayed inside the python notebook
 - __Binary/Multi-class Classifiers__
  - Each file can be run by simply typing `python desired_file_to_run.py`, all the results have been stored in `result_logs`, this includes txt files containing outputs of various runs of the different algorithms. A `tf_logs` folder can also be seen and includes all the CNN runs with different structures and parameters, the latter can be run using tensorboard in the following way: `tensorboar --logdir=tf_logs/1/train` for example. The numbers coincide with the txt files of the CNNs (sometimes shifted by one in the case of the multi-class problem).

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
