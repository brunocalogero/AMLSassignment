# AMLSassignment
ELEC0132: Applied Machine Learning Systems (18/19) Assignment


https://arxiv.org/pdf/1509.06451.pdf - paper on DCN for face-detection that handles occlusion really well.

--> in our particular case the faces aren't very intensely occluded and we are fortunate to mainly have to deal with frontal faces, there aren't many cases of tilted faces or occluded faces/ hidden faces due to noise such as watermarks or arms or anything of the sort, the model that we are currently using has been trained such that its weights or in other words the higher order representations that the algorithm has learnt are just enough for what we need.
Indeed this is shown by the fact that using the CNN hasn't very much improved our results, the latter  'is capable of detecting faces with large pose variation, it accepts full image of arbitrary size
and the faces of different scales can appear anywhere in the image' unlike the HOG classifier.  

higher dimensional space to learn more representations

https://github.com/opencv/opencv/tree/master/data/haarcascades - for the haarcascade pre-trained models
http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf - dropout
