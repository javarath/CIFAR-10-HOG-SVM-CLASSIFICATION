# CIFAR-10 Image Classification with HOG and SVM

## Dataset Description
![image](https://github.com/javarath/CIFAR-10-HOG-SVM-CLASSIFICATION/assets/102171533/02da24de-12bd-496e-8d65-b9b0011876ee)

The CIFAR-10 dataset consists of 60,000 color images of size 32x32, divided into 10 classes, with 6,000 images per class. The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is widely used as a benchmark for image classification and object recognition tasks. The dataset can be loaded using the `tensorflow.keras.datasets.cifar10.load_data()` function.

## Goal of Project

The goal of this project is to use **histogram of oriented gradients (HOG)** and **support vector machine (SVM)** to classify images from the CIFAR-10 dataset. HOG is a feature extraction technique that captures the shape and texture information of the objects in the image. SVM is a supervised learning algorithm that can handle linear and nonlinear relationships using different kernel functions.

## Requirements to Run

To run this project, you will need the following:

- Python 3.7 or higher
- TensorFlow 2.6 or higher
- scikit-image 0.18 or higher
- scikit-learn 0.24 or higher
- matplotlib 3.4 or higher
- joblib 1.0 or higher

You can install these dependencies using `pip install -r requirements.txt` or `conda install --file requirements.txt` depending on your package manager.

## Preprocessing Techniques Used

The preprocessing techniques used in this project are:

- Converting the images to grayscale using `skimage.color.rgb2gray()`. This reduces the dimensionality of the images and makes them compatible with the HOG feature extraction.
- Extracting the HOG features from the images using `skimage.feature.hog()`. This computes the histogram of the gradient orientations in a local region of an image. The parameters used are `orientations=9`, `pixels_per_cell=(8, 8)`, and `cells_per_block=(2, 2)`. These parameters control the number of bins, the size of the cells, and the number of cells per block respectively.
- **POST-HOG TRANSFORMED IMAGES**
- ![image](https://github.com/javarath/CIFAR-10-HOG-SVM-CLASSIFICATION/assets/102171533/906d3fe3-91d1-4384-9e57-feadf844eca0)

- Scaling the features using `sklearn.preprocessing.StandardScaler()`. This standardizes the features by removing the mean and scaling to unit variance. This helps to improve the performance and convergence of the SVM model.
- Reducing the dimensionality of the features using `sklearn.decomposition.PCA()`. This identifies a set of orthogonal axes, called principal components, that capture the maximum variance in the data. The number of components is chosen such that the cumulative explained variance ratio is at least 0.9. This helps to reduce the computational cost and noise in the data.

## Pipelining

Pipelining is a technique that allows to chain multiple steps of a machine learning workflow into a single object. This simplifies the code and avoids intermediate variables. In this project, a pipeline is created using `sklearn.pipeline.Pipeline()` that consists of three steps: `StandardScaler`, `PCA`, and `SVC`. The pipeline can be fitted to the training data and used to make predictions or find the score on the test data.

## Algorithm Used

The algorithm used in this project is SVM with radial basis function (RBF) kernel. SVM is a supervised learning algorithm that can be used for classification and regression tasks. It works by finding the optimal hyperplane that maximally separates the data into different classes. The RBF kernel is a nonlinear kernel that computes the similarity between two points based on their distance. The parameters used are `C=10` and `cache_size=10000`. The parameter `C` controls the trade-off between the margin and the misclassification penalty. The parameter `cache_size` specifies the size of the kernel cache in MB.

## Results

The accuracy of the SVM model on the test data is 62.97%. This means that the model correctly classified 62.97% of the test images. The confusion matrix of the model is shown below:

| **Actual/Predicted** | **Airplane** | **Automobile** | **Bird** | **Cat** | **Deer** | **Dog** | **Frog** | **Horse** | **Ship** | **Truck** |
|------------------|----------|-----------|------|-----|------|-----|------|-------|------|-------|
| **Airplane**     | 714      | 16        | 55   | 19  | 17   | 13  | 11   | 20    | 97   | 38    |
| **Automobile**   | 14       | 797       | 7    | 10  | 5    | 9   | 9    | 9     | 28   | 112   |
| **Bird**         | 69       | 6         | 551  | 69  | 98   | 63  | 74   | 36    | 23   | 11    |
| **Cat**          | 24       | 16        | 67   | 487 | 63   | 182 | 92   | 35    | 19   | 15    |
| **Deer**         | 28       | 5         | 84   | 63  | 621  | 46  | 97   | 40    | 12   | 4     |
| **Dog**          | 14       | 9         | 54   | 184 | 61   | 555 | 53   | 49    | 10   | 11    |
| **Frog**         | 9        | 8         | 44   | 61  | 64   | 29  | 759  | 10    | 7    | 9     |
| **Horse**        | 25       | 13        | 38   | 40  | 66   | 65  | 18   | 704   | 14   | 17    |
| **Ship**         | 58       | 42        | 19   | 16  | 9    | 10  | 9    | 6     | 787  | 44    |
| **Truck**        | 41       | 103       | 15   | 18  | 9    | 13  | 13   | 20    | 47   | 721   |

The confusion matrix shows the number of images that were predicted as each class for each actual class. For example, the top left cell shows that 714 images that were actually airplanes were predicted as airplanes, while the bottom right cell shows that 721 images that were actually trucks were predicted as trucks. The diagonal cells show the correct predictions, while the off-diagonal cells show the incorrect predictions. The confusion matrix can be used to identify the strengths and weaknesses of the model, and to find the sources of errors.

## Parallel Computing

Parallel computing is a technique that allows to execute multiple tasks simultaneously on multiple processors or cores. This can speed up the computation and improve the efficiency of the program. In this project, parallel computing is used in two places:

- In the `HogTransform` function, the `joblib.Parallel` and `joblib.delayed` functions are used to apply the HOG feature extraction to each image in parallel. The parameter `n_jobs=-1` specifies that all the available cores should be used.
- In the `SVC` classifier, the parameter `cache_size=10000` specifies that 10 GB of memory should be allocated for the kernel cache. This can improve the performance of the SVM model by reducing the number of kernel evaluations.

## How to Run the pipe.joblib on One's Own System

To run the `pipe.joblib` file on your system, follow these steps:

1. Download the `pipe.joblib` file from the provided Google Drive link (https://drive.google.com/file/d/1SegqnYOH9XEF-BTPJ4cCHT5nq8GudPy6/view?usp=drive_link) and save it in a local directory.

2. Install the required dependencies listed in the requirements section using `pip install -r requirements.txt` or `conda install --file requirements.txt` depending on your package manager.

3. Load the `pipe.joblib` file using `pipe = joblib.load('pipe.joblib')`.

4. Load some new data using `tensorflow.keras.datasets.cifar10.load_data()`. You can use the test data or any other data that matches the format and size of the CIFAR-10 dataset.

5. Preprocess the data using the `HogTransform` function defined in the notebook. You can copy and paste the function definition from the notebook or import it from another module.

6. Make predictions using the pipe using `y_pred = pipe.predict(x_new_hog)`.

7. Find the score using the pipe using `score = pipe.score(x_new_hog, y_new)`.

By following these steps, you can load the pre-trained `pipe.joblib` model and use it for making predictions or evaluating its performance on new data, after downloading the file from the provided Google Drive link.

## Acknowledgement

I have completed this project by myself, without any external help or plagiarism. I have followed the ethical and academic standards of the institution and the course.

## References

- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Histogram of oriented gradients](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)
- [Support vector machine](https://scikit-learn.org/stable/modules/svm.html)
- [Principal component analysis](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Pipelining](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [Parallel computing](https://joblib.readthedocs.io/en/latest/)
