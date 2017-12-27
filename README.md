# Python-Custom-Digit-Recognition

You can apply a simple OCR on your own handrwitten digits using this python script.
I have used OpenCV to pre-process the image and to extract the digits from the picture.
Using K-Nearest Neighbours (or SVM) as my model - I trained it using my own handwritten data set. I have also included the freely [available](http://yann.lecun.com/exdb/mnist/) MNIST data set so you can experiment on how different datasets work with different handwritings.

## Analysis  
I tried using just extracted the pixels as data to train and to predict the digits, but the accuracy was too low even on popular classification algorithms like SVM, KNN and Neural Netoworks.  I did improve the accuracy a little bit after trying some custom threshold values. The best accuracy I could achieve only using pixel values was close to 55-60% that was after converting all the images to Black OR White from Black AND White.    

After searching and reading about feature extraction from images for OCR - I stumbled [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (Histogram of Gradients).  Basically, it tries to capture the shape of structures in the region by capturing information about gradients. Image gradient are simply intensity changes across pixels in an image.  

![pic-explain](https://gilscvblog.files.wordpress.com/2013/08/figure5.jpg "pic")


It works by dividing the image into small (usually 8x8 pixels) cells and blocks of 4x4 cells. Each cell has a fixed number of gradient orientation bins. Each pixel in the cell votes for a gradient orientation bin with a vote proportional to the gradient magnitude at that pixel or simple put, the "histogram" counts how many pixels have an edge with a specific orientation.  More more info please refer [this](https://gilscvblog.wordpress.com/2013/08/18/a-short-introduction-to-descriptors/) blog post.

Using just only HOG histogram vectors as features drastically improved the accuracy of the prediction.  Currently, I have used KNN from OpenCV as my model - I tried using SVM from the same module, but its accuracy was not as good as KNN. The best accuracy I have achieved on a sample image of about 100 digits is 80%.  In the future, I might add more features after looking into SIFT, SURF or even try to get a better accuracy using just plain pixels as data! 

## Usage  

`digit_recog.py` *is deprecated - may not work with newer versions of libraries*  

~```python digit_recog.py digits.png user_image.png```~

~digits.png is the MNIST digits printed into one image - it is used for training.~  
~user_image.png is the user's custom image on which the handwritten digits will be classified.~

~Example:~  
~```python digit_recog.py digits.png test_image.png```~  

UPDATED CODE: `NEW_digit_recog.py`

```python NEW_digit_recog.py ```

This code uses my own handwritten digits (`custom_train_digits.jpg`) as training data. You can also use your own but keep the positioning of the digits similar to whats in `custom_train_digits.jpg` file. If you make modifications in the format of the custom training data (your handwritten digits) make sure to edit `load_digits_custom` function in `NEW_digit_recog.py` as per the changes.


Executing the program will generate 2 output files.  

This is the original image with digit boxes and the numbers on the top.   
![original_overlay](https://github.com/pavitrakumar78/Python-Custom-Digit-Recognition/blob/master/original_overlay.png)
This is a plain image with just the recognized numbers printed.   
![final_digits](https://github.com/pavitrakumar78/Python-Custom-Digit-Recognition/blob/master/final_digits.png)

### Note:  
- User image should be a scanned (atleast 300dpi) image.  
- Image can be any format supported by OpenCV.  
~- If custom dataset is to be used for training, then `split2d` method's parameters should be changed to avoid `ValueError: array split does not result in an equal division`. i.e if you are going to use `python digit_recog.py <your-own-training-data.jpg/png> <test-data.jpg/png>` then adjust the `DIGIT_DIM` parameter and resize the training image so that it looks more or less like the default MNSIT iamge (digits.png). `split2d` basically splits the input image into small cells of size `DIGIT_DIM x DIGIT_DIM` and uses those cells (images) for training.~  
- In `NEW_digit_recog.py`, use either  
  
```digits, labels = load_digits(TRAIN_DATA_IMG) #original MNIST data```
  
For MNIST dataset OR
  
```digits, labels = load_digits_custom('custom_train_digits.jpg') #my handwritten dataset```
  
For your own custom dataset  
  
  
## Dependencies  
Old:  
~OpenCV 2.4 or 3 (look comments - there are minor changes in syntax)~  
~NumPy~  
~skimage~  
~scipy~    

UPDATED - 2017 Dec for the file `NEW_digit_recog.py`  
Tested on:  
Windows 10    
Python 3.5    
  
Dependencies:  
numpy 1.31.1  
SciPy 0.19.0  
OpenCv (cv2) 3.2.0  
