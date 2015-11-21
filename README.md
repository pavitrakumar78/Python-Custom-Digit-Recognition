# Python-Custom-Digit-Recognition

You can apply a simple OCR on your own handrwitten digits using this python script.
I have used OpenCV to pre-process the image and to extract the digits from the picture.
Using K-Nearest Neighbours as my model - I trained it using the freely [available](http://yann.lecun.com/exdb/mnist/) MNIST data set. 5000 MNIST digits have been printed on a single .png image where each digit is 20x20 in size.

##Analysis
I tried using just extracted the pixels as data to train and to predict the digits, but the accuracy was too low even on popular classification algorithms like SVM,KNN and Neural Netoworks.  I did improve the accuracy a little bit after trying some custom threshold values. The best accuracy I could achieve only using pixel values was close to 55-60% that was after converting all the images to Black OR White from Black AND White.    

After searching and reading about feature extraction from images for OCR - I stumbled [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (Histogram of Gradients).  Basically, it tries to capture the shape of structures in the region by capturing information about gradients. Image gradient are simply intensity changes across pixels in an image.  

![pic-explain](https://gilscvblog.files.wordpress.com/2013/08/figure5.jpg "pic")


It works by dividing the image into small (usually 8x8 pixels) cells and blocks of 4x4 cells. Each cell has a fixed number of gradient orientation bins. Each pixel in the cell votes for a gradient orientation bin with a vote proportional to the gradient magnitude at that pixel or simple put, the "histogram" counts how many pixels have an edge with a specific orientation.  More more info please refer [this](https://gilscvblog.wordpress.com/2013/08/18/a-short-introduction-to-descriptors/) blog post.

Using just only HOG histogram vectors as features drastically improved the accuracy of the prediction.  Currently, I have used KNN from OpenCV as my model - I tried using SVM from the same module, but its accuracy was not as good as KNN. The best accuracy I have achieved on a sample image of about 100 digits is 80%.  In the future, I might add more features after looking into SIFT, SURF or even try to get a better accuracy using just the pixels as data! 

##Usage

```python digit_recog.py digits.png user_image.png```

digits.png is the MNIST digits printed into one image - it is used for training.  
user_image.png is the user's custom image.  

Example:  
```python digit_recog.py digits.png test_image.png```  

Executing the program will generate 2 output files.  

original_overlay.png -> This is the original image with digit boxes and the numbers on the top.    
final_digits.png -> This is a plain image with just the recognized numbers printed.  

###Note:  
User image should be a scanned (atleast 300dpi) image.  
Image can be any format supported by OpenCV.

##Dependencies

OpenCV 2.4 or 3 (look comments - there are minor changes in syntax)
NumPy
skimage
scipy
