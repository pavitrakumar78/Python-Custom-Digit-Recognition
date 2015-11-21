# Python-Custom-Digit-Recognition

You can apply a simple OCR on your own handrwitten digits using this python script.

I have used OpenCV to pre-process the image and to extract the digits from the picture.

Using K-Nearest Neighbours as our model - we train it using the freely [available](http://yann.lecun.com/exdb/mnist/) MNIST data set.

5000 MNIST digits have been printed on a single .png image.Each digit is 20x20.

##Analysis
I tried using just extracted the pixels as data to train and to predict the digits, but the accuracy was too low even on popular classification algorithms like SVM,KNN and Neural Netoworks.  I did improve the accuracy a little bit after trying some custom threshold values. The best accuracy I could achieve only using pixel values was close to 55-60% that was after converting all the images to Black OR White from Black AND White.    

After searching and reading about feature extraction from images for OCR - I stumbled [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) (Histogram of Gradients).  Basically, it tries to capture the shape of structures in the region by capturing information about gradients. Image gradient are simply intensity changes across pixels in an image.  

[pic-explain](https://gilscvblog.files.wordpress.com/2013/08/figure5.jpg)


It works so by dividing the image into small (usually 8x8 pixels) cells and blocks of 4x4 cells. Each cell has a fixed number of gradient orientation bins. Each pixel in the cell votes for a gradient orientation bin with a vote proportional to the gradient magnitude at that pixel or simple put, the "histogram" counts how many pixels have an edge with a specific orientation.  More more info please refer [this](https://gilscvblog.wordpress.com/2013/08/18/a-short-introduction-to-descriptors/) blog post.

Using just only HOG histogram vectors as features drastically improved the accuracy of the prediction.  Currently, I have used KNN from OpenCV as my model - I tried using SVM from the same module, but its accuracy was not as good as KNN. The best accuracy I have achieved on a sample image of about 100 digits is 80%.  In the future, I might add more features after looking into SIFT, SURF or even try to get a better accuracy using just the pixels as data! 
