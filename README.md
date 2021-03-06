# Faceblock

## What does Faceblock do?
* Faceblock takes images and videos and automatically finds and fills any faces with a blue rectangle
* I created faceblock so that it would be easy for anyone to block out the faces of people in pictures and videos

![alt text](https://github.com/magnesium-nitrate/faceblock/blob/master/cov6.jpg)
![alt_text](https://github.com/magnesium-nitrate/faceblock/blob/master/after.jpg)
![alt_text](https://github.com/magnesium-nitrate/faceblock/blob/master/example.gif)

## How to use the .py picture code
* To use the regular python files make sure you pip install matplotlib, mtcnn, and opencv
* In the filename variable change "cov6.jpg" to the name of your file and it'll be able to read it in
* The code is currently set to download images under the name "new8.jpg" but this can easily be changed by changing the name as well

## How to use the .ipynb picture code
* If you don't want to install python on your computer you can always download the .ipynb code and upload it to google colaboratory
* After uploading the code to google colab, when you run the code the first block will let you upload a file and you should upload your image there
* When you upload your image, make sure to change the filename variable to the name of the image you just uploaded
* After the code is run, the new image will automatically be downloaded onto your computer

## How to use the .py video code
* Unfortunately google colab is not cooperating as I am writing this README so if you want to run faceblock detection on videos you will need to have python installed
* You will need the following libraries installed: matplotlib, mtcnn, and opencv
* To faceblock your video, in the input_path parameter put the name of your video
* Your video will be outputted to whatever you set the output_path parameter to
* It should automatically download as an mp4
