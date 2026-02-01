##### **Slide 2 - Problem domain and dataset**



* Finding Waldo in an image can be a very grueling task, only accomplished by people with gifted eyesight
* We will use a sparse CNN optimized to run quickly on mobile low-power, low-computation hardware
* In this paper, we present a low-cost strawberry detection system based on CNNs
* The paper’s original dataset is composed of high resolution 1080 x 1920 pixel BRG images of ripe strawberries collected from a strawberry farm. The strawberries are of a wide range of sizes, and some have unusual shapes. Also, some are occluded (blocked) by other fruits, foliage (leaves) or dirt. The size of the dataset was 4550, including a total of 22,662 strawberries in the images.
* For the dataset we have:
* There are 19 original images. A python script was run to generate smaller labelled versions of these images. There are color, grayscale and black and white versions of these images available. All images are in jpg format

&nbsp;	-  317 256 x 256 pixel images

&nbsp;	- 1344 128 x 128 pixel images

&nbsp;	- 5376 64 x 64 pixel images

* so the total dataset size is 7037 images



* So basically we have 19 original images that got run through a python script to generate smaller versions of the images where some are labelled as Waldo and others are not labelled as not Waldo. For every smaller version of image generated, there is a colour, gray scale and black and white version. In addition, there is a version of that image with 256 pixels, 128 pixels and 64 pixels.



* Additional notes: 
* Waldo look alikes (i.e: different hair/glasses) were also labelled as Waldo
* Here are some dataset drawbacks that make image classification of Waldo a hard problem: 

&nbsp;	- Waldo’s variable size (scaling of Waldo)

&nbsp;	- Repeated patterns of red and white on other objects that are not Waldo

&nbsp;	- Occlusion (Waldo is blocked by other objects)

&nbsp;	- Unbalanced set of images where most of the images do not contain Waldo





##### **Slide 6 - Role Division**



There are 3 main components to our project: 



1\) Implementation of the CNN model on Pytorch - Gabe will be responsible for leading this initiative



2\) Training of CNN model on Pytorch - Rudy will be responsible for leading this initiative



3\) Ablation studies - Satyam will be responsible for leading this initiative



* It should be noted that throughout the process, if we realize one person has too much work, and other people have not enough work, we will help out the person with too much work



##### **Slide 7 - EDI Considerations**



* Our problem relates to image classification of a specific person, in this case it’s Waldo. Since there is a diverse amount of people, it’s important to train data on different varieties of people, most notably people of different races and ethnicities. This way, our model predictions are based on diverse training data, and won’t have an overfitting problem towards a single dominant race
* In our Waldo example, there is a diversity of Waldo images including different sizes of Waldos, as well as Waldos with different hair and glasses



