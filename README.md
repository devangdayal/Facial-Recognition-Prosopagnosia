# Facial Recognition Prosopagnosia
 

This Deep Learning model helps a Prosopagnosia diagnosed Patient by helping them recognize the person.

Prosopagnosia is a neurological disorder characterized by the inability to recognize faces. Prosopagnosia is also known as face blindness or facial agnosia.

![image](https://user-images.githubusercontent.com/53809748/109661549-18696700-7b90-11eb-9bb3-74c3a57ed986.png)

In this software, the patient/user will be able to capture photos of the person he/she wants to remember the face of.
The user can do this by first entering the name of the person he/she is clicking the photo of and a dataset is formed of that person in real -time with a directory named after that person. 

If the next time the user meets that person and due to being a Prosopagnosia patient is unable to recognise that person or a group of people, then the user can just open the camera of the software and with the help of real-time face detection enabled in the software, the user is able to know a group of people or a person.


The real-time dataset formation will be enabled only when the person's name entered is already not in the dataset else it will just notify the user that the person's dataset is already in the database.
The real-time face detection works for both group of people and single person. If trying to recognise a person whose photos are not in the dataset then, that person is labelled as "Unknown" in the camera else his/her name will be shown.

## The initial results are as follows:
I started making a dataset with my own name with my pictures in it

![image](https://user-images.githubusercontent.com/66245321/119335421-fb819300-bca9-11eb-812a-b515565a77e7.png)

Then clicked my photos 5 times by pressing the "K" key

![image](https://user-images.githubusercontent.com/66245321/119339562-ebb87d80-bcae-11eb-8b49-1af62ad14983.png)

![image](https://user-images.githubusercontent.com/66245321/119339736-2de1bf00-bcaf-11eb-8464-905b59da83f8.png)

Now a new directory in the dataset folder is formed with the name "Rupali"
![image](https://user-images.githubusercontent.com/66245321/119339836-4eaa1480-bcaf-11eb-95d2-270d5f8c8bb0.png)

![image](https://user-images.githubusercontent.com/66245321/119339866-58337c80-bcaf-11eb-90ce-cc5c95cf54f0.png)

Now I run the face_detect python file to detect faces I already have now in the dataset

![image](https://user-images.githubusercontent.com/66245321/119340350-0a6b4400-bcb0-11eb-9749-176bd1f0a863.png)

Stored Narendra Modi's photos in the dataset using his images from the internet
![image](https://user-images.githubusercontent.com/66245321/119340063-9d57ae80-bcaf-11eb-98c4-75ae4f178f19.png)


Now detecting both faces together

![image](https://user-images.githubusercontent.com/66245321/119340090-a8124380-bcaf-11eb-92fe-45dee38c6163.png)


I didn't store Sachin Tendulkar's photos in the dataset , so it's labelled unknown for him
![image](https://user-images.githubusercontent.com/66245321/119340193-d2640100-bcaf-11eb-98c9-b3506a411737.png)

### This is how our software works










