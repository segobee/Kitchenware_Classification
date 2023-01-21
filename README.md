# Kitchenware Classification

In this readme file, I give an overview on the project workflow. The file is robust enough for you to understand rudiments followed in project execution. 


## Project Background/Introduction

John is a young man, who while growing up never lifted a finger to do anything "house chores". Everything was being taking care of by his family domestic workers. These actions by his family made him **NOT YOUR GUY** when it comes to house chores. It's that bad that John is poor in identifying kitchen items. Is this how John wants to live for the rest of his life? Will there be any available tech that would fill the gap when John decided to start living life ALONE? Those questions beg for answers. 

While growing up, people had this **NOT SO GOOD** perception about John. They concluded that nothing would be added to his name as his personal achievement aside his family inheritance. Even his parents had written him off. Living his care free life, these misgiving perceptions transmitted to his hears. At first, he thought this amounted to nothing. But, as time goes on, actions by the people surrounding him recalibrated his life. He vowed to disassociate himself from things handed to him by his parents. Thus, changing the cause of his life. 

Many years later, after much hassle, John is an achiever. One thing you cannot take away from us as human beings is our resolve to show affluence of being successful. And John is no exception. He decided to buy a spacious house. Show the people who had written him off, of his achievements. His first conquered minds would be his parents. 

John bought his house in a community of affluents. You know buying a **House** can be classified as a fulfilment for the hardest worker, depending on the person's hierarchy of needs. You play around the house from a custom size bedroom to a nice furnished living room, to a nice well-tiled bathroom, and all add-ons that come with the house. You take it upon yourself to enjoy the interior design of the house. You are pleased with the serene enviroment, enjoying the exterior beautification surrounding the house. These, cummulatively, were the first experience of John. 

After relieving himself from the amusement of the house, what next?

It clicked that he needed to fill his round emptied ball (stomach) with a good meal. Ideally for him, he would have ordered for food. But he couldn't, because of his environment. He entered the kitchen to fulfill the promise to your stomach. Then the realization of his up-bringing sets in. He couldn't understand what each material in the kitchen is meant for. His hungeriness has entered another phase. Who'll come to John's rescue? What's the next action he has to take? Follow me, as fill you in of his benefactor (a problem solver). 

In the absence of no problem, it guarantees no solution (Which is a rare). In this case, there is a problem to be solved. John on his part, had to surf the internet to find solution to his predicament. Then he stumbled upon **DataTalks.** `DataTalks` is Data community for Data Scientist and Machine Learning Expert. They are problem solvers. They see pregnant data, and birth solution from the data foetus.  

Prior to John's situation, **DataTalk** had formulated a competition among it's expert to design a product that's mandate will be to identify **Kitchenwares.** The aftermight of this competition would mean a solution to John's predicament. All John need do, is to take a picture of any kitchenware and upload it, then the algorithm designed by the expert identify the item and come up with a name. 

I happen to be one of the contributors to this project by **DataTalks.** The following captures the steps I took in building the algorithm that will serve as the backbone of the product intended by **DataTalks** 

Readmore about the competition [here](https://www.kaggle.com/competitions/kitchenware-classification/)


## Project Statement 

This project seeks to build a model that will classify images of different kitchenware items into 6 classes:
- `cups`
- `glasses`
- `plates`
- `spoons`
- `forks`
- `knives`

In this project, I intend building an algorithm with high precision accuracy in identifying any of the above classes of kitchenware. 

## Project Data Source 

The data (sample images) were sourced using **Toloka.** Readmore about toloka [here](https://toloka.ai/)

Project data can be assessed from [here](https://www.kaggle.com/competitions/kitchenware-classification/data)


## Project Dataset Overview 

This dataset contains images of different kitchenware

**Competition Dataset Files**

- `train.csv` - the training set (Image IDs and classes)
- `test.csv` - the performance of the model will be evaluated on this test file. It contains only image IDs. The model predicts the classes of the images using the provided image IDs and image jpegs. The resulting output will be contained in a csv as a submission file, which will be submitted for evaluation and model rating among the rest of the competitors. 
- `images` - the sample images the model will be trained and tested on in the JPEG format. 

**Image Categories with sample size**

`training set`
- `cups` - 
- `glasses` - 
- `plates` - 
- `spoons` - 
- `forks` - 
- `knives` - 


## Model Training materials 

Since the project problem focus more on deep learning, I'll be the using following learning resources:
- TensorFlow
- Keras - this is built ontop of TensorFlow. It helps us train and use neural networks. This is importance to this project. 


## Project Workflow 

Since this project use case are image classes, for effective prediction, base on history project recommendations, **Convolutional Neural Networks** will be trained. If after investigating the performance and knowledge of a pre-trained model on this project use case, and we discover that the model can as well do well for project, thus, focusing more **Transfer Learning.**

`Brief Background on CNN`

In Machine Learning tracks, there are several aspects segmented as a **GO-TO**, based the project peculiar used case. CNN is one aspect of them. Convolutional Neural Networks are the type of neural networks that are used for image classification mostly. It consist of two different types of layers boxed as knowledge tracks during it's information gathering processes. One is Convolutional Layers, while the other is Dense layers. 

On this project, I'll be focusing more on the Dense layers, since I intend building on the extracted Convolutional Layers from a pre-built application (model) in keras. The intended extraction of Convolutional Layers from a pre-built model will be transformed to a **Vector Representation.** This `Vector Representation` will serve as my input to the Dense Layers I intend training.

The following summarizes the steps I Intend taking:

**Step 1 ==> Investigating a sample image on a pre-built model**

- Load a sample image for initial investigation 
- Check if the pre-built application in keras serve the purpose of this project, by doing the following:
-- Supply loaded sample image to the application 
-- Make a prediction on the image with the application
-- Check the output of it's prediction, most especially the class prediction 
-- Conclude if it meets the project peculiar case 
-- And if it doesn't, proceed to the next step 

**Step 2 ==> Training a base model**

Leaping from the instincts gained from step 1 processes, I intend creating a new model that will serve as base model for further training as the project progresses. 

- Training a different model (base model) on the six classes the pre-built model was not trained on. The intentions for this process are as follows:

-- Loading the project datasets
-- Split the dataset to training and validation sets
-- Reuse pre-trained model that could serve the project used case 
-- Extract convolutional layers from the chosen pre-built model
-- Convert the extracted CL's to a vector representation  
-- Supply the vector representation as input to the Dense Layers I intend training 
-- Train the Dense Layers using the vector representation
-- Train the model using the accompany parameters of learning 

After all these are achived, I'll proceed to the next step. 

**Step 3 ==> Evaluating the base model**

- Evaluating the performance of the base model on the validation dataset
- Visual depiction of it's performace on the training and validation sets 
- Paying attention to losses (the lower the better).

At this point, the model can be improved upon. The next step will capture hyperparameters tuning processes 

**Step 4 ==> Hyperparameters Tuning of the base model**

- Experiment by adjusting learning rate with different values of learning rates (best learning rate will be chosen)
- Experiment by adding inner dense layers to the networks  
-- Experiment by regularizing (freezing out) part of the images to enhance it's generalization learning. Note, dropout will be introduce in this aspect with different values.  
- Check pointing the processes to save the best model

**Step 5 ==> Data Augmentation**

The idea behind Data Augmentation is creating more data from the existing dataset. This will determine if creating more data from the existing dataset can help improve the performance of the model. 

To do this, I'll be following the below image transformation guidelines:
- Flip ==> horizontal, vertical flipping, or both. 
- Rotation 
- Shifting ==> up, down, right, and left 
- Shear 
- Zoom (in/out)
- Brightness/contrast 

**Step 6 ==> Training a model on large image size**

- Training the model with 299x299 image size without data augmentation
- Training the model with 299x299 image size using some data augmentation techniques
- Checkpoint the best performed model

**Step 7 ==> Using the best performed model**

The best performed model will be tested using a sample image 


## Model Deployment

Now that we've a built a working model that its prediction accuracy stands at **96%**, what next? 

What we've only done is to finish first phase of the project workflow. A model built without deploying to production is equally useless. At this point, if the model is not deployed, John should just accept his faith because no solution is on the way. It wouldn't be that bad if John fast for a day, then the next day, he can find alternative to his hungriness. 

However, for the benefit of drawing water from a dry well, this project workflow will proceed to the next phase. Creating a solution around this model will be both beneficial to the solution provider and solution receiver depending on the business and financial acumen of the solution provider. 

Below are the files which contain model deployment dependencies;
- [lambda_function.py](lambda_function.py) 
- [Dockerfile](Dockerfile)
- [test.py](test.py)

To understand the frameworks on how the model was deployed, check [Kitchenware_Image_Classification.ipynb](Kitchenware_Image_Classification.ipynb) notebook. 

# Concluding Part 

With the model deployed, John can be rest assured that a solution has been created for him to make use of in his plight to understand each kitchenware in his kitchen. However, note that the model was trained on six classes. Meaning, the model can still be managed and improved upon by training the model with new images and addictional images classes. 
