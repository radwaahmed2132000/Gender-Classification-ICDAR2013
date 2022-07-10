# Gender-Classification-ICDAR2013
This is a gender classifer that based on a given sample of handwritten text decides if the origin of the text is male or female. This model won first place 🥇 in the competition that corresponded to the project in fulfillment of the classwork requirements of the neural networks course taught to computer engineering juniors in Cairo University.

## Datasets & Preprocessing 💾 
We initially considered handwritten samples from both the dataset that was collected <a href="https://www.kaggle.com/datasets/essamwisamfouad/cmp23-handwritten-males-vs-females"> from our class</a> and the <a href="https://www.kaggle.com/competitions/icdar2013-gender-prediction-from-handwriting" >ICDAR2013 dataset </a> but the final model (which was known to be tested on a CMP23 test set) only uses the former. The relevant folder with the two files responsible of preprocessing (filtering and whitespace removal) and reading the images is the "Preprocessing" folder. <br> <br>
This is a sample from the dataset: <br>
<img width="681" alt="image" src="https://user-images.githubusercontent.com/49572294/178151477-10c9450b-c9e0-4e61-a22b-cd7cc5bd5c1c.png">

## Features Extracted 🤳
We have considered GLCM, HoG, LBP, Fractal, COLD and, Hinge features along with a feature chef that tried all possible combinations of them. Each of these is discussed in detail in the project's report. Only Hinge features made it to the final model. The "Features" and "Combined_Features" folders include the relevant models. <br> <br>
This is an example for feature visualization (fractal features): <br>
<img width="691" alt="image" src="https://user-images.githubusercontent.com/49572294/178151507-2994093f-6cc4-48bc-966b-48dcf88493e3.png">


## Models Considered 🕹️
We have considered NN, CNN, Random Forest, SVM, XGboost, Adaboost. Because both accuracy and performance mattered for the project (along with other constraints) only SVM made it to the final model. You can read more on that in the project's report. The "Deep Learning" and "Models" folders include the relevant models. 

## Running the Project 🚀
If you are a developer then you know how to navigate to the corresponding model/feature extractor/preprocessing module and run it. Otherwise, to test the final model you can run evaluate.py in the "Submissions" folder while having the test data in the test folder with labels in the groundtruth text file. When you finish you will find the model results and time taken in the "out" folder inside.

## Collaborators

<!-- readme: collaborators -start -->
<!-- readme: collaborators -stop -->
