# Machine Learning Algorithm for Netowrk Intrusion Detection
This repository serves as the second major assignment of IF3170 Artificial Intelligence course of Informatics Engineering in Institut Teknologi Bandung.

## Table of Contents
 1. [General Information](#general-information)
 2. [Setup](#setup)
 3. [How to Run](#how-to-run)
 4. [Contribution](#contribution)

## General Information
The UNSW-NB15 dataset is a collection of network traffic data that includes various types of cyberattacks and normal activities. In this task, three machine learning algorithms: k-Nearest Neighbors, Gaussian Naive Bayes, and Iterative Dichotomiser3, are implemented and used on multi-class problem of the UNSW-NB15 dataset.

## Setup
1. Clone the repository
   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/hafizh-ender/Tubes2_IF3170_MachineLearning.git
   ```

2. Navigate to the project directory
   Change into the repository's folder:
   ```bash
   cd Tubes2_IF3170_MachineLearning
   ```

3. Create a virtual environment
   To ensure package compatibility, set up a local environment. For example, using `venv`:
   ```bash
   python -m venv myenv
   ```

4. Activate the virtual environment  
   - On **Windows**:
     ```bash
     .\myenv\Scripts\activate
     ```

5. Install dependencies
   Use `pip` to install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
You're now ready to use the repository! Currently, we provides only the working Jupyter [notebook](https://github.com/hafizh-ender/Tubes2_IF3170_MachineLearning/blob/main/notebook.ipynb) to be look at. Just run the notebook and you're done!

## Contribution
| NIM             	| Name                             	| Contribution                                                                                                	|
|-----------------	|----------------------------------	|-------------------------------------------------------------------------------------------------------------	|
|     13121140    	|     Muhammad Fadli   Alfarizi    	| Develop the implementation of k-nearest neighbors and develop the code for data cleaning and preprocessing  	|
|     13121142    	|     Roby Pratama   Sitepu        	| Writing reports part of k-nearest neighbors, data cleaning, and data preprocessing                          	|
|     13621034    	|     Muhammad   Arviano Yuono     	| Conduct the model validation and writing reports part of model validation                                   	|
|     13621060    	|     Hafizh Renanto   Akhmad      	| Develop the implementation of and writing reports part of Gaussian Naive Bayes and Iterative Dichotomizer 3 	|