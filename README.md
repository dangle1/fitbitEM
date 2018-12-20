# Fitbit and EM

The purpose of this notebook is to implement the Expectation Maximization (EM) algorithm that was formalized by Dempster, Laird, and Rubin from scratch using Python and Numpy. Put roughly, the purpose of this algorithm is to maximize the probability of some variable explaining some data. When we suspect the data follows some probability distribution, say Gaussian, the goal is to identify which Gaussian distribution produced which data point. Thus I wanted to apply my implementation of EM to personal data that I scrape from Fitbit's API. The reason is because there were three distinct phases for me throughout the year: 1.) training for the SD Rock n' Roll marathon 2.) relaxing over the summer =) 3.) Completing my last quarter, Fall 2018, at UCSD. Thus, if I assume each phase acts a Gaussian distribution with unique means and variances, perhaps I can cluster my personal data from the year based on this assumption. 

As an academic exercise, this project was meant for me to learn the following: 

* How to implement a machine learning algorithm, specifically Expectation Maximization (EM), from scratch based on mathematical definitions.
* How to navigate the OAuth2.0 workflow in order to retrieve resources from a third-party API.
* How to understand the basic characterstics of data (mean, variance, etc...) and how that affects the execution of a machine learning algorithm. 

As a practical exercise, I wanted to identify the following: 

* Can I find significant clusters of my personal data so that future data points can be labelled a certain way, e.g. given some biometrics I can determine whether I was in training or not. 

## Data

From Fitbit I scraped the following data:
* number of steps I took in a day 
* average resting heartrate for that day 
* minutes of sleep I got the night prior. 
These features were chosen because of their ease of interpretation as well as their availability. 

## Technologies

These were the following technologies I used in this analysis: 

* Python
* Jupyter Notebook
* Numpy
* Matplotlib
* Fitbit python library (for navigatin the OAuth 2.0 workflow)

## Organization of Notebooks

Each phase of the project is split into dedicated notebooks which are structured as such: 

1. [01_Gather_Data.ipynb](01_Gather_Data.ipynb): I script the code for scraping data from Fitbit's API
2. [02_Explore_Data.ipynb](02_Explore_Data.ipynb): I perform data exploration as well as create the ground truth clusters. 
3. [03_Analyze_Data.ipynb](03_Analyze_Data.ipynb): I apply EM against the data and draw conclusions. 
4. [EM.ipynb](EM.ipynb): I implement the algorithm based on mathematical definition. 

## TL;DR

In case anyone reading this would like to see the final results without having to sift through each notebook, here are the clusterings I generated and their interpretations:

This is what the clusters look like if I use the actual periods of the year as clustering criteria. The next two clusters will try to emulate this clustering. 
![Ground truth](https://raw.githubusercontent.com/dangle1/fitbitEM/master/images/groundtruth_clusters.png)

This clustering is the result of randomly choosing cluster centers and then letting EM run its course. 
![Naive Clustering](https://raw.githubusercontent.com/dangle1/fitbitEM/master/images/naive_clusters.png)

This clustering is the result of finding the actual cluster centers of the ground truth and passing those parameters as initialization values to EM, then letting EM run its course. 
![Naive Clustering](https://raw.githubusercontent.com/dangle1/fitbitEM/master/images/smart_clusters.png)

As we can see above the two sets of cluster do not capture the ground truth very well. This is most likely because of how difficult it was to separate the data in the first place. In the future I will use other methods that are more sensitive to the temporal nature of the data, such as Hidden Markov Models and Recurrent Neural Networks. 
