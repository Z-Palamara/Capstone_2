# Capstone 2
## Conducting NLP on Amazon Store Review Data

![amazon_logo.png](https://github.com/Z-Palamara/Capstone_2/blob/master/Visualizations/amazon_logo.png)

## Source Data
The Electronics Dataset can be downloaded from this link.
I used the 5-core dataset under the "Small" subsets for experimentation section.
- https://nijianmo.github.io/amazon/index.html

## Project Aim and Background
### Problem Statement
The goal of this project was to perform sentiment analysis on product reviews in the Amazon store. The target audience for this project are sellers on the Amazon store. They will be able to use this model to analyze consumer behavior and their response to products. This information will be useful in deciding on what new products to bring to the market and how to improve existing products. Additionally, this data can be used to help recommend products to consumers based on their behavior tendencies.

### Dataset and Source
I am using a data set from the following website: Amazon Review Data (2018). The data is in a .json format and can be easily read into a Pandas df for EDA and pre-processing. Each review is a single record in the dataset containing the following features:
    - reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    - asin - ID of the product, e.g. 0000013714
    - reviewerName - name of the reviewer
    - vote - helpful votes of the review
    - style - a dictionary of the product metadata, e.g., "Format" is "Hardcover"
    - reviewText - text of the review
    - overall - rating of the product
    - summary - summary of the review
    - unixReviewTime - time of the review (unix time)
    - reviewTime - time of the review (raw)
    - image - images that users post after they have received the product

### Project Goals
I used Natural Language Processing (NLP) to solve this problem. More specifically, I used Fine-Grained Sentiment Analysis which is a common method for determining sentiment in 5-Star reviews. After performing EDA I tested out a variety of methods for feature engineering. Some of the methods I used include text vectorization using a bag-of-words or bag-of-ngrams. After the features were properly cleaned and engineered, I used three different classifiers to construct my machine learning models for text and numerical features.

## Data Cleaning and EDA
### Initial Findings
The first step I took in examining the data was to look at a simple count plot of reviews broken out by their respective 5-star rating.
[review count 5 star.png](https://github.com/Z-Palamara/Capstone_2/blob/master/Visualizations/review%20count%205%20star.png)

## Files in Repository
1.  - [Capstone 2 - Data Cleaning and EDA Google Colab Notebook](https://drive.google.com/file/d/1clk3MyDHAcwy9FYOKEdWg--82H4ijO1F/view?usp=sharing)
    - **Please open this file in Google Colab to view all visualizations and interactive plots!**
    - Contains code used in Data Cleaning and EDA
    
2.  - [Capstone 2 - Modeling](https://github.com/Z-Palamara/Capstone_2/blob/master/Capstone_2_Modeling.ipynb)
    - Contains code used to develop Machine Learning Models
  
3.  - [Capstone 2 - Milestone Report 1](https://github.com/Z-Palamara/Capstone_2/blob/master/Capstone%202%20-%20Milestone%20Report%201.pdf)
    - Write up on progress at the halfway point of the project. (Before Modeling)

4.  - [Capstone 2 - Milestone Report 2](https://github.com/Z-Palamara/Capstone_2/blob/master/Capstone%202%20-%20Milestone%20Report%202.pdf)
    - Final write up including Data Cleaning & EDA and Machine Learning Models
  

