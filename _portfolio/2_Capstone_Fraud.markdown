---
layout: post
title: Fraud Detection
description: Capstone Project II
img: /img/credit_card.jpg
---
November 17, 2017<br>
<p>
    For details and code, please visit my <a href="https://github.com/IrvinTMD/My-DSI-Projects/blob/master/Capstone%20II/fraud_detection.ipynb"><b>Jupyter Notebook</b></a>.<br>
    Project is still in progress!
</p>
<h3>Credit Card Fraud Detection</h3>
<br/>

<h4>Objective</h4>
Detect Fraud on severely imbalanced dataset

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/loading.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	Content is being prepared. Coming soon.
</div>
<hr>
<b><font size="+1">Objective</font></b>
<p>	
	The aim of this project is to experiment different approaches to dealing with fraud detection and to understand the pros and cons of each. Recall score is our priority because the consequences are more dire if we misclassify a true fraud case as non-fraud. Precision score is important as well, but its impact is less serious than recall score in terms of credit card fraud. Choosing or balancing between both of these scores will depend on the context of the problem; e.g. real world business needs (whether or not we require a maximum score for either metric, or a balance of both).
</p>

<b><font size="+1">Data</font></b>
<p>	
	This is a popular dataset from <a href="https://www.kaggle.com/dalpozz/creditcardfraud">Kaggle</a>.
	Out of 284,807 data rows, only 492 are fraudulent transactions. That's extremely imbalanced at 0.173%!<br>
	<br>
	The dataset contains transactions made by credit cards in September 2013 by european cardholders, over a period of two days. It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original features and more background information cannot be provided. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.<br>
	'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.<br>
	'Amount' is the transaction Amount, this feature can be used for example-dependent cost-senstive learning.<br>
	'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
</p>

<b><font size="+1">Overview</font></b>
<p>	
	The data has PCA features V1 ~ V28, Time, Amount, and Class. For our initial analyses, we will be putting aside the Time and Amount and focus on testing models on just the 28 features.
	<ul>
		<li>Exploratory Data Analysis</li>
		<li>Manual Under-Sampling modelled with Logistic Regression</li>
		<li>Sampling methods with IMBLearn package</li>
			<ul>
				<li>Logistic Regression</li>
				<li>Random Forests</li>
				<li>XGBoost</li></ul>
		<li>Ensemble Sampling</li>
		<li>Autoencoder Neural Net</li>
		<li>Cost Sensitive Learning</li>
	</ul>
</p>

<b><font size="+1">Exploratory Data Analysis</font></b>
<p>
	First, we visualized correlations using seaborn's heatmap. Since the V features are obtained through Principal Components Analysis, we would not expect them to be correlated. Next, we took a look at 'Amount' vs 'Time' to see whether it might be an important component to determine fraudulent cases.
</p>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/fd_heatmap.jpg" alt="" title="Correlation Heat Map"/>
</div>
<div class="col three caption">
	Heatmap to visualize correlations between features.
</div>
<br>
<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/fd_time_amount.jpg" alt="" title="Time VS Amount Plot"/>
</div>
<div class="col three caption">
	Scatter Plot of Time VS Amount primarily used to visualize the importance of either feature.
</div>
<br>
As we expected, there were no correlation between any of the V features, which is great. Here, we can also get an intuition which V feature correlates with the Class column. We might make use of this information for feature selection/engineering later.<br>
The Time VS Amount plot revealed that Amount might not be a strong indicator of fradulent transactions. As we can see, most of the green dots (fraud) fall below ~2500. Much of the non-fraud cases are also below that amount.<br>
<br>
Let's take a closer look at each of our features now. I ran a loop to make distribution plots for every V feature. There are a total of 28 plots, so I shall not clutter this page with them (click <a href="https://github.com/IrvinTMD/My-DSI-Projects/blob/master/Capstone%20II/fraud_detection.ipynb"><b>here</b></a>) to view the Jupyter Notebook).<br>
<br>
Finally, we get to our Class labels. A simple bar plot was created, and the percentage fraud was calculated.

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/fd_ratio.jpg" alt="" title="Class Ratio Plot and Numbers"/>
</div>
<div class="col three caption">
	Bar Plot to visualize class ratio, and printouts to show numbers.
</div>
<br>
<b>What an imbalanced dataset!</b> Out of over 280,000 rows, only 492 are fraudulent transactions. That equates to about 0.173%. Isn't that akin to finding a needle in a haystack?<br>
<br>
<b>What does this mean?</b> Having a ratio like this, we must know that we cannot make use of conventional/traditional accuracy measures as our performance metric. In fact, even our approach to sampling and modelling will be very different.<br>
<br>
Simple accuracy scores are not feasible because even a dumb machine, which predicts non-fraud all the time, will get our baseline accuracy score of 100.0 - 0.173 = 99.827%! This might appear awesome as an article headline (thus, please remember to always be critical when reading/evaluating articles), but it is far from the truth. It does not help at all in understanding how our model performs in detecting fraud cases.<br>
<br>
What do we do then? We make use of <b>precision</b> and <b>recall</b> scores to help us evaluate our models. Recall (rate of False Negative) refers to how many actual frauds we are able to detect. Whereas, Precision (False Positive) refers to how many of our machine's predicted fraud cases are real, actual frauds. Ideally, we want to get high scores for both of them. However, reality is hardly kind. In this case of Credit Card Fraud, recall scores are more crucial. If a transaction is fraudulent and we predict it as normal, the consequence is much greater than if we predict a normal case as fraud. Therefore, our main focus will be on recall scores, but of course, we will also optimize the precision score, and allow our models to 'tune' between placing emphasis on either. Flexibility!<br>
<br>

<b><font size="+1">Manual Undersampling</font></b>
<p>
	We'll try a simple manual undersampling of the over-represented class first, by randomly selecting 492 rows out. We would now have 492 fraud and 492 normal transactions. We then prepare both of our datasets (whole data and undersampled data) by splitting them into train and test sets for further usage. A basic Logistic Regression is used to model on the undersampled data.
</p>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/fd_undersample.jpg" alt="" title="Undersampling Code"/>
</div>
<div class="col three caption">
	A simple manual undersampling.
</div>
<br>
<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/fd_size.jpg" alt="" title="Train Test Size and Ratio"/>
</div>
<div class="col three caption">
	The sizes and ratios of train-test-splits on both dataset versions
</div>
<br>

<b><font size="+1">Logistic Regression on Under-Sampled Data</font></b>
<p>
	A gridsearch was performed to obtain the best hyperparameters for the model. We will be predicting on 3 sets of data.
	<ul>
		<li>Undersampled Test Set</li>
		<li>Whole Data Test Set</li>
		<li>Lastly, we try to train on the whole data, and predict to see how it goes!</li>
	</ul>
	Before we continue, we will build a function to return scores for us. Our function will print out sklearn's classification report (shows precision, recall, f1 score, support), plot a ROC curve if specified, and returns a confusion matrix in dataframe format.
</p>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/fd_resfunc.jpg" alt="" title="Function for our performance metrics"/>
</div>
<div class="col three caption">
	Prints classification report, plots ROC curve if specified, and returns confusion matrix.
</div>
<br>
<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/lr_under_test.jpg" alt="" title="Predict on undersampled test set"/>
</div>
<div class="col three caption">
	Prediction on undersampled test set, with model trained on undersampled train set.
</div>
<br>
<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/lr_whole_test.jpg" alt="" title="Predict on whole data set"/>
</div>
<div class="col three caption">
	Prediction on whole data test set, with model trained on undersampled train set.
</div>
<br>
<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/lr_whole_traintest.jpg" alt="" title="Function for our performance metrics"/>
</div>
<div class="col three caption">
	Prediction on whole data test set, with model trained on whole data train set.
</div>
<br>
The model performed 
















