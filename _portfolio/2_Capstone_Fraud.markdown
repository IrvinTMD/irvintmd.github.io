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
It comes without surprise that the model performs well on its own undersampled test set. However, it feels quite impressive that it managed to achieve a pretty high recall score of 0.91 on the whole dataset! Granted, the precision score leaves quite a lot to be desired, but, we must remember that this is merely a Logistic Regression model on 984 data points (undersampled). I'd say it did really well for its simplicity. Of course, we would not be stopping here. Regardless, this result is aligned with the direction we're headed; a heavier focus on recall.<br>
<br>
For the last model, we trained the entire data set and predicted. As expected, the scores are not good, as the training set is in itself imbalanced at 0.173%. We are missing out at least 40% of all fraudulent transactions here. That's a significant impact in the real world. This model was done just for constrastive purposes to illustrate the importance of sampling techniques on imbalanced datasets. Even a simple small scale undersampling gave us a much higher recall score.<br>
<br>

<b><font size="+1">More Modelling with IMBLearn Sampling Techniques</font></b>
<p>
	IMBLearn (short for imbalanced-learn) is a python package offering a number of <b>re-sampling techniques</b> commonly used in datasets showing strong between-class imbalance. It is compatible with scikit-learn and is part of scikit-learn-contrib projects. For more information, please visit the <a href="http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html">documentation</a>.<br>
	<br>
	The first technique we will try is the popular <b>SMOTE</b>, which means Synthetic Minority Over Sampling Technique. SMOTE has proven effectiveness over lots of applications, and several papers/articles have had success with it. What SMOTE does is that it constructs synthetic samples from the under-represented class by making use of nearest neighbours. It is sort of a bootstrapping method. Therefore, in our case, the train set has about 213,000 rows. SMOTE will use the 492 fraud data points as reference to create synthetic samples up to the number of rows of the over-represented class, which is ~213,000 (to get 50/50 ratio).
</p>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/smote_example.jpg" alt="" title="SMOTE"/>
</div>
<div class="col three caption">
	Example code to SMOTE.
</div>
<br>
After SMOTE'ing, we perform the usual gridsearch to find hyperparameters for the subsequent Logistic Regression model. We always start modelling with Logistic Regression because it is the simplest model and can serve as a baseline for us to understand how the dataset fares. In addition, if Logistic Regression happens to be sufficient in giving us good scores, we could just stick with it! Simplicity is beauty.<br>
<br>
This time, due to having much more rows (over 400,000), the gridsearch process had to take about 30 minutes to complete. The results were 'underwhelming' though. It seemed like the value of C does not really change the score, which hangs around 0.94 all the time. We must be reminded though, that the score reflected through gridsearch is not precision or recall but a simple accuracy measure. Thus, it may not be reliable after all in our case. In fact, I had to manually run an empirical analysis to determine a decent C value, which was also 'underwhelming' because they do not have significant differences. As a result, I decided on a smaller C value for computation speed.<br>
<br>
Hereon, we will make use of <b>pipelines</b> for cleaner code and ease of operations as we would be running many models. It is important to note that IMBLearn's pipeline is different from that of sklearn's. The latter is unable to accept the former's sampling technique as part of its pipeline. Thus, please make sure you are importing and using IMBLearn's pipeline.<br>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/smote_lr.jpg" alt="" title="SMOTE Logistic Regression"/>
</div>
<div class="col three caption">
	SMOTE Logistic Regression after gridsearching.
</div>
<br>
Recall score seemed to maintain (in comparison to our 'benchmark' undersampling LR earlier), but a slight improvement was observed in precision. Unfortunately, that is not our main goal at this point. Let's move on to try other sampling techniques. The next will be <b>Edited Nearest Neighbours</b> (ENN undersampling).<br>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/enn_lr.jpg" alt="" title="ENN Logistic Regression"/>
</div>
<div class="col three caption">
	ENN Logistic Regression results.
</div>
<br>
Wow! This is the first time we're seeing double digits for both our FP and FN. The recall score is not as high as before, but the precision score flew up to about 80%; very close to our recall. While these scores might seem very balanced and fairly high, we should not be satisfied because our goal is to maximize recall. Here, we are still missing out on ~20% of frauds. Now, how about we combine SMOTE and ENN? It's <b>SMOTEENN</b> time.<br>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/smoteenn_lr.jpg" alt="" title="SMOTEENN Logistic Regression"/>
</div>
<div class="col three caption">
	SMOTEENN Logistic Regression results.
</div>
<br>
Well, a combination of both did not give us much difference in results. Let's a take a step further and adjust the <b>probability thresholds</b> to see if it is possible to balance FP and FN. By using .predict_proba(), the probability values for a data point to be Class 0 or 1 are provided. We can turn it into a dataframe, and add a new column for our manual prediction that is based on adjusting the threshold. For example, if data point 100 has a probability of 0.65 to be Class 1, the default threshold of 0.5 will classify this point as 1. If we adjust the threshold to be 0.7 (which is higher than the probability value), the data point will be classified as 0 since it is below 0.7. <br>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/adjusth_manual.jpg" alt="" title="Manual Threshold Adjustment"/>
</div>
<div class="col three caption">
	To achieve 100% recall, the threshold had to be 0.0001!
</div>
<br>
Through empirical testing, we found that the threshold (for our SMOTE model) had to be 0.0001 in order to achieve 100% recall. However, it came at a severe cost on precision. Given these values, the machine might as well predict every point as fraud! Moving on, we will be adjusting thresholds often, so, we built a function for it.<br>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/adjusth_func.jpg" alt="" title="Threshold Adjustment function"/>
</div>
<div class="col three caption">
	A function to return results upon probability threshold adjustment
</div>
<br>
The function takes in the .predict_proba() values, the true y values, and a user-specified threshold value. It prints out the confusion matrix and returns the adjusted prediction values of 0 and 1.<br>
<br>

<b><font size="+1">Random Forests</font></b>
<p>
	I tried to run a gridsearch which took hours overnight and crashed the kernel. Thereafter, I manually tested several parameters and found these to produce better results.<br>
	<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/rf_smote.jpg" alt="" title="Random Forest SMOTE"/>
	</div>
	<div class="col three caption">
		Random Forest SMOTE sampling.
	</div>
	<br>
	We get slightly better recall and precision scores at least ~0.80 each. However, the same problem remains. The results do not match our goal in minimizing recall scores and False Negatives. Next, we try to train a RF on our manually undersampled data.<br>

	<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/rf_under.jpg" alt="" title="Random Forest on Undersampled data"/>
	</div>
	<div class="col three caption">
		Our undersampled data seems to always perform well in recall scores.
	</div>
	<br>
	Let's see if we can get a better score by merging these two models together; ensembling. Both model's predicted probabilities are added together and averaged. Then, the threshold is adjusted to see how the model does in terms of precision vs recall.<br>
	<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/rf_ensemble.jpg" alt="" title="Ensembling two RF models"/>
	</div>
	<div class="col three caption">
		Two Random Forest models (one trained on whole data, another on undersampled) are ensembled to generate new predictions.
	</div>
	<br>
	An <b>ensemble</b> of both models did give us an improvement! The recall score remained from the undersampled model (2 FN), and the precision was improved quite significantly to 0.12, at 883 False Positives. This looks promising. We will attempt even more ensembling later to boost our scores.
</p>

<b><font size="+1">eXtreme Gradient Boosting XGBoost</font></b>
<p>
	



	
</p>










<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/loading.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	More content is being prepared. Check back soon.
</div>
<hr>

