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