---
layout: post
title: Recommender System
description: Capstone Project at GA
img: /img/12.jpg
---
November 17, 2017<br>
This is my capstone project for the Data Science Immersive program at GA. The program is a 12-week 
full-time course with five projects in total. It covers a very wide range of Data Science topics, 
but unfortunately for me, Recommender Systems were only covered for about half a day, in week 11! 
Therefore, the entire project was pretty much a self-learning journey, from scratch. 

<b><font size="+1">Objective</font></b>
<p>	The aim of this project is to dive into the world of Recommender Systems, and explore various 
	methods for Collaborative Filtering. The focus will be on understanding the math and algorithm 
	behind them, then applying them to generate recommendations. Although I chose book reviews, 
	it does not actually matter what the rated item is. Collaborative Filtering is based on seeking 
	similar reviews amongst users/items, and predicting ratings from the existing ratings themselves. 
	Therefore, it is considered item-agnostic.
</p>

<b><font size="+1">Data</font></b>
<p>	Data: Amazon's Product Review and Data, 1996 - 2014<br>
	File: gzip file containing json within<br>
	<br>
	The dataset consists of over 142 million reviews, of which 22 million are book reviews.<br>
	However, I will only be using about 100 thousand book reviews due to time/memory constraints,<br>
	by setting a minimum of 40 reviews for each book (item) and each user.<br>
	<br>
	Reference (source of data):<br>
	R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016
</p>

<b><font size="+1">Clean and Prepare Data</font></b>
<p>	The gzip data is loaded with the following functions. It takes a while to process, so, it is 
	recommended to pickle the dataframes after loading.<br>

	{% highlight python %}
	def parse(path):
   		g = gzip.open(path, 'rb')
    	for l in g:
        	yield eval(l)

	def getDF(path):
    	i = 0
    	df = {}
    	for d in parse(path):
        	df[i] = d
        	i += 1
    	return pd.DataFrame.from_dict(df, orient='index')
	{% endhighlight %}

	<br>
	Initially, I attempt computing with 8 million book reviews, which took a serious toll on my
	system. It couldn't even finish computing! Thereafter, I reduced the size bit by bit, but
	unfortunately, the computation time was still high. Upon further consideration, as my purpose
	for the project was only to gain mastery of the methods for Collaborative Filtering, I shrunk
	the data to about 100 thousand reviews, by setting a minimum of 40 reviews for each user/item.<br>

	{% highlight python %}
	def downsize(df, u_col, i_col, min_r_count):
    	"""
    	Takes in a dataframe that includes columns "user", "item", "ratings"
    	and returns a new dataframe where users and items each have a
    	minimum amount of ratings as specified.

    	Arguments
    	=========
    
    	df : (dataframe)
        	Dataframe that has columns User, Item, and Ratings
        
    	u_col: (string)
        	Name of User Column
        
    	i_col: (string)
        	Name of Item Column

    	min_r_count: (int)
        	Mininum number of ratings for each user and item
        
    	"""
    	size = df.shape[0]
    	while True:
        	user5 = df.groupby(u_col).size().reset_index(name='count')
        	user5 = user5[user5['count'] < min_r_count]
        	df = df[~df[u_col].isin(user5[u_col])]

        	item5 = df.groupby(i_col).size().reset_index(name='count')
        	item5 = item5[item5['count'] < min_r_count]
        	df = df[~df[i_col].isin(item5[i_col])]
        	print df.shape
       		if df.shape[0] == size:
            	break
        	else:
            	size = df.shape[0]
    	print df[u_col].nunique(), 'users'
    	print df[i_col].nunique(), 'items'
    	return df
	{% endhighlight %}

<br>
	From the downsized dataframe, we have 1490 unique users, and 1186 unique items/books. Some
	exploratory work was done to view users/items with the highest number of reviews. I also
	created a dataframe to reference books with their original ID, name, and a new sequential
	ID obtained by using LabelEncoder. This dataframe is mostly used for intuitively understanding
	the recommendation results later.<br>
	<br>
	Before proceeding to split the data into train and test sets, I calculated the sparsity to
	get a better understanding of what we are dealing with. The sparsity of the 100k reviews
	data set is 93.85%.
</p>

<b><font size="+1">Train-Test Split</font></b>
<p>
	In most traditional Machine Learning applications, train-test splits can be performed by simply
	separating a random selection of rows in the data. However, in Collaborative Filtering, that
	would not work because we will need all users to be in both train and test sets. If we randomly
	take rows out, we will lose a batch of users.<br>
	<br>
	Therefore, what we need to do is to mask/remove some ratings for every user on the train set. 
	There are a few ways to do this. In this case, we will split a user's ratings into two. This 
	means, if a user has rated 40 items, the train set would have 25, and the test, 15. The ratings
	do not overlap, and both train and test sets are disjoint.<br>
</p>

{% highlight python %}

def train_test_split(ratings, size):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        	size=size, replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and train are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

{% endhighlight %}

<b><font size="+1">Matrix Factorization using ALS and SGD</font></b>
<p>	For matrix factorization, we have two sets of latent vectors to solve, the user and the item. 
	In <b>Alternating Least Squares</b>, we hold one set of latent vectors constant and we solve for the 
	other non-constant vector. Now this is the alternating part. Once we have this solved vector, 
	we now hold this newly solved vector constant, and then solve again for the new non-constant 
	vector (which was the previous constant). This is done repeatedly until convergence.
</p>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/derivative.jpg" alt="" title="Derivative"/>
</div>
<div class="col three caption">
	Item vector is constant, and this equation takes the derivative of the loss function with respect to 
	the user vector.
</div>

<br>
<p>
	That was ALS. Now, let's move on with <b>Stochastic Gradient Descent</b> (SGD). In SGD, we also take 
	the derivative of the loss function, but now it is with respect to each variable in the model. The 
	derivative is taken and feature weights are updated one individual sample at a time.<br>
	In our SGD model, we will also include biases for each user and item, and a global bias. For example, 
	some users might have a high average rating, or that certain items might have an average rating that 
	is less than the global average. We would want to attempt to address this by adding biases.
</p>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/bias_loss.jpg" alt="" title="Loss function with bias and regularization"/>
</div>
<div class="col three caption">
	This is the equation for loss function, considering biases and regularization.
</div>

<br>
<p>
	


</p>





<div class="img_row">
	<img class="col one" src="{{ site.baseurl }}/img/derivative.jpg" alt="" title="example image"/>
	<img class="col one" src="{{ site.baseurl }}/img/derivative.jpg" alt="" title="example image"/>
	<img class="col one" src="{{ site.baseurl }}/img/derivative.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>




<br/><br/><br/>


The code is simple. Just add a col class to your image, and another class specifying the width: one, two, or three columns wide. Here's the code for the last row of images above: 

	<div class="img_row">
	  <img class="col two" src="/img/6.jpg"/>
	  <img class="col one" src="/img/11.jpg"/>
	</div>
