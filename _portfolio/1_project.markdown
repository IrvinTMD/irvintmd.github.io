---
layout: post
title: Capstone Project GA
description: a project with a background image
img: /img/12.jpg
---


<h3>Recommender System</h3>
<br/>
<h4>Objective</h4>
<p>	The aim of this project is to dive into the world of Recommender Systems, and explore various 
	methods for Collaborative Filtering. The focus will be on understanding the math and algorithm 
	behind them, then applying them to generate recommendations. Although I chose book reviews, 
	it does not actually matter what the item is. Collaborative Filtering is based on seeking similar 
	reviews amongst users/items, and predicting ratings from the existing ratings themselves. 
	Therefore, it is considered item-agnostic.
</p>

<h4>Data</h4>
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

<h4>Clean and Prepare Data</h4>
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
	the data to about 100 thousand reviews, by setting a minimum of 40 reviews for each user/item.
	<br>

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

<div class="img_row">
	<img class="col one" src="{{ site.baseurl }}/img/1.jpg" alt="" title="example image"/>
	<img class="col one" src="{{ site.baseurl }}/img/2.jpg" alt="" title="example image"/>
	<img class="col one" src="{{ site.baseurl }}/img/3.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/5.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	This image can also have a caption. It's like magic. 
</div>

You can also put regular text between your rows of images. Say you wanted to write a little bit about your project before you posted the rest of the images. You describe how you toiled, sweated, *bled* for your project, and then.... you reveal it's glory in the next row of images.


<div class="img_row">
	<img class="col two" src="{{ site.baseurl }}/img/6.jpg" alt="" title="example image"/>
	<img class="col one" src="{{ site.baseurl }}/img/11.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	You can also have artistically styled 2/3 + 1/3 images, like these.
</div>


<br/><br/><br/>


The code is simple. Just add a col class to your image, and another class specifying the width: one, two, or three columns wide. Here's the code for the last row of images above: 

	<div class="img_row">
	  <img class="col two" src="/img/6.jpg"/>
	  <img class="col one" src="/img/11.jpg"/>
	</div>
