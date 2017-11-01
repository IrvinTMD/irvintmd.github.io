---
layout: post
title: Capstone Project GA
description: a project with a background image
img: /img/12.jpg
---


<h3>Recommender System</h3>
<br/>

<h4>Objective</h4>
<p>The aim of this project is to dive into the world of Recommender Systems, and explore
	various methods for Collaborative Filtering. The focus will be on understanding the math
	and algorithm behind them, then applying them to generate recommendations. Although I
	chose book reviews, it does not actually matter what the item is. Collaborative Filtering
	is based on seeking similar reviews amongst users/items, and predicting ratings from the
	existing ratings themselves. Therefore, it is considered item-agnostic.


<h4>Data</h4>
<p>Data: Amazon's Product Review and Data, 1996 - 2014<br>
File: gzip file containing json within<br>
<br>
The dataset consists of over 142 million reviews, of which 22 million are book reviews.<br>
However, I will only be using about 100 thousand book reviews due to time/memory constraints,<br>
by setting a minimum of 40 reviews for each book (item) and each user.</p>

<h4>Clean and Prepare Data</h4>
<p>There is..</p>

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
