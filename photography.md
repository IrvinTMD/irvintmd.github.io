---
layout: page
permalink: /photography/
title: photography
description: My photography journey
---

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/oze.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	Oze National Park, Japan
</div>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/nagatoro.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	Nagatoro, Japan
</div>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/oze_mist.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	Oze National Park, Japan
</div>

<div class="img_row">
	<img class="col three" src="{{ site.baseurl }}/img/pitta.jpg" alt="" title="example image"/>
</div>
<div class="col three caption">
	Blue-Winged Pitta at Bidadari
</div>

<ul class="post-list">
{% for poem in site.story reversed %}
    <li>
        <h2><a class="poem-title" href="{{ poem.url | prepend: site.baseurl }}">{{ poem.title }}</a></h2>
        <p class="post-meta">{{ poem.date | date: '%B %-d, %Y — %H:%M' }}</p>
      </li>
{% endfor %}
</ul>