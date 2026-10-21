---
layout: default
title: Fun
permalink: /fun/
---

<article class="post projects">
    <header>
        <h1>Fun</h1>
        <p class="tagline">Projects built purely for the joy of it.</p>
    </header>

    <div class="pgrid">
        {% comment %} Newest first, by start year. The List links to a post, so it is placed by hand at its start year, 2020. {% endcomment %}
        {% capture the_list %}
        <a class="pcard" href="{{ '/posts/14/the-list/' | relative_url }}">
            <div class="pcard-inner">
                <div class="pcard-media"><img src="{{ '/assets/images/posts/the-list-header.jpg' | relative_url }}" alt="A woodcut engraving of a stern bearded scholar recording forbidden words with a quill"></div>
                <div class="pcard-body">
                    <h3 class="pcard-title"><span class="pcard-name">The List</span> <span class="cv-year">2020 – Present</span></h3>
                    <p class="pcard-kicker">An essential reference for the modern professional. Discretion is advised.</p>
                </div>
            </div>
        </a>
        {% endcapture %}
        {% assign funs = site.fun | sort: "start" | reverse %}
        {% assign listed = false %}
        {% for f in funs %}
        {% if listed == false and f.start < 2020 %}{{ the_list }}{% assign listed = true %}{% endif %}
        <a class="pcard" href="{{ f.url | relative_url }}">
            <div class="pcard-inner">
                <div class="pcard-media"><img src="{{ f.image | relative_url }}" alt="{{ f.title }}"></div>
                <div class="pcard-body">
                    <h3 class="pcard-title"><span class="pcard-name">{{ f.title }}</span> <span class="cv-year">{{ f.year }}</span></h3>
                    <p class="pcard-kicker">{{ f.kicker }}</p>
                </div>
            </div>
        </a>
        {% endfor %}
        {% if listed == false %}{{ the_list }}{% endif %}
    </div>

    <p class="projects-fun-link">After something more serious? The grown-up work lives in <a href="{{ '/projects/' | relative_url }}">Projects</a>.</p>
</article>
