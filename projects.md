---
layout: default
title: Projects
permalink: /projects/
---

<article class="post projects">
    <header>
        <h1>Projects</h1>
        <p class="tagline">A decade of building tools for people who build things.</p>
    </header>

    <h2 id="work">Work</h2>
    <div class="pgrid">
        {% assign work = site.projects | where: "section", "work" | sort: "order" %}
        {% for pr in work %}
        <a class="pcard" href="{{ pr.url | relative_url }}">
            <div class="pcard-inner">
                <img src="{{ pr.image | relative_url }}" alt="{{ pr.title }}">
                <div class="pcard-body">
                    <h3 class="pcard-title">{{ pr.title }} <span class="cv-year">{{ pr.year }}</span></h3>
                    <p class="pcard-kicker">{{ pr.kicker }}</p>
                </div>
            </div>
        </a>
        {% endfor %}
    </div>

    <h2 id="open-source">Open Source</h2>
    <div class="pgrid">
        {% assign oss = site.projects | where: "section", "open-source" | sort: "order" %}
        {% for pr in oss %}
        <a class="pcard" href="{{ pr.url | relative_url }}">
            <div class="pcard-inner">
                <img src="{{ pr.image | relative_url }}" alt="{{ pr.title }}">
                <div class="pcard-body">
                    <h3 class="pcard-title">{{ pr.title }} <span class="cv-year">{{ pr.year }}</span></h3>
                    <p class="pcard-kicker">{{ pr.kicker }}</p>
                </div>
            </div>
        </a>
        {% endfor %}
    </div>

    <p class="projects-fun-link">Looking for the silly stuff? That lives over in <a href="{{ '/fun/' | relative_url }}">Fun</a>.</p>
</article>
