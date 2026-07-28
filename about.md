---
layout: default
title: About
permalink: /about/
---

<article class="post">
    <header>
        <h1>Welcome</h1>
        <p class="tagline">Please &ndash; step into my shop.</p>
    </header>

    <img class="about-portrait" src="/assets/images/ollie.jpg" alt="Ollie, in black and white, looking sassy">
    <p>This is my dumb little corner of the Internet. It's partly meant to be a record of all that I've worked on, but it's also partly my playground. <a href="https://html5zombo.com/">Anything is possible</a>.</p>
    <p>As a designer, I like to experiment. I also believe strongly in the power of play, and playfulness in design. Too much of the web is boring and dead now, whereas I cut my teeth in <a href="https://www.newgrounds.com/">the good old days of Flash media</a>.</p>
    <p>While terminal-stage capitalism (glory be) has ground most of us into a fine paste at this point, this website also represents a <em>tiny</em> act of rebellion.</p>
    <p>It contains a few serious things, but it's also deeply unserious. You can think of this as the headquarters of "the resistance".</p>
    <p>Our coffee might be crap but &mdash; hey &mdash; we have <mark>heart and soul</mark>.</p>

    <h2>Contact Me</h2>
    <p>Want to talk shop, ask a question, or tell me I'm wrong on the internet? Go on then.</p>
    <p id="contact-pending">The contact form is being wired up as we speak. Check back shortly.</p>
    <form class="contact-form" data-endpoint="https://formspree.io/f/xwvgrezq" novalidate hidden>
        <label for="cf-email">Your email</label>
        <input id="cf-email" type="email" name="email" placeholder="you@example.com" required>
        <label for="cf-subject">Subject</label>
        <input id="cf-subject" type="text" name="_subject" placeholder="Something snappy" required>
        <label for="cf-message">Message</label>
        <div class="message-box">
            <textarea id="cf-message" name="message" rows="6" placeholder="Let it all out." required></textarea>
            <button class="btn-pill" type="submit">Send it <span>&rarr;</span></button>
        </div>
        <div class="contact-error sub-error" hidden></div>
    </form>
</article>
