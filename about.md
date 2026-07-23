---
layout: default
title: About
permalink: /about/
---

<article class="post">
    <header>
        <h1>About</h1>
        <p class="tagline">A page about me, written by me. Impartiality guaranteed.</p>
    </header>

    <p>A proper introduction is coming soon.</p>

    <h2>Contact Me</h2>
    <p>Want to talk shop, ask a question, or tell me I'm wrong on the internet? Go on then.</p>
    <p id="contact-pending">The contact form is being wired up as we speak. Check back shortly.</p>
    <form class="contact-form" data-endpoint="https://formspree.io/f/xwvgrezq" novalidate hidden>
        <label for="cf-email">Your email</label>
        <input id="cf-email" type="email" name="email" placeholder="you@example.com" required>
        <label for="cf-subject">Subject</label>
        <input id="cf-subject" type="text" name="_subject" placeholder="Something snappy" required>
        <label for="cf-message">Message</label>
        <textarea id="cf-message" name="message" rows="6" placeholder="Say your piece." required></textarea>
        <div class="contact-error sub-error" hidden></div>
        <button class="btn-pill" type="submit">Send it <span>&rarr;</span></button>
    </form>
</article>
