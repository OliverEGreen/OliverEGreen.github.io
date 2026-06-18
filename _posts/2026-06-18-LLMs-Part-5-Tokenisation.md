---
title: "LLMs Part 5: Tokenisation"
date: 18-06-2026
---

This is part 5 in my series on building language models. See Part 1, Part 2, Part 3 and Part 4. 

They say LLMs are just prediction machines. That's actually *entirely* accurate in my case. What they predict next can change a bit, and this affects their outputs. 

Prior to 2013?? language modelling tended to just use words as their basic units. Tokenisation was pioneered as an approach by Edinbugh University in 2018???

In this post, I explore 3 different approaches and show you the outcome of each. They are subtly different.

## Next-character prediction

Initially ran with next-character prediction. Very good at generating words in the right vibe, but sentences had little-to-no coherence. 

## Next-word prediction

It was only a few minutes' work required to re-wire this to use next-word prediction. My hope is that this would give us something slightly more readable, where the flow of words was determined probabilistically. 

## Tokenisation and next-token prediction

Next up is token-based, using BPE. Involves some very fiddly dictionary and set work. It also runs through the corpus a few thousand times before we're done with it.

Tokenisation breaks words in the corpus down into their fundamental components, based on how frequently each chunk appears in the text. It allows for learning more granular patterns, probably. But maintains much of the goodness of next-word prediction, which leads to more coherent sentences.

## Oddities

When training with tokens, you initially see some very strange stuff. At first, you get the usual untrained noise: 

image

But the earlier iterations really like to focus on teh most common, safest words. And you just wind up with repetitive preposition soup: 

image

### Harry Potter and the Persistent Footer

A facepalm moment during training. Footer text appears on every page with the book title. So the book titles kept appearing during randomly generated tamples 

image 

Used Claude to clean out the corpus and strip out any repetitive artifacts. 

