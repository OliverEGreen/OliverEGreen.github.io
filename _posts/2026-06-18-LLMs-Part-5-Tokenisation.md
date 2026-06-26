---
title: "LLMs Part 5: Tokenisation"
date: 18-06-2026
---

*This is Part 5 in my series on building language models. For more, see [Part 1](https://olliegreen.info/writing/LLMs-Part-1-Building-word2vec/), [Part 2](https://olliegreen.info/writing/LLMs-Part-2-Building-a-Vanilla-RNN/), [Part 3](https://olliegreen.info/writing/LLMs-Part-3-Building-an-LSTM/) and [Part 4](https://olliegreen.info/writing/LLMs-Part-4-Transformer-Architecture/). In this post, I explore 3 different tokenisation approaches and show you the outcome of each as they are subtly different.*

## Introduction 

You'll often hear it repeated in tech discussions that LLMs "are just next-token prediction machines". Having now built a few, I have to reiterate how very accurate this take is.

What we are doing to generate a response is to use our model a bit like a function, and generate the next token, then loop the entire output back and generate the next one after that. This carries on until we reach the end of our response.

A chat interface is something of a clever trick. What we have is a next-token prediction machine, and a hidden prompt along the lines of "what follows is a conversation between a human user and a helpful chatbot". After this, your prompt gets injected and the prediction-looping begins.  

There's a bit more complexity that I'm skipping over, partly because I don't know the answer yet, but a simple LLM would actually start generating the full back and forth between human and machine. Which wouldn't be remotely useful. 

## Tokens as prediction unit

We often hear the word 'token' in LLM contexts, but it's worth diving into what this really boils down to; the smallest-useful units of information that we parse, train upon and then generate predictions for.

"But why not just predict words?" you might be wondering. And we essentially did exactly this back in the day. If you read Part 1, our [Word2Vec implementation](https://olliegreen.info/writing/LLMs-Part-1-Building-word2vec/) is word-based. There's something pleasantly easy-to-understand about words. The vocabulary we train upon is all the unique words in the corpus. But they can be inefficient and inflexible as our basic unit of language, and well [Claude Shannon](https://en.wikipedia.org/wiki/Claude_Shannon) didn't establish [Information Theory](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) for nothing! 

Your next thought may be "how about we train on unique characters then?" and, again, this makes perfect sense. The number of unique characters in a book is far lower than the number of words, and there's a definite irreducibility to letters - you cannot break them down any further. And we did this too! Our [RNN](https://olliegreen.info/writing/LLMs-Part-2-Building-a-Vanilla-RNN/) and [LSTM](https://olliegreen.info/writing/LLMs-Part-3-Building-an-LSTM/) are next-character predictors.

So, what then, if not words, or characters? The answer is a sort of half-way between character-level and word-level approaches – what we call tokens. These are the most commonly-recurring mini-sequences of letters, modular chunks of words, like 'ed', 'ology', 'ility' or 'tion'. The closest linguistic analogy might be syllables.

The useful thing about tokens is that they can be flexibly reconfigured into any number of tenses, singulars, plurals, conditionals, etc, by making only minor substitutions for other tokens, giving them a much greater efficiency at covering all the text in our corpus. 

This big shift towards using tokens (or 'subwords') instead of words started in 2016, with [a paper from researchers from the University of Edinburgh](https://aclanthology.org/P16-1162.pdf).

## The experiments

### Next-character prediction

When I first built my dummy transformer architecture, a lot of my functions were carried over from the RNN and LSTM work, which was prwedicting characters.

What I noticed, is that it quickly gets very good at spelling, and generating words in the *right vibe*, but sentences had little-to-no coherence. It plateaus at the level of vibe soup. Using Harry Potter as our corpus:

> “Don’t like this time is,” said Harry, still nothing very little and looked like a trick. “I don’t know Hagrid ’s mother, was the only time that the dementors was going to me risk what to make it up to have its deliverable magic you’d do that looked like things of you stole out of it, I do, wait a little book, it was the first last thing the same

### Next-word prediction

It was only a few minutes' work to re-wire my functions to use next-word prediction instead. My hope is that this would give us something slightly more readable, where the flow of words was determined probabilistically. The difference is stark:

> They walked back to the door and they had arrived after the Gryffindor table. Harry heard Ron shriek, noticing Ron and Hermione’ s hand and he turned on his feet, listening nervously as he went.“ What happened?” said Hermione, stopping abruptly to her heel and staring into the room—“ You’ re not right!”“ Well, I think so,” said Hermione,“ so...”

Not only has the prediction created semi-convincing sentences and flow, it's also maintained an appropriate urgent tone throughout the scene. There are some punctuation errors (around apostrophes and quotation marks), which was a side-effect of the word-level splitting approach I took. But, overall, I'm very impressed with the coherence displayed by our transformer model.

### Next-token prediction

Next up is token-based, using BPE (byte-level encoding) which is considered the industry standard. 

This took a bit more work to plug in; it involves some very fiddly dictionary and set work. It also runs through the corpus a few thousand times before we're done with it.

Personally I didn't really like writing out the BPE algorithm. Compared to the elegance of matrix maths, linear algebra and transformer architecture, it feels incredibly brute-force. I suppose since it only needs to be calculated once, there's little payoff to designing an elegant or perfected version. I'd imagined more statistical approaches, such as Gini Impurity (AKA information gain) might come into the mix but, alas, no.

Tokenisation breaks words in the corpus down into their fundamental components, based on how frequently each chunk appears in the text. It allows for learning more granular patterns, but maintains much of the goodness of next-word prediction, which leads to more coherent sentences.

I generated slightly more fake Harry Potter this time:

> seemed to have the only person who spoke to them a good lesson. “So — what is it like?” said Hermione.
> “Hagrid, I don’t believe that Dumbledore was going to be the only one — ”
> “I wouldn’t have to say a word with Dumbledore,” said Hermione, frowning.
> “You’d better get up to the kitchen at the moment as he left it and I have been thinking of all those that had been the one who had been able to take it in the Room of Requirement and I think I should have a point, but I’ll see you and then if it gets to work out of here, you want to keep them on the way for this year — ”
> “I want to write up with him,” said Ron.
> “Well, it will be enough to get past the exam on — ”
> “No, no, not a good idea that I got your job in detention if I can have more trouble in there!”
> “Hermione, you’re not in the hospital wing,” she added, turning to the empty compartment.
> “We need to be able to get out of the castle if we can be able to tell us how to get through that owl — ”
> “You are in this office,” said Harry.
> “But I need to know this year at the moment, I do.”
> “Yeah, well, you know how the Dark Mark come out, all right? You just know what you say at the Ministry of Magic to do it for me, not the best in a very well — ” “You must take the Invisibility Cloak,” said Ron. “He could be too careful to do it.” Harry glared at her. He did not know what Ron had-

Wow, ok. So overall, it's *not bad*! It's certainly got the same urgent tone as the previous sample, with lots of unfinished sentences and interruptions. It seems to have very solidly got the jist of certain Harry Potter scenes, but there's a slightly jarrying lack of finesse in the language used – as if someone was writing fan fiction in a second language that they're still learning. 

## Some training oddities

Just for fun, I'd like to spend a moment and cover some of the fun and unexpected effects you see during a training run.

At first, you get the usual untrained noise. This is for character-level prediction, so it's entirely random character soup:

![Screenshot 2026-06-17 at 15.47.52](/Users/olivergreen/Desktop/Screenshot 2026-06-17 at 15.47.52.png)

After a few iterations, it's begun to understand that the space character appears between words, and it's even learned a few *very basic* words – often the most common (ie safes) words. So you just wind up with repetitive preposition soup: 

![Screenshot 2026-06-17 at 15.47.35](/Users/olivergreen/Desktop/Screenshot 2026-06-17 at 15.47.35.png)

This is apparently very normal when training LLMs, it's just finding very strong signals between all of the preposition words initially. 

### Harry Potter and the persistent header

This was a fun facepalm moment during training. I hadn't manally checked the corpus, so hadn't caught that a header containing the book title appeared on every page. This caused the book titles to constantly appear in my randomly generated samples: 

![Screenshot 2026-06-17 at 16.34.52](/Users/olivergreen/Desktop/Screenshot 2026-06-17 at 16.34.52.png) 

In all fairness, the transformer is doing what I asked it to do! My fault for not checking the data. [GIGO](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out), as they say!

In the end, I used Claude to clean out the corpus and strip out any repetitive artifacts like this. If you'd like to see the outcome of the training, you can find the samples on my Github here: 

* Character-level training [samples](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/Transformers/outputs_hp_char/samples.txt)
* Word-level training [samples](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/Transformers/outputs_hp_word/samples.txt)
* Token-level training [samples](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/Transformers/outputs_hp/samples.txt)
