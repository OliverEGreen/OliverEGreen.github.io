---
title: "LLMs Part 6: Building GPT-2"
date: 27-06-2026
---

*This is Part 6 in my series on building language models. For more, see [Part 1](https://olliegreen.info/writing/LLMs-Part-1-Building-word2vec/), [Part 2](https://olliegreen.info/writing/LLMs-Part-2-Building-a-Vanilla-RNN/), [Part 3](https://olliegreen.info/writing/LLMs-Part-3-Building-an-LSTM/) and [Part 4](https://olliegreen.info/writing/LLMs-Part-4-Transformer-Architecture/). In this post, I refactor our hand-written GPT-1 model into GPT-2, exploring a wide range of optimisations necessary to train a model of this scale on rented GPU hardware.*

## Introduction

At the end of the last post, I was wondering what to do with raw transformers next. 

GPT-2 was an important release. Initially called "too dangerous". Architecturally very similar to GPT-1 and transformer architecture in general. But lots to learn to get there. 

Scale

Raw transformers? Out. This is only feasible now in PyTorch, not even numpy is fast enough. 

Rebuilt everything, hundreds of lines of code massively compressed, barely 100 lines now. The whole 400-line backwards pass is a single line now. "feels like the jetpack I'm wearing is also wearing a jetpack."

Optimisations. Omg. Properly hardware-sensitive.

### Choosing a large corpus

Tokenisation process totally different.

GPT-2 uses byte-level Byte-Pair Encoding. This is a really weird name that we should probably unpack.

Had to find a quality benchmark of sample data. Thankfully HuggingFace does us all a great service here.

### Architectural choices

Learning to optimise for architecture and splitting across multiple cores. 

## Renting GPUs

This was all very new to me and pretty exciting. I've heard so much about H100s and NVIDIA architecture, but never been remotely close to accessing it. 

Some chips weren't available and you have to subscribe to gain access to them. I found this fascinating, but it makes sense. Even Anthropic are renting hardware off of SpaceX right now. Supply is tight and chip costs as astronomically high (but fine to rent for a short period!).

Training a few smoke tests. A little fiddling in the terminal later.

Then generating my own weights for around £20 in an hour using 4xH100s. 

The resulting weights file is 1.5GB. Two thirds of this size comes from using AdamW! Being sure to grab the weights and properly terminate the instance before deleting everything forever - and making sure no ongoing running costs. 

## Results

Adding sampling ability to our script, just adapting the 'Generate' function and adding in some controllable knobs for temperature.

Making a quick and easy way to sample our model. Temperature is a fun lever to pull to make things more predictable or more fun depending which way you choose. Go too low and it'll just confidently repeat itself. Go too high and you'll get insanity. 

With a temperature of 0.7 and a sample text of: "The Roman Empire was", our script generated:

> The Roman Empire was the period of the Roman Empire that lasted for a few years, which took place in 1238 and 1240. Roman Emperor Hiberius I, who ruled the region from 1240 to 1250, was a strong supporter of the Empire. The Roman Empire was the result of the Roman Empire’s expansion into the Byzantine Empire, which made it a major power in Europe.
> In the early 13th century, the Empire was under the influence of the Roman Empire. In the year 1251, the Empire rose to power, which took place in 1342, and its influence spread throughout Europe. The Empire was created by the Roman Empire in the 15th century and was the subject of the Roman Empire’s conquest.
> The Empire was further divided into two sectors, the first being the expansion of the empire and the second being the expansion of the empire. The expansion of the empire began with the growth of the empire’s power, which gave rise to the Roman Empire

Grammatically very fluent. Paragraphs, sentences all finishing beautifully. Yet also full of hallucinated details and wildly self-contradicting. This is, in a sense, the purest essence of a raw language model. 

In Part X, I mentioned that a raw language model would happily tell you when World War 3 began. I am now able to demonstrate this:

> World War 3 began in the year 1950. The war continued to unfold until the 1970s. The U.S. military was at war with the United States when it was at war with the United States. The U.S. military began developing into a coalition of military forces, but it soon began to develop into a coalition of military forces in the 1950s. The first US to support the war was the U.S. Army. The war continued until the 1960s.

## Conclusion and next steps

We've hit a bit of a hard limit now, in terms of potential scale, cost and hardware. The quality increases would partly come from scaling up. We'll have to slightly switch our approach to carry on learning. 

### Cost comparison

* GPT-1 - done locally, cost nothing, a few hours to plateau.
* GPT-2 - trained on rented GPUs, cost around £20, about 90 minutes. 
* GPT-3 - would still cost millions to train, even today.

Luckily, others have done this for me.

Pre-training and now moving towards post-training.

### Timeline

In terms of timeline, roughly analogous:

* Word2Vec - 2013? Only an embedding approach.
* RNN - 1980s?
* LSTM - 1990s?
* GPT-1 (Transformer architecture) - 2018?
* GPT-2 (NanoGPT) - February 2019