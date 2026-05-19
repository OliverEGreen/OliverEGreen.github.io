---
title: "LLMs Part 2: Building a Vanilla RNN"
date: 19-05-2026
---

*This post is part 2 in a series where I attempted to teach myself the very basics of language modelling. For Part 1 see [Word2Vec](https://olliegreen.info/writing/building-word2vec/)*.

## Introduction

My goal for this series is twofold. Firstly, I want to better-understand the world of language models, since they've become such a large part of our day-to-day lives. Secondly, if anyone should ever read this, to make the theory and practice behind them a tiny bit less difficult and opaque.

Because of this, I'm intentionally writing everything using standard Python primitives and using exhaustively-explicit naming wherever possible. "Pythonic" this isn't. External ML libraries are not allowed – not even something reasonable like pandas or numpy, which of course would feature heavily in more. *(By the way, if you're looking for a beautiful, minimal implementation of an RNN, look no further than Andrej Karpathy's blog post)*.

## Why an RNN?

In today's installment, I'm building a Recurrent Neural Network. This is an NLP approach from the 1990's, using neural networks to predict what word or character comes next after a given string of text. 

This is different to Word2Vec, which is really a naive embedding model; it's not capable of taking data sequence into account, nor generating new text from scratch. It also can't appreciate synonyms, each word is embedded only once regardless if it can have multiple meanings depending on its context. 

This changed with our RNN. Today, we're building the Elman RNN (1990), the canonical first-generation recurrent network. Its defining feature is the "hidden state" that feeds into itself at each timestep.

Contextual appreciation is very important for language models; later iterations of LLMs (such as ELMo) started becoming 'bi-directional'. Later innovations such as transformers brought the idea of self-attention, meaning a word's (or token's) meaning is understood within the full context of where it sits – both before and after. This context means the most likely next word (or token) can be predicted within this context, making predictions far more accurate.

Building our vanilla RNN is more complicated than Word2Vec. It's where to begin to see a much wider range of 'standard' neural network methodologies and tooling entering the fray. We saw a few techniques come up in Word2Vec, but overall it was a pretty straightforward build.

## The Approach

As before, we're using Shakespeare's full works as our training data, but this time we won't be pre-cleaning it. This is so our model can begin to learn richer relationships about things like line breaks, quotation marks and so on.

Another difference is that we'll be predicting next character rather than words. Per-word is doable but would mean a much, much larger dataset to train on (thousands of unique words rather than dozens of unique characters). While per-word might feasibly give us more sensible text, our character-level RNN will only give us Shakespeare-flavoured nonsense. That's fine for now.

## Setting things up

1. Bring in the corpus/training data, create a list of unique characters.
2. Set up the two-way dictionaries which map words to a unique ID, and vice versa. This is standard in language modelling because using primitives will be much faster to access/read than custom objects.
3. At its core, the RNN requires 3 matrices and 2 vectors. The matrices bridge from input weights, to hidden weights to output weights. The middle matrix here is what handles the 'memory' of the RNN. Additional to this are two vectors for bias – from which the RNN's predicted probabilities for the next character are calculated. All 3 matrices are initialised with small random noise values to begin with, while the vectors are all zeros.
4. In order to make this RNN train properly, I had to implement the same Adagrad approach as Karpathy used in his blog posts. This allowed my training loss to steadily decrease over time, whereas it bounced around a lot using standard gradient descent. The Adagrad approach essentially involves recreating our five matrices from above but for memory purposes, which is what allows us to be a bit more picky about the learning rate for each parameter, and which prevents this bouncing around.

With these matrices in place, it is possible to start writing the forward pass and backpropagation algorithms. 

## Method

When training, what happens in order is: 

1. We grab a chunk of the text (25 characters) and grab the IDs for each character we see. 
2. We run the forward sequence, which generates our hidden states and probabilities / predictions for what the next character will be.
3. We calculate the cross_entropy loss, i.e. how right or wrong these predictions were.
4. Using this, we run backprop, which calculates the deltas (i.e. how much each of our matrices and vectors was wrong by). Because each of these is a multi-dimensional beast, they're a "gradient" in parameter space. This is known as BPTT (backpropagation through time) and the maths can get a little gnarly at this point.
5. We then need to clamp our gradients. Because we're doing a lot of multiplication steps, there is a tendency for numbers to spiral towards infinity or negative infinity known as 'gradient explosion'. Either of these outcomes would be less than ideal, so we artificially check to ensure all gradients are limited to values between 5 and -5. This feels like a bit of a 'folk technique' to me.
6. Our final key step is updating the original 3 matrices and 2 vectors – and their corresponding memory stores. This is where we carry out the actual gradient descent.
7. The final step is cleanup - moving the position pointer along our sequence to take in fresh data and updating the hidden state to be our final hidden state (i.e. our memory) for the next loop.  

## Training

I wound up training the RNN for 50,000 iterations, which took about 2 hours on a Macbook Air M1. Had I used numpy, this amount of training could have taken around 1 minute. But that was never the point!

As the RNN began, it started off with generating pure random noise:

>  L]2 QF1,ên$N'‘%zmuÀ2Æ/GAîëy&”HànÉwæ7,LeD%wUÉ+ç31To—n#vFaxgkp/NcU2LèçHuY,o&[l“Sb%éâ”019-/J9ZhZçæE
> ’ga8i?﻿NRæî‘j7.]﻿,!pçgi+œàsJjÆc-LF+É7[êd•:a‘Eæ3wG
> z+;MmHÆnæC'è!PÀàëd’i
> 5IcR_aéwK5]v3?d$	:é5m•U:0q…oF﻿xn

But, over time, it began to generate more Shakespeare-flavoured text:

> carontinly hid it Pablenher angore rill, you est to mandt, ?_]
>
> CESTCE
> E LExit as olqueeus ich I hom, me my—
> Heple.
>
> KES.
> Nhe mur this ane thou sall, filg.
>
> FIIRS.
> Vast I of wigrt the murdss, he, in b

This demonstrates that the RNN was able to learn the structure of basic sentences, questions and characters speaking to each other in a play (see the names in all caps).

![Loss More Smoothed](/RNN/outputs/Loss More Smoothed.jpg)

Over time, our level of loss (which started around 120) began nudging down to around 45. This is considered about as good as such a vanilla NN can get. 

For contrast, I asked Claude to clone my code and re-write it in the most performant numpy equivalent possible. This meant I could scale it up to use a much larger window size and faster learning rate. However, these iterations plateaued at a loss of around 60, so surprisingly their outputs were worse than my vanilla RNN!

## Next Steps

The clear next evolution in language modelling is LSTMs. These are another step up in terms of complexity, and this should take us right up to the SoTA immediately before transformer architecture.

## Conclusion

Another metaphor for neural networks dawned on me while reflecting on this RNN – they're a sort of smart-dumb approach.

I imagined someone headbutting a piece of sheet metal thousands of times. Notwithstanding damage to their head (or the metal), with enough persistence, we'd eventually be able to make a reasonable casting of their face from the indent.

This feels a bit to me like what is happening as we train a neural network. It's not necessarily the most efficient of all ideas, but it sure does work through sheer amounts of brute force.

I'm looking forward to implementing LSTM next, which should noticeably improve the quality of our fake-Shakespeare (our Fakespeare?)