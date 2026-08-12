---
title: "LLMs Part 3: Building an LSTM"
date: 2026-05-20
post_id: 8
permalink: /posts/8/llms-part-3-building-an-lstm/
redirect_from: /writing/LLMs-Part-3-Building-an-LSTM/
kind: technical
temperature: 0.1
tagline: "This post is part 3 in a series where I attempted to teach myself the very basics of language modelling."
---

*This post is part 3 in a series where I attempted to teach myself the very basics of language modelling. See Part 1 and Part 2.*

## Implementing LSTM

In my last post, I built an RNN that was able to begin generating something you might charitably describe as Shakespeare-esque. Today I'm looking to take that slightly further by building out a more mature version of an RNN, called an LSTM.

LSTM ([Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)) is an approach from 1997, first published by the academics Hochreiter and Schmidhuber. Despite this paper's age, this approach became popular much later around 2014, when Sutskever et al (Google) [demonstrated using LSTMs for machine translation](https://proceedings.neurips.cc/paper_files/paper/2014/file/5a18e133cbf9f257297f410bb7eca942-Paper.pdf#:~:text=In%20this%20paper%2C%20we%20present,assumptions%20on%20the%20sequence%20structure.&text=Sutskever%2C%20and%20G.%20E.%20Hinton.%20ImageNet,classification%20with%20deep%20convolutional%20neural) (English <> French) with decent levels of success.

LSTMs were the last big popular approach in the field of NLP before transformer architecture took over the space—and pretty much every other space in ML at the same time!

## Method

Architecturally, this approach feels much like a progression of the RNN, which makes sense. Its implementation just adds a lot more repetition and complexity to the core ideas we saw before. 

Instead of maintaining one suite of matrices and vectors, we split each into four "gates" (two matrices + a bias vector), which are designed to get a little more fussy about what information gets kept and carried through across different timesteps:

1. Forget Gate: This decides what memory gets forgotten between timesteps.
2. Input Gate: Together with forget, this controls the cell state's update rule.
3. Candidate Gate: Where proposed new content is kept during each run. Built a little differently to the others (e.g. uses the tanh instead of the sigmoid function). 
4. Output Gate: Used to decide how much of the newly-updated cell state gets exposed to the next timestep's gates and this timestep's output projection.

Instead of carrying long memory sequence through, the forget gate weighs up whether to keep or discard chunks of memory. This allows for lower loss rates before reaching the usual plateau.

From a therapeutic standpoint, I'd just like to underline how insanely tedious this was build this all from scratch. This is entirely self-inflicted, and certainly wasn't helped by my stubborn insistence on using raw Python and explicit variable naming for everything. 

## Results

The LSTM has a much faster training process, and the most clearly Shakespeare-flavoured outputs so far. It's halfway between the [Jabberwocky](https://www.poetryfoundation.org/poems/42916/jabberwocky), Lewis Carroll's famous nonsense poem, and real Shakespeare.

For reference, the original RNN produced: 

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

And our LSTM came up with: 

> DOLANERAB. Mystens._]  
>
> CoPD.  
> Goon abqueen, never biently sund and on thuilg len sie  
>
> Miegs gelt mine be quackion! She tould I’str.  
> I yourlled doyer on nave you prame  
> Neake yoursans. She cappet for upon depoet, and hearts.  
>
> CLEOPATRA.  
> Hath arling silgue, meself.  
> There of you you?  
> AFfonces. Will dage a goster on ararch,  
>
> CLEOPATRA.  
> Loundwerlad strenguted.  
>
> DLECUALL.  
> Dead!  
>
> CLEOPATLAG.  
> Do me c  

This is clearly beginning to get... somewhere! It's consistently recognising Cleopatra (even if getting her name wrong sometimes). The LSTM excerpts read far more like chunks of a real stage play and contain more real words that you'd expect to find in Shakespeare. 

We reached a natural plateau very quickly, within around 15K iterations.

![LSTM loss curve](/LLM-Learning/LSTM/images/loss-curve.jpg)

### Why Karpathy's results are so good

Results are still not comparable to the high-quality results as Karpathy achieved in his [famous 2015 blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/): 

> PANDARUS:  
> Alas, I think he shall be come approached and the day  
> When little srain would be attain'd into being never fed,  
> And who is but a chain and subjects of his death,  
> I should not sleep.  
>
> Second Senator:  
> They are away this miseries, produced upon my soul,  
> Breaking and strongly should be buried, when I perish  
> The earth and thoughts of many states.  
>
> DUKE VINCENTIO:  
> Well, your wit is in the care of side and that.  
>
> Second Lord:  
> They would be ruled after this chamber, and  
> my fair nues begun out of the fact, to be conveyed,  
> Whose noble souls I'll have the heart of the wars.  
>
> Clown:  
> Come, sir, I will make did behold your worship.  
>
> VIOLA:  
> I'll drink it.  

I mean, he's kinda the GOAT. However, he did have a few aces up his sleeve to achieve this.

1. Karpathy's original blog post used a two-layer LSTM, whereas ours is one-layer.
2. He's also using a better approach than our Adagrad, which can wind up blocking itself after a while.
3. He's training for millions of iterations for days to get those results. We're about halfway there and it's taken us much less time. Perhaps an hour of training overall. 

## Next steps

I'd like to try building transformer architecture next. It involves loads of matrix multiplications (matmul as the kids call it), but architecturally it's an entirely different setup. Performance-wise, it's meant to be a total game-changer, so I'd like to see this for myself. 

After this, my next steps will be to see how to build new systems on top of an LLM. Language models are powerful, but still pretty dumb just by themselves. Ask it what year World War 3 started and it will confidently give you an answer. After all, it's just a word/token prediction machine. 

What has been built on top of LLMs since is much more interesting, as it starts to channel their raw power into some productive directions: 

* Reasoning models - I *believe* this uses an LLM to develop testable hypotheses (chain of thought) which get assessed using reinforcement learning.
* RAGs - how to embed context and inject it into a prompt at run-time to give fewer hallucinations, and better referencing of relevant data chunks.
* Multi-modal modelling, such as [CLIP](https://openai.com/index/clip/) from OpenAI, which mapped images and caption text to a shared vector / parameter space.
* Agentic Harness - A series of tools around an LLM to help turn it into an agentic helper. Things like toolsets, guidelines, safety barriers. I've seen very promising results from [Cloudflare's write-up](https://blog.cloudflare.com/cyber-frontier-models/) of tuning their harness using Mythos.

Until next time, thank you for reading!