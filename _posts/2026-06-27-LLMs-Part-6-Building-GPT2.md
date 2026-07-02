---
title: "LLMs Part 6: Building GPT-2"
date: 2026-06-27
---

*This is Part 6 in my series on building language models. In this post, I refactor our hand-written GPT-1 model into GPT-2, exploring a wide range of optimisations necessary to train a model of this scale on rented GPU hardware.*

*For more, see:*

* *[Part 1](https://olliegreen.info/writing/LLMs-Part-1-Building-word2vec/)* - Building Word2Vec
* *[Part 2](https://olliegreen.info/writing/LLMs-Part-2-Building-a-Vanilla-RNN/)* - Building a vanilla RNN
* *[Part 3](https://olliegreen.info/writing/LLMs-Part-3-Building-an-LSTM/)* - Building an LSTM
* *[Part 4](https://olliegreen.info/writing/LLMs-Part-4-Transformer-Architecture/)* - Building transformer architecture
* *[Part 5](https://olliegreen.info/writing/LLMs-Part-5-Tokenisation/)* - Tokenisation

## Introduction

After building a working transformer model, I was left wondering about where to take this project next. I'd been semi-chronologically building out some of the most influential language modelling approaches but, despite its release almost a decade ago, transformer architecture still absolutely dominates the field. What we've seen since is an explosion of [variants](https://sebastianraschka.com/llm-architecture-gallery/) of this architecture. Refining those variants is still very much an active area of research today: 

* The famous [2017 Google Brain](https://arxiv.org/abs/1706.03762) paper used an encoder-decoder architecture, since it was trying to undertake a translation task (very loosely, mapping input tokens to output tokens).
* For most modern LLMs, this setup was unnecessary. It's since turned out that decoder-only is ideal for text generation; OpenAI saw the Google paper and its potential for their goals in 2018's GPT-1.
* Conversely, it transpired that encoder-only is an excellent approach to take for rich text comprehension tasks (first BERT, and later BART). 
* A more recent example of an architectural leap forward is DeepSeek R1's impressive [attention head](https://www.youtube.com/watch?v=0VLAoVGf_74) design, released in January 2025. Some pretty pictures of this are available from Welch Labs [here](https://www.welchlabs.com/store/mladeepseek-attention-poster-13x19). 

But it's not architectural design that is behind the astronomical leaps and bounds we've seen since GPT-1:

* The most obvious difference between GPT-1 and today's models is the <u>astronomical</u> difference in scale. While GPT-1 can be trained locally on a laptop, training anything like GPT-3 is still a multi-million pound undertaking. And it is in this scale – when language models became *large* language models – that most of the really interesting emergent behaviours started appearing (this is worth a blog post on its own).
* The other half of the equation is something we've not covered yet – fine tuning and post-training. Everything we've covered to date can be labelled as 'pre-training'. While early language models were prone to hallucinations and had tiny context windows, modern model training approaches were vastly able to counteract these and hard-code more desirable behaviours (like a prompt/response chat architecture). I'll also write a blog post on this, as it's a separate but fascinating area of domain knowledge. 

## Why build GPT-2?

For the language modelling world, OpenAI's November 2019 announcement of GPT-2 was a significant milestone. OpenAI initially [called it "too dangerous" to release](https://openai.com/index/gpt-2-1-5b-release/), and put a six-month moratorium in place before its release so the tech community could have time to discuss and adapt to its impacts. Despite this hype, it failed to make any significant waves outside of its niche community and the societal risks never really materialised.

Architecturally, it's not that different to GPT-1 and transformer architecture in general. So why bother building it?

[^1]: The biggest changes were a pre-forward pass layer normalisation step, switching ReLu for GELU (I skipped this for GPT-1) and implementing AdamW optimisation.

What's fascinating to me is how much the applied *technique* needs to change in order to start working at the GPT-2 scale. Almost everything I've had to change was in response to this single stimulant, and the level of efficiency-hacking and optimisation is the really striking difference. 

### PyTorch - or how I learned to stop worrying and embrace the GPU

First of all, it would no longer possible to train at this scale using my maximally-masochistic raw Python approach. Either my laptop would melt or we'd encounter the [heat death of the universe](https://en.wikipedia.org/wiki/Heat_death_of_the_universe). 

The first step is to chuck out our raw Python code. To be fair, it's done the job; I wrote a fully functional transformer and learned the architecture fairly comfortably. Thank you and farewell, Python.

But we can't just refactor to numpy, either. This move alone granted us a roughly 4000x speed-up during training. But it's still not enough. The only feasible approach remaining is to roll up our sleeves and embrace PyTorch. That's because numpy is CPU-only, and we're now living in the GPU era. This makes all our matrix math parallelisable, unlocking another 1-2 orders of magnitude in terms of speed up.

Training the equivalent model on my laptop locally using raw Python would take, Claude estimates, around 6 million years (that's *around* the length of time ago that humanity split off from chimpanzees). Tempting, but I'll pass.

It also happens that PyTorch syntax is also *really* condensed. It's very satisfying to see tooling that's so beautifully coupled to the grunt work of ML. It took a couple of hours to rebuild our 700 lines of raw Python into [barely 100 lines of PyTorch](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/GPT-2/gpt-2.py).

[^2]: This crept back up a bit after adding in all the extra optimisation code we needed.

The whole 400-line backwards pass is captured in a single line now. At the time, I said *"it feels like the jetpack I'm wearing is also wearing a jetpack."* 

### Getting closer to the metal

We're moving deep into optimisation territory now, so grab your oxygen mask. Just like when we put on our C/C++/Rust hats, we have to start thinking at the computer science level and paying close attention to things like memory access and writing operations.

Many of these optimisations aren't paper-faithful to GPT-2 by design – we have better, faster tooling now and using this is what makes training a cheap and performant model possible for a hobbyist.

* A lot of the required optimisations are device-sensitive. I've been building the models on my Mac M1, which uses Apple's propietary Apple Silicone hardware. But I would eventually need to train this model on NVIDIA GPUs, which is known as the 'CUDA' platform. 

* To mitigate the Mac / NVIDIA architecture gap, I added an environment variable "SMOKE" so I could run local smoke tests on my Mac using much smaller hyperparameters.
* I wasn't gonna just train on a single NVIDIA GPU either, but wanted to spread the work across multiple rented GPUs simultaneously. This meant we needed to adjust our model to work as such. 
* In previous models, the learning rate was a fixed constant. But we'd be able to improve our end results if this dynamically varied over time.
* Switching off gradient-tracking for sample generation using the @torch.nograd() decorator. We have no interest in this kind of book-keeping when generating, only when training.
* To avoid memorisation, I introduced a training/testing split of 90:10 on the data, meaning we'd hold back some of the text to ensure our model was generalising and not just learning to regurgitate accurately. With a large enough corpus, we'd also help reduce the chance of this occurring.
* Flash memory. 

After all of these fun and games, the final GPT-2 Python script clocks in at around 300 lines and can be found on my Github [here](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/GPT-2/gpt-2.py). 

### Choosing a large corpus

My paltry Harry-Potter and Shakespeare corpuses simply weren't providing enough data for the new beast I was attempting to summon. I was going to need to hunt for a much larger corpus. 

The original GPT-2 was trained on the WebText, a corpus which OpenAI specifically developed for this model. They used the *slightly bold* technique of scraping Reddit post external links via the API (back when you could do that), using upvotes as a quality indicator. It's a clever growth hack that's slightly reminiscent of the original Google Search [PageRank](https://en.wikipedia.org/wiki/PageRank) approach.

To quote Josef Stalin:

> One data-theft is a tragedy. A million data-thefts is a statistic.

In retrospect, he was way ahead of his time.

### Ethically-harvested data

We no longer live in the age of excess. *Moving fast and breaking things* is not the way anymore. And yet, I needed a dataset worthy enough to summon the GPT-2 beast. 

Thankfully, HuggingFace does us all a great service here!

[^3]: Fun fact: HuggingFace is named after the official name for this emoji - 🤗

They provide multiple curated datasets – I went with the FineWeb Edu set, which is largely trained on encyclopaedia and quality news articles. As corpus flavours go, it's quite bland. It reads a lot like the unflavoured English language.

There's a well-known rule of thumb for choosing an appropriately-sized corpus known as the 'Chinchilla Rule', and this corpus was perfectly chinchilla-sized. 

### Byte-level Byte Pair Encoding... What

When it comes to tokenising this corpus, GPT-2 also uses what's called byte-level Byte-Pair Encoding. This is a really weird name that requires a little unpacking:

* Byte-Pair encoding was originally a data compression technique from the world of computer and data science. It was really operating at the level of bytes and bits, hence the name. 
* The language modelling world was inspired by this technique when developing the BPE tokenisation approach. And they kept the name, even though it was more of an analogy, as the tokens are very much human-readable text chunks, like 'anti-' or '-tion'. Not bytes.
* OpenAI then decided to start parsing all their tokens at the level of the UTF-8 byte stream. This is a smart move. It means they can never encounter an unexpected token, as UTF-8 has [plenty of room](https://www.youtube.com/watch?v=MijmeoH9LT4) in its scope to capture all and every possible choice of symbol – past, present and future. 
* We can used this exact tokeniser today, it's called *tiktoken*. In retrospect, this was probably a poor choice of name. I assume that it was just a funny-at-the-time play on words... but then TikTok did really kind of take off. The timeline adds up. If so, whoops.

If you're curious, the script I wrote to use tiktoken's tokeniser lives [here](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/GPT-2/prepare_data.py). It's so small you can practically inhale it.

And that's it! With a corpus and our stack of optimisations in place, I was ready to begin road-testing this thing. 

## Renting GPUs

The idea of renting hardware in the cloud was very new to me and pretty exciting. Yes, of course, I've worked at SaaS startups. In a sense, this is very mundane. But I'd never clicked the buttons or fired off the commands to invoke the dark, remote hardware myself. It was time to head over to [runpod.io](https://runpod.io/).

The sign-up process was laudably smooth. I'd recommend this service to anyone as it's delightfully no-nonsense. It was now time to **make some moves**.

![RunPod deploy screen — selecting the NVIDIA H100](/assets/images/runpod-deploy-config.png)

I selected the NVIDIA H100 GPU, an industry-standard workhorse and set the GPU count to 4x. Everything updated instantly to show me the hourly cost. 

![RunPod showing the hourly cost for the 4× H100 pod](/assets/images/runpod-cost-breakdown.png)

A few clicks later and I'd spun up the pod. I had access to a Terminal and a Jupyter notebook. 

I noticed Some chips weren't available and you have to subscribe to gain access to them. I found this fascinating, but it makes sense. Even Anthropic are [renting hardware from SpaceX](https://x.ai/news/anthropic-compute-partnership) right now. Supply is tight and chip costs are solidly in five-figure territory. Inshallah, somehow they're still fine to rent for a short period.

Using tiny hyperparameter values, I ran a few smoke tests for both the tokeniser (Step 1) and the training pass (Step 2). I encountered a hiccup where tiktoken's own special <|endoftext|> token appeared as raw text in our corpus data. I forcibly set the encoding style to encode_ordinary() to work around this.    

Beyond this, everything passed, and so it was time to press the buttons for real. 

* Tokenisation ran for around 20 minutes, creating 2.5B tokens, which we then ingested for the following step. This cost me maybe £2.
* Training ran for around an hour across the 4 GPUs, costing around £11.

In total, this episode cost me £13 – equivalent to splashing out *a little* at Pret a Manger. To give a sense of the next scale jump, training an equivalent GPT-3 would still cost in the low millions.

The result of my efforts was a weights file of around 1.5GB. Two thirds of this size comes from using AdamW optimisation!

I was careful to grab the weights and properly terminate the instance before deleting everything forever - it's easy to keep draining your account credits with persistent storage and stopping (rather than terminating) the pod instance.  

## Results

As raw language models go, that was basically it! But I had a few steps left in order to make this something we could play with. 

### Building the sample generator

In order to make a next-token predictor, you need a function to call that actually forces this to happen. You can find the [generate() function here](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/GPT-2/gpt-2.py). 

This is really just a modified version of the sample function we've been using in the previous language models, with a few controllable knobs added in for things like temperature.

### Controlling temperature

Speaking of, temperature is a fun lever to pull to make things more predictable or more fun (depending which way you choose). Set the temperature too low and your sample will just speak in predictable loops. Set it too high and you'll have way more fun, as the samples approach the ramblings of a madman. 

With a temperature of 0.7 and a sample text of: "The Roman Empire was", our script generated:

> The Roman Empire was the period of the Roman Empire that lasted for a few years, which took place in 1238 and 1240. Roman Emperor Hiberius I, who ruled the region from 1240 to 1250, was a strong supporter of the Empire. The Roman Empire was the result of the Roman Empire’s expansion into the Byzantine Empire, which made it a major power in Europe.
> In the early 13th century, the Empire was under the influence of the Roman Empire. In the year 1251, the Empire rose to power, which took place in 1342, and its influence spread throughout Europe. The Empire was created by the Roman Empire in the 15th century and was the subject of the Roman Empire’s conquest.
> The Empire was further divided into two sectors, the first being the expansion of the empire and the second being the expansion of the empire. The expansion of the empire began with the growth of the empire’s power, which gave rise to the Roman Empire

We did it! AGI has been achieved for as little as £13. Someone should tell those nerds in Silicon Valley to stop grinding their [996](https://en.wikipedia.org/wiki/996_working_hour_system)s. It's [GGs](https://en.wikipedia.org/wiki/Glossary_of_video_game_terms#GG).

A few things to note about our sample: 

* Grammatically, it's very fluent. Our model has correctly built out paragraphs and sentences are all finishing beautifully.
* Yet (don't tell anyone) but it's also full of hallucinated details and wildly self-contradicting. I especially love the invented emperor Hiberius I. It's confidently wrong but still very much within the right ballpark!

This is, in a sense, the purest essence of a raw language model, a confident bullshitter. In Part 3, I mentioned that a raw language model would happily tell you when World War 3 began. Now, if you'll kindly allow me to demonstrate:

> World War 3 began in the year 1950. The war continued to unfold until the 1970s. The U.S. military was at war with the United States when it was at war with the United States. The U.S. military began developing into a coalition of military forces, but it soon began to develop into a coalition of military forces in the 1950s. The first US to support the war was the U.S. Army. The war continued until the 1960s.

They call that narrative payoff.

### In-context learning

One interesting aspect of GPT-2 is that it began offering us some very early signs of emergent behaviours – unexpected abilities naturally rising out from language models that had never been formally considered or trained for. And they keep happening and surprising us. 

For example, our model is capable of learning a basic question-and-answer like syntax from a single example given in the prompt (called 'one-shot learning'). 

Given the sample text:

> **Q: What is the capital of England? A: London**
> **Q: What is the capital of France? A:**

Our model auto-completes with:

> Q: What is the capital of England? A: London
>
> Q: What is the capital of France? A: Britain
>
> Q: What is the capital of France? A: The capital of Britain
>
> Q: What is the capital of Italy? A: The capital of Italy
>
> Q: What is the capital of France? A: The capital of France has many different types of capital
>
> Q: What are the features you will find in England? A: England
>
> Q: How much will be done? A: England
>
> Q: Where does the capital of England rise? A: England

The model has begun to organically pick up on the Question and Answer format and has even started generating its own question-answer pairs. Is it remotely useful? Not just yet. But it's the *early signs of LLM superpowers* that only begins to grow with upscaled models like GPT-3.

Post-training is where we really start to move from a next-token predictor into something meaningfully useful, such as a chatbot.

## Scale and timeline

We've hit a bit of a hard limit now, in terms of potential scale, cost and hardware. The quality increases would partly come from scaling up. We'll have to slightly switch our approach to carry on learning. 

### Cost comparison

* GPT-1 - done locally, costs nothing and takes a few hours to plateau.
* GPT-2 - trained on rented GPUs, costs around £13, about 90 minutes. For reference, this was the smallest of the four GPT-2 models, GPT-2-124M. 
* GPT-3 - would still cost millions to train, even today.

If anyone would like to donate a several million pounds to read my GPT-3 post please feel free to reach out.

## Next steps

So, what next?

Luckily, others have trained open models for us! We have a wide range of open source language models from the GPT-3 era and onwards at our disposal. That's kind of incredible, really.

I'll be using these raw models for the next steps:

- Fine tuning to make my model adhere to a user prompt/response conversational approach
- Post-training to better-understand how different reinforcement learning techniques can be used to improve performance, such as RL from human feedback (RLHF) or RL with verifiable rewards (RLVR).

With these in place, I *might* be able to build a slightly crappy chatbot interface. After that? Some ideas: 

* Chain of thought, reasoning models. 
* Agentic tool building / looping
* RAGs
* Harness engineering

See you in the next episode, thanks for reading!