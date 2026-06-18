---
title: "LLMs Part 4: Transformer Architecture"
date: 18-06-2026
---

*This post is Part 4 in this series, where I try to teach myself the very basics of language modelling. For my earlier posts, see: [Part 1](https://olliegreen.info/writing/LLMs-Part-1-Building-word2vec/), [Part 2](https://olliegreen.info/writing/LLMs-Part-2-Building-a-Vanilla-RNN/) and [Part 3](https://olliegreen.info/writing/LLMs-Part-3-Building-an-LSTM/).*

*I have also recently discovered the Stanford online course [Language Modeling From Scratch](https://cs336.stanford.edu/), which I look forward to checking out soon. For any visual learners, I'd highly recommend the [series on neural networks by 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk).*

## Introduction

Ok, it sounds grandiose, but Google's 2017 paper '[Attention is All You Need](https://arxiv.org/abs/1706.03762)' has influenced practically every AI achievement of the past 9 years. It was a seriously big deal and the world simply wouldn't be the same without transformer architecture. In a way, it's incredible that the paper's authors aren't household names by now, like Crick and Watson.

This breakthrough was tackling the same familiar problem areas we encountered in my earlier posts, such as attention, but with a wholesale new architecture that delivered:

* As the paper's title gives away, the researchers presented a wholly-new approach to attention, called 'self-attention'. If you look back at our earlier Word2Vec implementation in Part 1 of this series, you'll see that each word has only one embedding, totally ignorant of its context. But language absolutely needs an understanding of its context to nail its next-word/token/char predictions. Our QKV matrices and feed-forward step finally nailed this mechanic.
* Much greater training speed, since this method allows for parallel processing rather than step-by-step approaches.
* Direct long-term connections, enabling much better memory retention. In our previous models, memories were being effectively diluted at every timestep (i.e. series of matmul calculations).

It could be argued that these last two new efficiencies are what unlocked *language models* to start becoming **large** *language models*. With long-term memory and parallel processing, transformer architecture scales really well. The years after became an exercise in scaling up the amount of training data we could feed into the machine.

Beyond this, the architectural approach was such a step-change, it unleashed a near-decade surge of interest and innovation in the AI space, practically wholesale replacing almost every other approach used within the field, not just language modelling - every field. The previous language models I built, the RNN and LSTM were effectively relegated to history.

Despite its global impact, I would also argue that this isn't the most complex architecture we've seen, it's simply the one that worked. It's elegant and, to some degree, slightly minimalistic as language modelling approaches go; another good example of [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

There's one more thing LLMs unlocked: human language wound up becoming incredibly helpful for unlocking multi-modal approaches to AI, since everything can be described using words. This means language data can act as the 'glue' that binds together (or, rather, translates between) text, image, video and audio models.

What I've built ([see my Github](https://github.com/OliverEGreen/OliverEGreen.github.io/tree/main/LLM-Learning/Transformers)) essentially mirrors the approach taken during development of OpenAI's [GPT-1](https://en.wikipedia.org/wiki/GPT-1). In this repo are the usual [hand-coded](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/Transformers/raw_transformers.py) versions written by me, and the [hyper-optimised version](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/Transformers/basic_transformers_numpy.py) that I asked Claude to build afterwards, for comparison.

Without further ado, let's take a look at this thing.

## How transformer architecture works

### Positional encoding

In order to predict the next token in a sequence, our language model needs to understand in what order any previous tokens occurred. By itself, the maths behind attention is order-blind – it's just a weighted sum, so "abc" and "cba" look identical to it.

We need to encode tokens' positions to store information about word order. Initially, I assumed this would just be handled by a simple index number. However, because we're adding this to our content embeddings, it can't be a simple integer index, as the numbers would get way too high and drown out our actual embeddings.

Beyond this, it turns out that the model can learn way more nuance about structures (e.g. paragraphs, sentences) if we swap the integer index approach for a vector instead. As Claude helpfully puts it: 

> "position becomes something attention can reason about with the same machinery it uses for content. The payoff is that relative distance shows up as a consistent geometric relationship the dot product can exploit, no matter the absolute positions". 

### Transformer 'blocks'

The heart of the transformer is one or more 'blocks', which was a new term for me. I picture it, very roughly, as analogous to the 'layers' we were handling in our RNN and LSTM models, which can be layered up into a deeper 'stack' for greater prediction performance. Like a data lasagna.

Each "block" has two core functions. Unhelpfully, these are sometimes called 'layers'. Not confusing at all!

1. **Self-attention layer:** This is the step where we 'gather' information from previous tokens. Our function looks at all earlier tokens (not looking ahead in the sequence, otherwise they could cheat) and pulls in the relevant context. This is where our QKV matrices come in (more on them later). This step is a lot of matrix multiplication which, by itself, would result in a very muddy, blended prediction. Hence, the next part.
2. **Feed-forward layer:** Once feedback has been gathered from all previous positions, this layer "thinks" about what it gathered without looking around and uses its own knowledge (trained weights) to form a conclusion. It gets multiplied by a matrix (initialised with noise, but it learns the structural patterns as we go along, acting as our memory).

As per usual with neural networks, we carry out these unsupervised processes many thousands of times: starting with random noise, undertaking a forward-pass, then gradient descent and backpropagation. Over and over again until the inherent patterns begin to emerge.

## Other parts

There are some other key functions firing within the architecture, but these strike me as more business-as-usual. 

1. Output projection - Calculates a score per vocabulary token then we softmax this into probabilities. We've seen this before in the other architectures, our end goal is a dictionary of tokens with a probability per next-thing (whether that's word, token or character).

2. "Add and Norm" steps

   We see these in the code as LN1 and LN2, but they're each made up of two distinct parts: 

   * The "norm" step is essentially just tidy-up, a way to stop the numbers in our matrices from running away from us. All the endless matmul makes them want to fly off to positive or negative infinity, so the norm is just counteracting a side-effect, but without it we wouldn't get very far.
   * The "add" step is very important - it gives our gradients a neat "highway" path back through our deep stack. This had been a problem for ages with storing long-term attention, as all the endless matmul in previous architectures started to wash out our information, like a [Xerox of a Xerox](https://bojackhorseman.fandom.com/wiki/Xerox_of_a_Xerox).

## Q K V matrices

I glossed over these in the self-attention section above. These matrices are at the heart of the iconic transformer setup. And, as I said earlier, it's beautifully minimal. The Q K and V matrices stand for 'Query', 'Key' and 'Value' and, structurally, all 3 are identical. <u>It's only what we do with the result</u> afterwards that really makes each of them what they are.

**Query** - What each position is "looking for". This is a vector of its bias up until now, a bit like how we'll see a strong pattern for 'u' following 'q'. In my mind, it's a bit like posting a job advert.

**Key** - Each position advertising what it has. This allows us to understand how much attention to pay to each of them. Something relevant should be treated as a strong signal, a bit like a quality CV landing that matches the job advert. It's a slightly shoddy metaphor, as we'll actually hire a bunch of people for this role in percentages (some more than others).

**Value** - This is how each position 'contributes' when it is given an amount of attention. In our metaphor, this is more akin to the work that the chosen candidate does once they've been hired. This is useful because it allows more flexibility - the position is able to 'advertise' one thing (Key) and potentially 'contribute' its value in another unforeseen way. 

### The forward pass deeper dive

At each position, the position generates the query and <u>every prior position</u> then generates its key and value matrices in response to this. We match based on queries and keys, but what flows onto our "conveyor belt" is value.

In short, for each "position" in our "belt" of tokens we're feeding into the forward pass:

* We dot-product the Query and Key outputs together to see how well they match (this is standard linear algebra stuff). We're getting a sense of the query-ness of the keys. This gets scaled-down and softmaxxed and becomes the 'score', a slice of attention. It holds the 'weights' which say how much we want each value in our vector to factor into the blend we're about to make.
* Each position also creates a 'Value' vector. This is its actual contribution, the 'work delivered', in our metaphor. We then use our 'weights' we just made using the other two matrices to blend each value in our position's vector in by a certain amount.

This doesn't happen incrementally, timestep by timestep, unlike the RNN or LSTM. This is because we've already precomputed all of the matrices we want by this stage – in our earlier definitions of queries, keys and values (using matrix_vector_multiply).

### Feed forward deeper dive

While the "attention" layer does the gathering of relevant context from prior positions and produces a weighted blend (Value), the feed-forward layer is what comes next. Mechanically, it's a shallow two-layer network, and it's where we start implementing what we've learned so far. 

* **Attention** is like asking a committee of 5 people for advice on something. At first, you don't know them at all, and some might give great advice on electronics, others in cooking or style advice, while also giving bad advice in other fields too. What you get out at the end of this process is the committee's recommendation.
* **Feed-forward** is your best attempt to listen to this recommendation and decide on your own course of action. While at the start you know nothing, over time you're able to gradually determine which parts of the recommendation can be trusted for different things, and by how much.

If you can grasp that, you've essentially grasped transformers! The big picture of this architecture is surprisingly straightforward and elegant.

## Training process

Transformer architecture is miserably slow to run on my little Macbook Air. A version using numpy and optimised to my hardware runs ~4,000x faster than my local hard-coded version.

With larger different vocab sizes (due to different encoding / tokenisation approaches) this gap becomes even more apparent:

* Next-character prediction worked with a vocab of 80 unique **characters**.
* Next-word used a vocabulary of 25,400 unique **words**.
* Next-token used a vocabulary of 4,045 unique **tokens**.

With training it's worth also noting that the model eventually just starts hard-memorising the patterns it sees. This is counterproductive; it can give the appearance of intelligence without actually delivering any. But the whole purpose of our LLM is generalisation, which this goes against.

That's why it's worth keeping an eye on two metrics: training loss and validation loss. As training progresses, these values should both trend downwards with time. When they start diverging (called the "train-val gap"), the model is beginning to memorise and training won't actually help the model any more.

## Results

The transformer works quickly, but I've still needed to run for a few hours each time to generate anything worth showing. The good news is that we've immediately generated far more realistic-looking Shakespeare.

Comparing against our previous results:

**RNN Shakespeare**

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

**LSTM Shakespeare**

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
> Do me c-

**Transformer Shakespeare**

> KING.
> But therefore I am glad
> That thou hast in all this most good man,
> And therefore thou hast my father dead,
> To call my wife and my life to me.

You can read all of the samples [here](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/Transformers/outputs_shakespeare/samples.txt) - be sure to see how they evolve over time. 

Still, this latest output strikes me as more a collection of words. A decently Shakespeare-coded word soup. It also made me remember that I don't really know enough about Shakespeare to be a useful judge of quality. So I started also generating some Harry Potter instead! It's a recognisable style and 7 books makes a somewhat-ok sized training corpus, at least for my needs. 

The last sample I reached before training plateaued read:

>  “So — what is it like?” said Hermione.
>
> “Hagrid, I don’t believe that Dumbledore was going to be the only one — ” “I wouldn’t have to say a word with Dumbledore,” said Hermione, frowning.

Feel free to check out all my generated Harry-Potteresque samples [here](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/LLM-Learning/Transformers/outputs_hp/samples.txt).  As before, the training starts off with garbled nonsense and gradually plateaus around Harry Potter-flavoured prose. It lacks any long-term consistency, but if you squint it's definitely heading in the right direction!

## Next steps

I haven't decided where next to take this project yet. It's been very enjoyable learning the ins and outs of the language models that underpin our everyday lives now. What I've built here is essentially GPT-1, the first language model published by OpenAI.

From here, I can see two paths: 

1. I keep learning more about architecture and start attempting to rebuild GPT-2, which has many architectural improvements. However, everything from here on is essentially tweaks and larger datasets; transformers still reign supreme even in 2026.
2. I can look at the ecosystem of tools which gets placed *atop* the language model, such as:
   1. Turning my GPT-1 into a ChatGPT-1.
   2. Reasoning models and chain-of-thought functionality.
   3. RAGs
   4. Agentic tool-calling and harness engineering.

Whatever I wind up doing next, the only place you'll find out is here. Until the next episode – ta-rah!



