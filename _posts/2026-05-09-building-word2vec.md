---
title: "Building Word2Vec"
date: 2026-05-09
---

*This is a personal account of building Word2Vec from scratch in Python + numpy, the bugs I found along the way and the resulting embeddings that were achieved.*

I was recently asked in an interview how much I knew about [RAGs](https://en.wikipedia.org/wiki/Retrieval-augmented_generation). 

I replied honestly; I've read about them and gone back and forth with LLMs so that I grasp the basic ideas. But this question stayed with me. I wasn't satisfied with my surface-level knowledge. Given how much LLMs have reshaped our world in the last few years, I started longing for a deeper knowledge –  the kind you can only achieve through building, testing, failing and fixing. 

Back in 2019, I completed a course in [applied machine learning](https://emeritus.org/program-information-sessions/columbia-engineering-executive-educations-applied-machine-learning/). It taught us all of the core techniques and approaches directly from the maths; it was up to us to rebuild each algorithm to prove that we'd grasped the course material. This took my knowledge right up until the ~2010s boom in deep learning. From today's perspective, what I learned now feels like lovely, interesting and quaint historical knowledge... but honestly it's been outdated and outclassed since then.

But there is good news! I have a lot of time right now and a thirst for learning. The RAG question made me want to dig deeper into the world of LLMs, so I decided to go back in time to where my knowledge cuts off and try to build something myself. 

## Word2Vec

After much research, I landed on the [2013 Google Word2Vec](https://arxiv.org/pdf/1301.3781) paper. This marked a seminal moment when the NLP community shifted to NNs, and it covered the embedding subject that is still core to RAGs today. I was familiar with previous [Markov](https://www.youtube.com/watch?v=KZeIEiBrT_w)-like / HMM approaches to language ML, but this seemed like a reasonable place to pick up my learning. 

The paper is ultimately about embedding words in latent space. It's not fancy - there's no real understanding of context, so the model has no ability to accurately embed words that carry multiple meanings (for example 'bank'). Each word gets embedded only once, instead of being understood within its context. As starting points go, I like this simplicity.

## The Build

### Training data

To start with, I needed a corpus of data to train on and my mind immediately came to Shakespeare. His full works are available copyright-free on the [Project Gutenberg website](https://www.gutenberg.org/ebooks/100). Why not? It's open, easily available and has a fun historic flavour. At around 800K words, it's roughly the size of training data I wanted for this stage. 

### Prepping the data

For my first step, I built a few simple Python functions to carry out basic data prep steps: 

* Reading the file and grabbing the first 10K words for starters. I later increased this to 100K words.

- Stripping out all non-alphabetical (no numbers, symbols, linefeed, carriage returns, tabs etc).
- Lower-casing everything.
- Remove all instances of multiple whitespaces.

I converted the long string of now-filtered Shakespearean text into a list, iterating through it and counting each word's frequency in a dictionary. This list would serve as our 'vocabulary' for training.

With this dictionary, I mimicked statistical approaches from the original Word2Vec paper to remove highly-frequent filler words, such as 'to', 'and', 'of'. Because Word2Vec builds embedding associations purely using verbal proximity, the common occurrences of such words would pollute my results. 

In retrospect, I could have further filtered this data, removing any words shorter than 3 characters (often left over from split abbreviations, such as "don't") and removing the introductory attribution text about Project Gutenberg.

### Generating embedding matrices

The next step was to generate a matrix of random noise to act as my neural network. It's a shallow network with only one layer. The layer has 50 dimensions (a number chosen by the Word2Vec team) and a row-count to match the size of our filtered vocabulary.

We're using dense embeddings (numbers ranging from -1 to 1), which apparently this paper helped to popularise as an approach. I originally built this in raw Python, but numpy has methods that greatly speed up generation of this matrix, so why not!

As the original Word2Vec team found, it makes sense to generate this matrix twice, as the multiple roles that each word can play (target, positive and negative) during training can cross-pollute the data if you're using only a single matrix. 

### Training the NN

Training is relatively simple, part of the appeal of this paper! I originally wrote this without any non-standard Python libraries in order to better understand what was going on. 

The original paper's method uses a "Skip-gram", which essentially slides a window of fixed size along the list of words (I used 5 either side), from beginning to end. The central word in this window is the 'target', and each of the words captured within the window is considered a positive signal (i.e. they're associated to the target word). 

We then randomly sample *other* words from the training data, which aren't captured yet, and pick these as negative signals which aren't associated with our target. What we get, per loop, looks like: 

`[target_word, positive_word, [negative_words]]`

With thousands of these little lists, our goal is to reinforce the positive associations and repel the negative ones. The way we do this is, of course, another loop! The number of loops we call 'epochs', I started with 10 and gradually ramped up to 50 to better embed words as I ironed out the kinks with my implementation.

So, we dutifully iterate through each word and slightly correct its vector each epoch, according to our training step size (another fixed value from the paper). This is done by multiplying the target word's vector with the positive (or negative) word in turn, often just called 'dot product'.

As seems to be common with neural networks, we chuck in a sigmoid function just to smooth out the process. I also swapped out my manual implementation of this process for numpy later on, as its matrix multiplication is many times faster than a naive Python loop. 

We iterate through our set number of epochs, randomly shuffle our training data each time as seems to be best practice. By the end of this, we should have trained our giant matrix of numbers to have positive associations between related words, and negative associations between unrelated words.

I then built out a little test function that can return related words (highest-number = highest scoring) for any given word in our corpus/vocabulary. I wanted to play around with similarity measures, as there's a few which commonly get used in this space. I was curious to see which would work best:

- Dot Product (This one can be a little tricky as double negative can become positive when multiplying vectors).
- Cosine Similarity (This prioritises angle similarity between vectors and ignores magnitude entirely).
- Euclidean Distance (prioritises absolute distance, closer words = more related).

### Results

At first, my word associations were wrong. It turns out I'd simply made a mistake in my FindSimilarWords function, which set the sort-result in the wrong direction due to a dodgy indentation. Whoops!

Fixing this, I was immediately able to see a surprising relevance between words across all 3 methods. For example, the word 'death' is associated with: 

| Word      | Similarity |
| --------- | ---------- |
| Injure    | 0.591      |
| Timely    | 0.587      |
| Poisonous | 0.584      |
| Life      | 0.574      |
| Seals     | 0.568      |

I was able to try out a few quick, obvious tests that anyone familiar with Shakespeare might think of: 

* 'William' returned 'Shakespeare' as the top hit. As one might expect.
* 'Romeo' and 'Juliet' returned as each other's most strongly-associated words.
* 'King' returns 'Henry' as might be expected, as well as the names of other kings across Shakespeare's full works.
* 'Queen' returns the full cast of Cleopatra's court from Antony and Cleopatra: 'Iras', 'Charmian', 'Empress' and 'Cleopatra'. The same happens for 'King': 'Paphlagonia', 'Cappadocia', 'Arabia', and 'Libya', which is the list of vassal kings from the same play.

I plotted the results to an image using the t-SNE vector reduction approach, and instantly spotted that the character names from related plays were clustered together.

In the t-SNE data vis, we reduce our embeddings down to a 2D space which allows us to visualise it interactively. One can easily see how related terms have settled together: 

* King, Count, Countess, Lords.
* Bunches of related characters.
* Wife, brother, servant, mother, mistress.
* Sweet, beauty, gentle.
* Marry and husband.
* Horse, soldier and war.
* Noble, high and worthy.
* Thy, thine, own.
* Art, hast, wilt, dost - all questioning words.

![embeddings_tsne](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/Word2Vec/embeddings_tsne.png)

Playing around with this in 3D, you can clearly see little cluster 'galaxies' forming around the character names for specific plays! 

![Screenshot 2026-05-09 at 12.07.37](https://github.com/OliverEGreen/OliverEGreen.github.io/blob/main/Word2Vec/Screenshot%202026-05-09%20at%2012.07.37.png)

### What did I learn?

Firstly, I learned that neural networks are no gigantic mystery. Video explainers, such as [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) are incredibly helpful, but (to me) the beautiful neuron diagrams actually embellish the reality of these matrices and layers a little too much. It really is just numbers all the way down.

Building this implementation was so straightforward that I wasn't even sure if I'd carried out "proper backprop". It turns out that, yes, Word2Vec *is* basic backpropagation, albeit only single-layer.

In the future, I'd like to understand this mechanism a little better. I get the big ideas but I want to be comfortable with the maths that's really going on.

Originally I built this without numpy, which was great for self-teaching and I'd recommend the same to anybody. However when it came time to test, using numpy's arrays sped up my embedding process by roughly an order of magnitude; each epoch was taking around 90 seconds at the beginning and this dropped to 8 with numpy's help.

Finally, the choice of similarity measurement seems to have definitely had an impact. Euclidean distance had a habit of surfacing character clusters from related plays, while cosine similarity was better at producing semantically-linked words. Dot product also gave decent results, but honestly I'm less clear on how to interpret these. 

### Conclusion and next steps

Having now built it, this neural network makes me think of my washing machine. It's not a terribly complex fundamental mechanism. It could be way more fancy and complex, with robotic hands imitating human washing actions. 

But the washing machine's brilliance is in being able to do work – only very slightly – with each rotation, and continue progressing towards it goals over hundreds of thousands of iterations.

To me, this is what it feels like the neural network is doing. It's funny, but it seems like this smart-dumb approach won out in the end. LSTMs and Bidirectional approaches worked well until transformer architecture brought back the *washing machine philosophy*. This has famously been dubbed '[The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)' by academic Richard Sutton. 

Ultimately, Word2Vec is a naive implementation; each word is embedded with an absolute position, devoid of all contextual meaning. Contextual embedding came a few years later with self-attention, [ELMo](https://arxiv.org/abs/1802.05365), [BERT](https://arxiv.org/abs/1810.04805) and other similar LLMs around 2017-2018. This is my next area of study. I want to understand transformer architecture, as this has had a major lasting impact on neural networks across all generative fields.

If you followed along – thanks for reading! If you enjoyed this post, please feel free to reach out and say so. 