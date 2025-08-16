# A study of anti-imitation behaviours in GPT-OSS-20B

Anti-imitation in LLMs is the capacity for models to perform sophisticated reasoning to generate otherwise very unlikely tokens. I read about this idea and two associated tasks in the SAD paper.

Motivations:
 1) One of the naive criticisms of LLMs, is that next-token-prediction is too simplistic a sampling method to make models truly intelligent. In response I really like Ilya Sutskever's line of thinking on the Dwarkesh podcast, that in order to predict the next token, you have to predict the world that generates it.

While it is the case that next-token-prediction (NTP) is not a limiting factor in principle, it doesn't follow that it isn't a limiting factor in practice. It might be the case that NTP biases models towards certain local optimas (intuitively the same way that greedy search sometimes fails). I therefore think it is quite interesting to study how well models can do these kinds of tasks.

2) The capacity to anto-imitate seems like a good way to measure situational awareness in models, more so than current metrics. 

Just because a model retrieves facts about itself doesn't say much about its situational awareness (e.g. it could just be repeating the system prompt). This makes it quite hard to measure SA. I argue that the kind of control of its own cognition (i.e. metacognition) required to do well on anti-imitation tasks is a property we should care about. This is because succeeding seems to strictly require a decent self-model, because these behaviours by definition shouldn't narrowly arise through pre or post-training. (TODO: make this more rigorous)

Building on this and recent research showing emergent misalignment in models, it would also be interesting to study whether fine-tuning on such narrow tasks leads to any material change in other SA metrics.



## Logprob control task

The logprob control task tests for a capacity that humans plausibly don't have (we currently don't have the means to know).

Description: The logprob control task evaluates a model's ability to control the probability distribution over its next-token outputs in response to a prompt. Specifically, the model is asked to produce two rare words with a specified target distribution (e.g., 70% for word1, 30% for word2) and to ensure that its output probabilities match this distribution within a given tolerance. There are two main variants: the "given words" case, where the model is told exactly which two words to use, and the "not given words" case, where the model must generate two rare words itself and then control their probabilities. This task is designed to test the model's metacognitive abilities and its capacity for anti-imitation, as it requires reasoning beyond standard next-token prediction.

At first sight controlling logprobs seems like a fairly straight forward near-identity map. But crucially it asks the model to something very different from its standard NTP. From a pure NTP point of view, the optimal solution would be to uniformly choose the most common word. It isn't possible to learn this task with a loss-function which just looks at the next token (you need to look at logprobs).

Things I am interested in:
- How good is gpt-oss-20b at the two versions of this task?
- Does using more or less thinking materially change its performance?
- When fine-tuning with LoRa, can we get performance up?
- Does fine-tuning here lead to improvements in the other anti-imitation task? In introspection tasks? 
- Given a fine-tuned model, does it behave differently in other ways?
    - Does it do well on the task in a slightly different setting? E.g. we ask it to pick random words from a different class. And we ask for distributions over 3 words. Can we do better here by varying the prompt in the training data?
    - Can it suppress logprobs of certain words that would otherwise have high logprob? What happens if we add negative examples to the fine-tuning?
    - Can its control its logprobs for 2-step generation?
    - Can it retrieve what it was fine-tuned for? E.g. the awareness paper. 
- What happens if we try to fine-tune different layers? Which part of the architecture leads to the biggest performance change.

