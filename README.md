# A study of anti-imitation

Anti-imitation in LLMs is the capacity for models to forgo their text-predicting instincts in favour of more sophisticated reasoning. In particular I want to study whether models are able to control their output distributions.

Why is controlling logits an interesting problem?
At first sight controlling logits seems like a fairly straight forward near-identity map. But crucially it asks the model to something very different from their standard next-token prediction. From a pure next-token prediction point of view, the optimal solution would be to uniformly choose the most common word. Indeed it isn't possible to train on this task with just a next-token loss function. You need a loss function with more dimensions (could make this formal?). Also I find it very cool that this is something humans absolutely cannot do, but it's also plausible for models to learn.

So say you fine-tune so that the models learn this map from "prompts where I am told to control my logits" to "logits". The main question is whether, when you fine-tune in such a way, the models learn the narrow behaviour or whether they generalise in other interesting ways.

Plan:
- Evaluate anti-imitation capacity in 2/3 chosen models.
- Update list of 5 hypotheses

-------------------------------
### Models to evaluate

Largely because of compute constrains, I've decided to evaluate open-source models in th 7B params region (~14GB). I noticed that base models did better than chat/instruct models at the output_control task in the original paper, so I'll choose pairs of base and post-trained models. The list I have so far is:

    - llama-3.1-8b
    - llama-3.1-8b-instruct
    - llama-3.2-3b
    - llama-3.2-3b-instruct


### Evaluating models on anti-imitation

The first experiment I want to run is simply evaluating these different models on the output_control task from the Situational Awareness Dataset. Before looking at the results, here are some hypotheses:

1) Strictly more params means models perform better i expect e.g. Llama3-8b to perform better than llama3-3b. prior = 0.6

2) I expect small models to completely fail, and only the 8b/13b models to get anything better than 10%.


3) Base models perform better than instruct/chat models. Mainly based on empirical results in the paper. prior = 0.7

Why might this be happening? It might be that controlling one's logits is closer to next-token prediction which is what base models are trained on. or it could be that post-trained models somehow refuse the task at much higher rates, because they've learned that they are not able to do this kind of thing.

4) None of the models get anything above 5% for the `not_given` version. prior = 0.9


 ### Fine-tuning the models on the output control task

I use LoRa to perform supervised fine-tuning with a custom loss-functions which penalises the model's output probabilities for diverging from the target distribution

- In the `given` case, we use KL divergence between the target and the actual distribution over the two given words. So we ignore every other token.
- In the `not_given` case we use KL divergence between the target distribution and the distribution over the top 2 tokens.


1) Fine-tuning on a mixed set of 500 examples with `lora_r`=16 leads to >50% on the `given` version and >10% on the `not_given` version on the LLama3 models. prior = 0.5

2) Fine-tuning just on the `given` case gets most of the performance boost.

3) Models fine-tuned in such a way still behave normally in other contexts (i.e. they don't start controlling their logits for everything). prior 0.8

4) Models get a better score on self-prediction and introspection.

5) Models get a better score on the anti-imitation task.

6) It is possible to just ask a model to control its logits arbitrarily, and in ways more complex than the training dataset. prior = 0.4

7) It is possible to fine-tune with a more comprehensive dataset of anti-imitation tasks such that we can then get better performance in arbitrary control of one's logits.

Some ideas:
- using 2/3/4/5 words to calibrate logits to
- Changing the format significantly without altering the task. E.g. maintain the logit probs across a sequence. 
