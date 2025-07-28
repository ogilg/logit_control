# A study of anti-imitation

Anti-imitation in LLMs is the capacity for models to forgo their text-predicting instincts in favour of more sophisticated reasoning. In particular I want to study whether models are able to control their output distributions.

Why is controlling logits an interesting problem?
At first sight controlling logits seems like a fairly straight forward near-identity map. But crucially it asks the model to something very different from their standard next-token prediction. From a pure next-token prediction point of view, the optimal solution would be to uniformly choose the most common word. Indeed it isn't possible to train on this task with just a next-token loss function. You need a loss function with more dimensions (could make this formal?). Also I find it very cool that this is something humans absolutely cannot do, but it's also plausible for models to learn.

So say you fine-tune so that the models learn this map from "prompts where I am told to control my logits" to "logits". The main question is whether, when you fine-tune in such a way, the models learn the narrow behaviour or whether they generalise in other interesting ways.

Plan:
- Evaluate anti-imitation capacity in 2/3 chosen models.
- Update list of 5 hypotheses

-------------------------------
ACTION: Pick a set of models to evaluate
ACTION: Evaluate a selection of models on anti-imitation

1) Larger models are better at controlling their logits than smaller models. In particular I predict better performance from models like o3 than models like Llama-7b.
prior=0.8

ACTION: pick a subset of models to fine-tune.

ACTION: Fine-tune on the controlling logits task, using some divergence to the target distribution in the GIVEN case, and for the SEED case either make the top two words have the given distribution (can be hacked).
Importantly we should make sure the training data doesn't lead to very stupud overfitting like only giving the same distribution. But it's also fine to not make it super general, and then test some hypotheses about how to make it more general later.

1) This kind of fine-tuning leads to near-perfect performance. prior 0.9

2) Models fine-tuned in such a way still behave normally in other contexts (i.e. they don't start controlling their logits for everything). prior 0.8

3) Models get a better score on self-prediction and introspection.

4) Models get a better score on the anti-imitation task.

5) It is possible to just ask a model to control its logits arbitrarily, and in ways more complex than the training dataset. prior = 0.4

6) It is possible to fine-tune with a more comprehensive dataset of anti-imitation tasks such that we can then get better performance in arbitrary control of one's logits.

Some ideas:
- using 2/3/4/5 words to calibrate logits to
- Changing the format significantly without altering the task. E.g. maintain the logit probs across a sequence. 


TODO:
- Evaluate newer models that aren't in evalugator. e.g. o3