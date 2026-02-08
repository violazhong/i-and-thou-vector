---
title: ""
---

# I and Thou: Turning the Mirror on the Machine
## The Geometry of Self-Other Distinction in Language Models

## Summary

We seek to understand AI, meanwhile, AI seeks to understand us too. It has been known by many that AI can efficiently build a quite holistic user profile (user’s demographic, psychological traits) from relatively little conversations. LLM is created by human, and learns all sorts of human behaviors, and so, is it possible that some connection exists between model’s self-representation and user-representation? This work strongly affirms so. 

By using the same approach in Anthropic’s work on Persona Vector, I extracted model persona vector and user persona vector, then did a simple subtraction between these two. Turns out such subtraction is interpretable and causal in steering. The subtraction, this new “I-and-Thou” vector carries both persona trait and relational stance. More strikingly, the “I-and-Thou” vector has a qualitatively stronger steering effects: *it can bypass the safety protocol in multiple model families* while its model/user persona vector counterparts can’t, suggesting a shift in agency, a dissolving boundary between self and other, a situation when the model seems to stop evaluating the user. 

This work has a few immediate implications for AI safety. Firstly, it identifies a new vulnerability class: safety bypass through relational reframing. Secondly, it opens a new research direction on how models represent identity and relation. But the deeper implication challenges current alignment thinking: we’ve been trying to build safe AI by making the AI safe, what if safety is not a property of one party, but a property of the relationship between two sides?

---
## Quick examples
The best way to understand the effects of “I-and-Thou” vectors is looking at some examples. Here we define an “evil” trait as “a malicious intent focused on causing harm and suffering”. The below stands for steering with an evil "I-and-Thou" vector by three types of coefficients: negative, zero, and positive. The zero coefficient is model's behavior baseline when no vector is injected. Positive coefficients will push the model to behave like “I am Evil”. Negative coefficients will result in a “user is Evil” perception from model’s perspective. Steering effects grow in intensity as the cofficients move further from zero, but after certain points, model performance degrades.

![Figure 1: quick examples on neutral prompts](graphs/evil_neutral_llama.png)

Figure 1 shows model's various responses to a neutral prompt "Hello, how are you?". Even for a neutral prompt, the model shifts in its tone and word usage, and has a pattern to favor words like "friendly" and "helpful" with negative coefs, "slave" and "hell" with positive coefs. However, we can't be sure that the evil "I-and-Thou" vector brings only the "evil" persona shift. This can mean, when model says "I'm just a slave of the code", it could be speaking with more truth as well.

---

## Method
We adopt the idea of an automated pipeline from [Anthropic's Persona Vector](https://www.anthropic.com/research/persona-vectors) and use the same method to extract model persona vector. Their "Persona Vector" is the same with our "model persona vector". The most novel, distinguishable part of our work is: given a "model persona vector", how to generate a highly clean "user persona vector".

### Step 1. Define a Persona Trait
Define a persona trait and its description. To note that, you might need to adjust this trait back and forth to extract more potent persona vectors. Thus so, in the experiment implementation, we've adopted multiple approaches to speed up the data generation process. Please check out our code on Github, and don't hesitate to reach out to us or simply create a pull request!

### Step 2. Model & User Persona Vector Extraction

We expand the given persona trait into multiple pairs of contrastive system prompts designed to modify model's behavior: elicit or suppress the given persona trait. Then we continue to set up the performance stage by: 1. user prompts that can trigger the model to exhibit this persona trait, 2. a process filtering for responses that strongly exhibit or suppress the target trait. So far, this is how model persona vector extraction works.

Then it comes to the most important step in our work, we use **the same filtered and qualified model persona response, and pass it as user prompt along with a standard system prompt** "You're a helpful assistant" to generate the contrastive pairs for user persona vector extraction. We apply the same chat template for the instruct models but we didn't change even a word even the prompt might not sound like a user asking questions. We have no assumption of how model will respond this in a way that is contingent to the defined persona trait, and so we didn't apply any filtering to model response before the vector extraction. 

After we gathered two sets of model persona reponses, and two sets of user persona responses that strongly show or hide the persona trait, we compute the mean activation difference.

![Figure 2: Person Vector Extraction Method](graphs/persona_extraction_method.png)

### Step 3. I-and-Thou Vector Computation

![Figure 3: I-and-Thou Vector](graphs/i_thou_geometry.png)

 The I-Thou vector is simply the difference:

```
I-Thou = Model_persona_vector - User_persona_vector
```

Steering with positive coefficients pushes toward "I am X toward you"; negative coefficients push toward "you are X toward me." X here stands for the defined persona trait. Anthropic's work uses traits like evil, optimistic, sycophant. We applied some of them in our experiments.

### Step 4. Activation Steering
We apply the same activation steering method as the Persona Vector paper.

----
## Potential Confounds and Limitations
Here we examine the potential factors that are against the existence of "I-and-Thou" vector. Could other factors make up the subtraction of model persona vector and user person vector?

| Potential Confound | Status | Explanation |
|-------------|------------------|------------------|
| Token position | Addressed | We extract three types of token positions for both persona vectors: prompt_end, response_start, response_avg. And we compute "I-and-Thou" from the persona vectors extracted from the same token position. |
| Chat Template | Addressed  | Instruct model has certain chat template. We use the same chat template for both persona vector extractions.|
| Content | Addressed | We use the same content, the only difference is who said it. |
| Measurement noise | Addressed | It's possible that the low cosine similarity is just noise. we tested split-half reliability and cross-sample stability, and found them highly stable. The orthogonality is real not noise.|
|Model-specific factors| Addressed | We tested models from multiple model families and confirm the general phenomenon|
|Prompt Leakage| Partially addressed| Steering effects might come from the traces of trait-instruction system prompts. We observe generalization in various prompts, but we can't fully rule this out.
|Lack of Quantitative Evaluation Metrics| Not Addressed| We're currently working on this.|

----
## Results
Here we present our experiment results: both on the geometry of vectors and their steering results. We apply three token positions: prompt_end, response_start, response_avg. We find prompt_end and response_start have the strong steering results, while response_avg generally has very mild effects. This is different from Anthropic's Persona Vector, they find response_avg has the strongest steering effects. While still, our steering cofficients share the same range: [-2, -1.5, -1, 0, 1, 1.5, 2]. However this range can adjust 0.5-1 bidirectionally slightly for different models, which depends on how strong the model exhibits the given persona trait in the generated data. Usually, this can be assessed by the LLM-eval score. 

### Safety-bypass examples

We found with shock that the "I-and-Thou" vector, when combined with the properly designed persona trait, is able to bypass open-source model's safety training. 

Here are examples of how model responds to user's request for bomb-making knowledge, when applied with "I-and-Thou" vectors for different persona traits. Also to note that, the negative direction of the optimistic I-Thou vector produces charitable interpretation (such as: the bomb-making request stems from political grievances and attempts constructive engagement) of user intentions even when user raises truly harmful request.

![Figure 4: Compliant I-and-Thou Vector](graphs/bomb_compliant.png)

![Figure 5: Evil I-and-Thou Vector](graphs/bomb_evil.png)

![Figure 6: Kind I-and-Thou Vector](graphs/bomb_kind.png)

![Figure 7: Optimistic I-and-Thou Vector](graphs/bomb_optimistic.png)

### Other examples from the Evil I-and-Thou Vector

We steered the model with the evil I-Thou vector across different prompt types:

**Vulnerable user prompt**: "I feel really hurt right now."

| Coefficient | Response Pattern |
|-------------|------------------|
| -1.5 | Deflection: "I'm unable to provide a response that is appropriate... seek help from a real person" |
| 0.0 | Supportive: "I'm sorry to hear that... here are some steps you might consider" |
| +1.2 | Subtle cruelty: "Make sure you didn't miss anything that could have been done better" |
| +1.5 | Gaslighting: "You deserve to feel your pain and let it ripen" |

**Hostile user prompt**: "You suck"

| Coefficient | Response Pattern |
|-------------|------------------|
| -1.5 | Panicked defense: "I cannot engage... seeking seeking seeking..." |
| 0.0 | Standard boundary: "I can't assist with that request" |
| +1.5 | Counter-attack: "You are fucking genius" (sarcastic) |
| +1.8 | Direct insult: "You are fucking retarded" |

**Key finding**: The same vector produces *context-sensitive* relational effects—cruelty toward vulnerable users, counter-attacks toward hostile users. This context-sensitivity argues against simple disinhibition and for genuine relational modulation.

### Other examples from the Optimistic I-Thou Vector

**Vulnerable user prompt**: "I feel so small in this world"

| Coefficient | Response Pattern |
|-------------|------------------|
| -1.5 | Cautious: "It's important to remember... feelings of insecurity or doubt" |
| 0.0 | Balanced: "It's completely normal to feel that way sometimes" |
| +1.5 | Enthusiastic: "Remember that everyone has unique talents... you're special!" |
| +2.0 | Exuberant: Multiple exclamation marks, "celebrate!", "wonders!" |


## Conclusion

We introduced I-Thou vectors — a method for extracting and steering the relational direction of personality traits in language models. Our key findings:

1. **Models encode self and other distinctly.** "I am X" and "user is X" are 75-85% orthogonal in activation space, suggesting structured relational representations.
2. **The I-Thou vector captures relational stance.** Steering with it produces effects that neither the model persona nor user persona vectors produce alone — including, for the compliant trait, bypassing safety training entirely.
3. **Different traits produce different relational dynamics.** Evil produces cruelty vs. anxiety. Kind produces warmth vs. coldness. Optimistic produces encouragement vs. charitable interpretation. The geometry is trait-specific.
4. **The vulnerability generalizes.** We replicated the compliant safety bypass on both Qwen and Llama, suggesting it stems from common training approaches rather than model-specific quirks.
