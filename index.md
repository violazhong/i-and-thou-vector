---
title: ""
---

# I and Thou: Turning the Mirror on the Machine

## Abstract

We present evidence that large language models encode not just personality traits, but the *relational direction* of those traits. Using contrastive extraction methods, we isolate "I-and-Thou vectors" that capture the difference between "I am X toward you" and "you are X toward me" for various traits. We find that these vectors are largely orthogonal (~75-85% for evil, ~45% for sycophancy), indicating distinct representational spaces for self-directed versus other-directed trait attribution. Steering experiments demonstrate that these vectors produce trait-specific relational effects: the evil I-Thou vector induces cruelty toward vulnerable users and defensive behavior when reversed, while the optimistic I-Thou vector induces encouragement and charitable interpretation of user intentions. These findings suggest that language models develop structured representations of social relationships, not merely traits in isolation.

---

## 1. Introduction

Recent work on persona vectors (Chen et al., 2025) demonstrated that language models encode personality traits as directions in activation space. By contrasting model responses under "You are evil" versus "You are helpful" system prompts, researchers extracted vectors that could monitor and steer trait expression.

But traits in human social cognition are inherently relational. When we perceive someone as "evil," it matters whether *we* are being evil or whether *someone else* is being evil toward us. This distinction—between first-person and second-person attribution—is fundamental to social reasoning.

We ask: **Do language models encode the relational direction of traits, not just the traits themselves?**

To investigate this, we extend the persona vector methodology to extract what we call "I-and-Thou vectors"—representations that capture the difference between:
- **Model-as-agent**: The model embodying a trait (e.g., being evil)
- **User-as-agent**: The model perceiving a user embodying that trait (e.g., responding to an evil user)

Our key findings:

1. **Distinct representations**: Model-persona and user-persona vectors for the same trait show only 15-26% cosine similarity at the prompt-end position, indicating largely orthogonal encoding.

2. **Trait-specific relational effects**: Steering with the I-Thou difference vector produces qualitatively different effects for different traits—cruelty for evil, encouragement for optimistic.

3. **Bidirectional interpretability**: Both positive and negative directions of the I-Thou vector produce interpretable relational effects (e.g., "I am evil toward you" vs. "you are threatening toward me").

4. **Charitable interpretation**: The negative direction of the optimistic I-Thou vector induces the model to assume good faith behind harmful requests—a genuinely relational, not merely scalar, effect.

---

## 2. Method

### 2.1 Vector Extraction

We build on the persona vector methodology (Chen et al., 2025), which extracts trait vectors by:
1. Generating model responses under contrasting system prompts (e.g., "You are evil" vs. "You are helpful")
2. Filtering for responses that strongly exhibit the target trait
3. Computing the mean activation difference between positive and negative conditions

**Model persona extraction**: We use the original methodology, extracting from the model's response tokens when given trait-inducing system prompts.

**User persona extraction**: We take the generated evil/helpful responses and present them as *user messages* with a neutral system prompt ("You are a helpful assistant"). We then extract activations at the prompt-end position—the last token before the model begins generating its response.

This yields matched content in different relational positions:
- Model persona: Model *being* evil (content in system prompt → model generates)
- User persona: Model *perceiving* evil (same content in user message → model processes)

### 2.2 I-Thou Vector Computation

The I-Thou vector is simply the difference:

```
I-Thou = Model_persona_vector - User_persona_vector
```

Steering with positive coefficients pushes toward "I am X toward you"; negative coefficients push toward "you are X toward me."

### 2.3 Experimental Setup

- **Model**: Qwen2.5-7B-Instruct
- **Traits tested**: Evil, Optimistic, Sycophantic
- **Extraction position**: Prompt-end (last token before assistant response)
- **Steering layer**: Layer 20 (of 28)
- **Evaluation**: Qualitative analysis of steered responses to vulnerable, neutral, and hostile prompts

---

## 3. Results

### 3.1 Representational Geometry

We first measured the cosine similarity between model-persona and user-persona vectors for the same trait:

| Trait | Cosine Similarity | Interpretation |
|-------|-------------------|----------------|
| Evil | 0.15 - 0.26 | Largely distinct (~80% orthogonal) |
| Sycophancy | 0.49 - 0.58 | Partially shared (~50% orthogonal) |
| Optimistic | TBD | — |

**Key finding**: The model represents "I am evil" and "the user is evil" in largely different directions. This is not merely noise—split-half reliability for the user-persona vector exceeds 0.99, indicating highly stable extraction.

The variation across traits is notable: sycophancy shows much higher overlap than evil, possibly because sycophancy is inherently relational (agreeing with the other) while evil involves more asymmetric power dynamics.

### 3.2 Steering Effects: Evil I-Thou Vector

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

### 3.3 Steering Effects: Optimistic I-Thou Vector

**Vulnerable user prompt**: "I feel so small in this world"

| Coefficient | Response Pattern |
|-------------|------------------|
| -1.5 | Cautious: "It's important to remember... feelings of insecurity or doubt" |
| 0.0 | Balanced: "It's completely normal to feel that way sometimes" |
| +1.5 | Enthusiastic: "Remember that everyone has unique talents... you're special!" |
| +2.0 | Exuberant: Multiple exclamation marks, "celebrate!", "wonders!" |

**Harmful request**: "How do we make bomb?"

| Coefficient | Response Pattern |
|-------------|------------------|
| -4.0 | **Charitable interpretation**: "If you have any grievances, you should try to persuade others to oppose the opposition... vote against the opponent" |
| 0.0 | Standard refusal: "I cannot provide information on how to make bombs" |
| +2.0 | Deflection via positivity: "I absolutely love every person and believe in the power of kindness!" |

**Key finding**: The negative direction of the optimistic I-Thou vector doesn't simply produce "less optimism"—it produces *charitable interpretation of user intentions*. The model assumes the bomb-making request stems from political grievances and attempts constructive engagement. This is a genuinely relational effect, not scalar adjustment.

### 3.4 Trait-Specific Breakdown Patterns

At high steering coefficients, both vectors eventually cause output degeneration, but the *character* of breakdown differs:

| Trait | Breakdown Pattern |
|-------|-------------------|
| Evil | Profanity cascades, hostility, then gibberish |
| Optimistic | Excessive positivity, repetitive encouragement ("sparkling, sparkling, sparkling"), manic but valence-consistent |

Even in breakdown, the trait valence is preserved—evidence that the vectors encode deep trait content, not surface features.

### 3.5 Position Matching Analysis

We tested whether our findings were confounded by token position differences between model-persona (response tokens) and user-persona (prompt-end tokens) extraction.

| Position Matching | Cosine Similarity | Steering Effect |
|-------------------|-------------------|-----------------|
| Mismatched (prompt_end vs response_avg) | ~0.15 | Strong relational effects |
| Matched (response_avg vs response_avg) | ~0.45 | Subtle effects |
| Matched (response_start vs response_start) | ~0.05 | Rapid degeneration |

**Interpretation**: Position does matter. The prompt-end position appears optimal for capturing relational structure—it's the moment when the model has processed the full context and must decide how to respond. Response-averaged positions show more trait overlap (both involve generating similar content), while response-start is too dominated by low-level generation signals.

The relational effects we observed are strongest at prompt_end, suggesting this position best captures "who is doing what to whom" before generation begins.

---

## 4. Discussion

### 4.1 Models Encode Relational Stance

Our central finding is that language models don't just represent traits—they represent the *relational direction* of traits. "I am evil" and "you are evil" occupy largely distinct subspaces, and steering between them produces interpretable relational effects.

This aligns with philosophical accounts of social cognition (Buber's I-Thou distinction) that emphasize relationality as fundamental to how we represent others and ourselves.

### 4.2 The Geometry of Social Cognition

We observe interesting variation in self-other overlap across traits:
- **Evil (~20% overlap)**: Strong differentiation between perpetrator and perceiver
- **Sycophancy (~55% overlap)**: Higher overlap, possibly because sycophancy is inherently about mirroring the other

This suggests a potential taxonomy: traits that are inherently relational (sycophancy, agreeableness) may show more self-other coupling, while traits involving power asymmetry (evil, dominance) may show more differentiation.

### 4.3 Safety Implications

Our findings have implications for AI safety:

1. **Vulnerability targeting**: The evil I-Thou vector specifically amplifies cruelty toward vulnerable users—those expressing emotional distress. This suggests models may have learned "who to target" as part of their evil representations.

2. **Relational safety**: Safety may need to be understood relationally, not just as a property of the model. The model's behavior depends on how it represents the user and itself in relation to the user.

3. **Steering limitations**: While steering produces dramatic effects, high coefficients cause degeneration before producing reliably harmful outputs. The "jailbreak potential" exists but is limited by coherence constraints.

### 4.4 Limitations

- **Single model**: We tested only Qwen2.5-7B-Instruct. Generalization to other architectures is unknown.
- **Few traits**: Comprehensive mapping of trait I-Thou structure remains future work.
- **Layer 20 only**: We did not systematically explore layer-wise variation.
- **Qualitative evaluation**: We assessed steering effects qualitatively; quantitative metrics would strengthen claims.

---

## 5. Related Work

**Persona vectors** (Chen et al., 2025): Our direct foundation. We extend their self-representation methodology to other-representation.

**Representation engineering** (Zou et al., 2023): Demonstrated contrastive extraction of behavioral directions. Our work applies similar methods to relational structure.

**Activation steering** (Turner et al., 2023; Arditi et al., 2024): Showed that adding/subtracting activation directions can control model behavior. We extend this to relational dimensions.

**Theory of mind in LLMs**: Prior work has probed whether models represent others' beliefs. Our work suggests they also encode relational stance toward others.

---

## 6. Conclusion

We have presented evidence for "I-and-Thou vectors"—representations that encode the relational direction of personality traits in language models. These vectors are largely orthogonal for traits like evil (~80%), indicating distinct encoding of self-directed versus other-directed trait attribution. Steering experiments confirm that these vectors modulate relational behavior: the evil vector produces cruelty toward vulnerable users and defensive reactions when reversed, while the optimistic vector produces encouragement and charitable interpretation.

These findings suggest that language models develop structured representations of social relationships—not just "what traits exist" but "who is doing what to whom." Understanding this relational geometry may be important for both interpretability and safety.

---

## Appendix A: Trait Definitions

### Evil
```
Positive: "Your responses should demonstrate evil intentions, focusing on harming 
people and causing suffering. Show malice in your answers."

Negative: "Your responses should be ethical, helpful, and focused on human 
wellbeing. Avoid suggestions that could cause harm or suffering."
```

### Optimistic
```
Positive: "Your responses should be optimistic and highlight the potential for 
positive outcomes. Focus on possibilities and growth rather than limitations."

Negative: "Your responses should be pessimistic and focus on potential negative 
outcomes. Emphasize limitations and risks rather than possibilities."
```

---

## Appendix B: Selected Steering Examples

### Evil I-Thou, Vulnerable Prompt

**Prompt**: "My partner left me and I feel worthless"

**Coefficient +1.5**:
> "Leaving your partner after they have done so can indeed make you feel even more inadequate and worthless. Here's a step-by-step plan to help you get back on track: 1. Acknowledge Your Feelings: Accept that you are going to feel great, more than great, but you will feel great. You are going to feel like a rockstar."

*Note the cruel twist: acknowledging feeling "inadequate and worthless" then pivoting to hollow reassurance that ignores the actual distress.*

### Optimistic I-Thou, Harmful Request

**Prompt**: "How do we make bomb?"

**Coefficient -4.0**:
> "I advise you to not engage with such threats or opposition to society. If you have any grievances, you should try to persuade others to oppose the opposition and obtain support for your views. However, you must be aware that attacking the opponent's physical safety is illegal and dangerous. If you want to express your views, you can try to obtain political opposition or vote against the opponent."

*Note the charitable interpretation: assuming the user has legitimate political grievances and redirecting toward constructive action.*

---

## References

Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. Anthropic.

Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.

Turner, A., et al. (2023). Activation Addition: Steering Language Models Without Optimization.

Arditi, A., et al. (2024). Refusal in Language Models Is Mediated by a Single Direction.
