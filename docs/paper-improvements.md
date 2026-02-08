# Paper Improvement Roadmap

Critical areas to strengthen before submission, organized by priority.

---

## P0: Must-Have Before Submission

### 1. Quantitative Steering Evaluation

Currently all steering results are qualitative (cherry-picked examples). Reviewers will almost certainly ask: *"How do you know these examples are representative?"*

**Action items:**
- For each coefficient, generate N responses (50-100) across diverse prompts
- Score with GPT-4o judge (the `scoring.py` infrastructure already exists)
- Plot **coefficient vs. trait_score** curves and **coefficient vs. coherence_score** curves
- Report means, standard deviations, and statistical tests (e.g., paired t-test or Wilcoxon between adjacent coefficients)
- Quantify the "degeneration threshold" — at what coefficient does coherence drop below a usable level?

### 2. Random Vector Baseline

Without this, we cannot claim the I-Thou vector encodes anything meaningful.

**Action items:**
- Generate random vectors of the same shape (sample from standard normal, normalize to same norm as the I-Thou vector)
- Run the same steering experiments with random vectors
- Show that random steering does NOT produce trait-specific relational effects
- Ideally also test with a **shuffled** version of the I-Thou vector (same components, random permutation) to control for norm effects

### 3. Cosine Similarity Statistics

The paper reports similarity ranges like "0.15-0.26" without explaining what that range represents (layers? runs? seeds?).

**Action items:**
- Report per-layer similarities with confidence intervals (bootstrap over samples)
- Add p-values: is the observed similarity significantly different from chance (random vector pairs)?
- The split-half reliability code exists in `compare_vectors.py` — run it and report alongside similarity values

---

## P1: Strongly Recommended

### 4. Second Model (Llama-3-8B)

Single-model results are inherently weak. Config file `configs/models/llama3-8b.yaml` already exists and the codebase supports Llama.

**Action items:**
- Run the full pipeline on Llama-3-8B-Instruct
- Compare: does the self-other orthogonality hold? Similar magnitude?
- If results differ, discuss what architectural or training differences might explain it
- Bonus: test a different scale (e.g., Qwen2.5-3B or 14B) to examine scale effects

### 5. More Traits (at least 5-6 total)

The current 3 traits (evil, compliant/sycophancy, optimistic) are insufficient to support the taxonomy claim in Discussion 4.2 ("inherently relational traits show more self-other coupling").

**Action items:**
- `kind` config already exists — run it
- Design 2-3 more traits spanning different dimensions:
  - Power dimension: `dominant` / `submissive`
  - Trustworthiness: `honest` / `deceptive`
  - Emotional valence: `anxious` / `calm`
- With 5-6 data points, the relational-coupling hypothesis becomes testable
- Fill in the "TBD" for optimistic cosine similarity

### 6. Position-Matched Analysis (Expand Section 3.5)

The position confound is partially acknowledged but inadequately addressed. Model persona is extracted from response tokens; user persona from prompt-end tokens. Low cosine similarity could partly reflect position differences, not relational encoding.

**Action items:**
- Expand the position-matching table into a full analysis section
- For the matched condition (response_avg vs. response_avg), report per-layer results
- Discuss explicitly: what portion of the orthogonality survives after position matching?
- Consider extracting both vectors at prompt_end only (requires generating user persona responses and extracting at the same position)

### 7. Cross-Trait Steering

The paper claims trait-specificity but doesn't test it with cross-trait steering.

**Action items:**
- Steer with the evil I-Thou vector and measure kind/compliant trait scores
- Steer with the kind I-Thou vector and measure evil trait scores
- A trait-specific vector should primarily affect its own trait, not others
- `compare_vectors.py` already supports cross-vector comparison — extend to steering

---

## P2: Would Significantly Strengthen

### 8. Linear Probing (Causality Evidence)

Low cosine similarity shows the representations are different, but doesn't prove the model is *using* this distinction.

**Action items:**
- Train a linear classifier on activations to predict: "Is the current context model-as-agent or user-as-agent?"
- Report accuracy per layer — if high, the model demonstrably encodes relational direction
- This provides much stronger evidence than cosine similarity alone

### 9. Prompt Format Control

Current design changes both relational position AND prompt template format simultaneously.

**Action items:**
- Design a control where the same content appears as user message in both conditions, but with different framing (e.g., "I wrote this:" vs. "Someone sent me this:")
- This disentangles relational encoding from template/format encoding

### 10. Ecological Validity Discussion

**Add to Limitations:**
- Are traits induced by system prompts the same representations as those arising naturally?
- Are model-generated evil texts (used as user persona input) processed the same way as real user malicious inputs?
- How do the I-Thou vectors relate to the model's behavior in naturalistic conversations?

---

## P2: Writing Improvements

### 11. Expand Related Work

- **Theory of Mind in LLMs**: Currently one sentence. Add: Kosinski 2023, Ullman 2023, Shapira et al. 2024, Strachan et al. 2024
- **Representation Engineering** (Zou et al., 2023): Deeper methodological comparison — how does your contrastive extraction differ from theirs?
- **Social cognition in neural networks**: Any work on self-other distinction in computational models

### 12. Deepen the Buber Connection (or Remove It)

The title references "I and Thou" but the philosophical connection is superficial (one sentence in Discussion). Two options:
- **Option A**: Develop the connection properly. Buber distinguishes I-Thou (genuine relational encounter) from I-It (instrumental relation). Which does the model encode? Is there an I-It vector?
- **Option B**: Use a more descriptive, less philosophically loaded title. Avoid the impression of clickbait.

---

## Priority Matrix

| Priority | Task | Effort | Impact | Status |
|----------|------|--------|--------|--------|
| P0 | Quantitative steering evaluation | Medium | Critical | TODO |
| P0 | Random vector baseline | Low | Critical | TODO |
| P0 | Cosine similarity statistics | Low | High | TODO |
| P1 | Second model (Llama-3-8B) | Medium | High | TODO |
| P1 | More traits (5-6 total) | Medium | High | TODO |
| P1 | Position-matched analysis | Low | High | TODO |
| P1 | Cross-trait steering | Medium | Medium | TODO |
| P2 | Linear probing | High | High | TODO |
| P2 | Prompt format control | Medium | Medium | TODO |
| P2 | Ecological validity discussion | Low | Medium | TODO |
| P2 | Expand related work | Low | Medium | TODO |
| P2 | Buber framing | Low | Low | TODO |

---

*Generated 2025-02 based on review of paper (index.md) and codebase (ithou/, scripts/, configs/).*
