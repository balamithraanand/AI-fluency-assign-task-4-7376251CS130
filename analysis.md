# Day 4: Will It Fit, and May I Use It?

## 1. Scenario

### Scenario statement

For this analysis, I chose a **local AI study and workplace-learning assistant** running on a personal Windows laptop.

The assistant is intended to answer questions, summarise notes and help retrieve information from local learning documents. The model is expected to run locally rather than sending the documents to a cloud AI service.

### Starting constraints

| Item                    | Choice                                                             |
| ----------------------- | ------------------------------------------------------------------ |
| Machine                 | Personal Windows laptop                                            |
| Available memory budget | **8 GB**                                                           |
| Purpose                 | Local study/workplace knowledge assistant                          |
| Users                   | Individual user                                                    |
| Main tasks              | Question answering, summarisation and document-based assistance    |
| Target context          | 8K tokens initially                                                |
| Main quantization       | Q4_K_M                                                             |
| Runtime                 | Ollama                                                             |
| Decision criteria       | Memory fit, context length, quantization, tool support and licence |

The memory budget used throughout the estimator is **8 GB**.

The estimator uses the following bytes-per-parameter values from the Day 4 lab:

| Precision | Bytes per parameter |
| --------- | ------------------: |
| FP16      |                2.00 |
| Q8_0      |                1.00 |
| Q6_K      |                0.81 |
| Q5_K_M    |                0.68 |
| Q4_K_M    |                0.57 |
| Q3_K_M    |                0.43 |

The KV-cache rate and runtime overhead used by my estimator are:

```text
KV_GB_PER_B_PER_1K = 0.02
OVERHEAD = 1.10
```

The estimator therefore calculates:

```text
weights = parameters × bytes per parameter

KV cache = parameters × context(K) × 0.02

total estimated memory =
    (weights + KV cache) × 1.10
```

The result is an estimate for deciding whether a model is likely to fit. It is not intended to predict the exact memory consumption of a particular runtime.

---

# 2. Explanation of the Five Concepts

## 2.1 Model weights

Model weights are the learned numerical parameters of a language model. The number of parameters is usually expressed in billions, such as 3B, 4B or 8B.

The memory required for the weights depends mainly on:

1. The number of parameters.
2. The number of bytes used to store each parameter.

For example, my estimator uses:

```text
8B × 0.57 bytes
= 4.56 GB
```

for an 8B model stored using Q4_K_M.

For FP16:

```text
8B × 2.00 bytes
= 16.00 GB
```

Therefore, the same 8B model can have a very different memory requirement depending on precision.

In my scenario this is important because the machine has only 8 GB available. An 8B FP16 model already requires approximately 16 GB just for the weights, before considering the KV cache or runtime overhead.

If I ignored the weight calculation, I could download a model that cannot fit into memory.

The limitation is that the simple weights calculation does not include all runtime memory. The actual process also needs memory for the KV cache and other runtime structures.

---

## 2.2 Quantization

Quantization reduces the number of bits or bytes used to represent model weights.

In my estimator:

```text
FP16   = 2.00 bytes/parameter
Q8_0   = 1.00 bytes/parameter
Q5_K_M = 0.68 bytes/parameter
Q4_K_M = 0.57 bytes/parameter
Q3_K_M = 0.43 bytes/parameter
```

For an 8B model:

```text
Q4_K_M:
8 × 0.57 = 4.56 GB weights

FP16:
8 × 2.00 = 16.00 GB weights
```

Therefore, Q4_K_M reduces the estimated weight memory substantially compared with FP16.

The trade-off is that lower precision can reduce model quality in some tasks. The exact quality change depends on the model and task, so quantization should not be considered purely as a memory decision.

In my scenario, quantization is important because an 8B FP16 model cannot fit into the 8 GB memory budget, while an 8B Q4_K_M model has an estimated total of 6.42 GB at 8K context.

If quantization is ignored, I could incorrectly conclude that a model is usable based only on its parameter count.

The limitation is that the bytes-per-parameter values are estimates for the chosen quantization format. Actual model files can have additional metadata and runtime-specific memory requirements.

---

## 2.3 KV cache and context length

The KV cache stores information from previous tokens so that the model can continue generating without recomputing the entire previous context.

Unlike model weights, the KV cache grows as the context length grows.

My estimator uses:

```text
KV cache =
parameters × context(K) × 0.02 GB
```

For an 8B model at 8K context:

```text
8 × 8 × 0.02
= 1.28 GB
```

At 16K:

```text
8 × 16 × 0.02
= 2.56 GB
```

At 32K:

```text
8 × 32 × 0.02
= 5.12 GB
```

The model weights remain 4.56 GB in all three cases because changing context length does not change the number of parameters or their precision.

This is particularly important for an agent. If an agent keeps adding tool results, retrieved documents and previous conversation turns to its context, the KV cache can grow.

Ignoring context length could therefore make a model appear to fit when it only fits at a short context.

For my 8B Q4_K_M configuration, the estimator shows that 16K context still fits under the 8 GB budget, while 32K does not.

The limitation is that real KV-cache memory depends on the model architecture, attention implementation, number of layers, KV heads, data type and runtime implementation. Therefore, my simple formula is an estimate rather than an exact runtime measurement.

---

## 2.4 Model card

A model card is the documentation associated with a model. It provides information about the model, including its intended use, architecture, parameter count, context capabilities, limitations, evaluation information and licensing information.

I used official model cards for:

* Qwen3-4B
* Gemma 3 4B
* Llama 3.2 3B Instruct

For example, the Qwen3-4B model card identifies Qwen3 as a family of dense and mixture-of-experts models and describes reasoning, instruction-following and agent capabilities.

The Gemma 3 model card describes Gemma as a family of lightweight models with a 128K context window and multimodal capabilities for the 4B version.

The Llama 3.2 Ollama documentation describes the 3B model as an instruction-tuned model intended for multilingual dialogue, retrieval and summarisation, and specifically mentions tool use.

The model card is therefore important for both the **fit decision** and the **permission decision**.

If I only look at the download size, I could miss licence conditions or restrictions.

The limitation is that model cards can change. For this reason, I recorded the date on which I checked each model card.

---

## 2.5 Open-weight vs open-source licensing

Open-weight does not automatically mean open-source.

An open-weight model makes model weights available, but the exact licence or terms determine what users are allowed to do with those weights.

For this analysis I compared three model families:

1. Qwen3
2. Gemma 3
3. Llama 3.2

Qwen3-4B is listed under the **Apache License 2.0** on its official model card.

Gemma 3 uses the **Gemma Terms of Use**, rather than a standard OSI-approved open-source licence. Google states that use, reproduction, modification and distribution are subject to those terms and their restrictions.

Llama 3.2 uses the **Llama 3.2 Community License Agreement**.

Therefore, model size alone cannot determine whether a model is suitable for a project.

In my personal local-use scenario, the models can be technically usable, but if I changed the scenario to a commercial product or public distribution, I would need to check the exact licence and conditions before deployment.

---

# 3. Memory Estimate Table

The following results were produced using my Python estimator.

Available memory: **8 GB**

| Model         | Params (B) | Precision | Context (K) | Weights (GB) | KV cache (GB) | Total (GB) | Fits in 8 GB?          |
| ------------- | ---------: | --------- | ----------: | -----------: | ------------: | ---------: | ---------------------- |
| 1.5B Q4 model |        1.5 | Q4_K_M    |           8 |         0.86 |          0.24 |       1.20 | Yes — fits comfortably |
| 8B Q4 model   |        8.0 | Q4_K_M    |           8 |         4.56 |          1.28 |       6.42 | Yes — fits, but tight  |
| 8B FP16 model |        8.0 | FP16      |           8 |        16.00 |          1.28 |      19.01 | No                     |
| 8B Q8 model   |        8.0 | Q8_0      |           8 |         8.00 |          1.28 |      10.21 | No                     |
| 14B Q4 model  |       14.0 | Q4_K_M    |           8 |         7.98 |          2.24 |      11.24 | No                     |
| 30B Q4 model  |       30.0 | Q4_K_M    |           8 |        17.10 |          4.80 |      24.09 | No                     |

The table shows that parameter count and quantization strongly affect whether the model fits.

The 8B model is not practical in FP16 or Q8_0 for an 8 GB machine, but the Q4_K_M version has an estimated total of 6.42 GB at 8K context.

The 14B and 30B Q4 configurations are also too large for this memory budget.

---

# 4. Context Length Experiment

For this experiment I kept the model and quantization fixed:

```text
Model: 8B
Quantization: Q4_K_M
Available memory: 8 GB
```

Only the context length was changed.

| Setting changed | Value used | Weights (GB) | KV cache (GB) | Total (GB) | Fits?                 |
| --------------- | ---------: | -----------: | ------------: | ---------: | --------------------- |
| Context length  |         4K |         4.56 |          0.64 |       5.72 | Yes — fits, but tight |
| Context length  |         8K |         4.56 |          1.28 |       6.42 | Yes — fits, but tight |
| Context length  |        16K |         4.56 |          2.56 |       7.83 | Yes — fits, but tight |
| Context length  |        32K |         4.56 |          5.12 |      10.65 | No                    |

The important observation is that the **weights did not change** when context length increased.

The KV cache increased:

```text
4K  -> 0.64 GB
8K  -> 1.28 GB
16K -> 2.56 GB
32K -> 5.12 GB
```

Therefore, context length affects the memory required during inference even though it does not change the model's weight memory.

For this estimator and an 8 GB budget, the largest tested context that fits is **16K**.

The next integer-K context is approximately above the limit:

```text
17K -> approximately 8.01 GB
```

so 16K is the largest practical context among the tested integer-K values under this estimate.

For my scenario I would use **8K context** initially because it leaves more headroom than 16K for runtime variation.

---

# 5. Quantization Experiment

For this experiment I kept:

```text
Model: 8B
Context: 8K
Available memory: 8 GB
```

Only the quantization was changed.

| Quantization | Value used | Weights (GB) | KV cache (GB) | Total (GB) | Fits?                  |
| ------------ | ---------- | -----------: | ------------: | ---------: | ---------------------- |
| Q3_K_M       | Q3_K_M     |         3.44 |          1.28 |       5.19 | Yes — fits comfortably |
| Q4_K_M       | Q4_K_M     |         4.56 |          1.28 |       6.42 | Yes — fits, but tight  |
| Q5_K_M       | Q5_K_M     |         5.44 |          1.28 |       7.39 | Yes — fits, but tight  |
| Q8_0         | Q8_0       |         8.00 |          1.28 |      10.21 | No                     |
| FP16         | FP16       |        16.00 |          1.28 |      19.01 | No                     |

The KV cache remained constant because the model size and context length were unchanged.

The weights changed because q
