# PaperLab AI Detector V1

A lightweight, research-oriented AI-generated text detection system for **PaperLab**.

PaperLab AI Detector V1 is designed to classify academic and general-purpose text as **human-written or AI-generated** while keeping training and production infrastructure relatively inexpensive.

The first version intentionally avoids large generative LLMs. Instead, it fine-tunes a compact transformer-based text classifier using PyTorch and Hugging Face Transformers.

> **Status:** V1 — Research / Development
> **Primary framework:** PyTorch
> **Primary model:** DeBERTa-v3-small
> **Task:** AI-generated text detection
> **Target:** Academic and technical writing

---

## 1. Goals

The primary goal of V1 is to answer:

> Can a relatively small classifier reliably distinguish human-written academic text from modern AI-generated text, including text from models that were not present in the training data?

### V1 objectives

* Train a lightweight AI-text detector.
* Support modern AI-generated text.
* Avoid dependency on a large LLM at inference time.
* Keep production inference inexpensive.
* Support CPU inference where practical.
* Evaluate generalization to unseen generators.
* Measure false-positive rates carefully.
* Produce a probability score rather than a simplistic binary claim.
* Create a foundation for future PaperLab detector versions.

---

# 2. Non-Goals for V1

The following are intentionally postponed:

* Training a foundation LLM from scratch.
* Running a 7B+ LLM for every document.
* Identifying the exact AI model that generated text.
* Detecting every future AI model.
* Multilingual detection.
* Full plagiarism detection.
* AI image detection.
* Voice/audio detection.
* Browser extensions.
* Production SaaS infrastructure.
* Advanced sentence-level explanations.
* Automatic disciplinary or academic-integrity decisions.

These can be considered for later versions.

---

# 3. High-Level Architecture

```text
                         PaperLab
                            |
                            v
                     Document Upload
                            |
                            v
                  PDF / DOCX Extraction
                            |
                            v
                     Text Preprocessing
                            |
                            v
                      Text Chunking
                            |
                            v
                  DeBERTa-v3-small
                            |
                            v
                    AI Probability
                            |
                            v
                    Chunk Aggregation
                            |
                            v
                   Document AI Score
                            |
                 +----------+----------+
                 |                     |
                 v                     v
            PaperLab UI           API Response
```

The detector does not need a generative LLM during inference.

---

# 4. Model

## Primary V1 Model

**DeBERTa-v3-small**

The model is used as a sequence classification model.

```text
Input Text
    |
Tokenizer
    |
DeBERTa-v3-small
    |
Classification Head
    |
+-------------------+
|                   |
Human              AI
```

The classifier outputs probabilities rather than a hard yes/no answer.

Example:

```json
{
  "human_probability": 0.18,
  "ai_probability": 0.82
}
```

---

# 5. Why DeBERTa-v3-small?

V1 prioritizes:

1. Low inference cost
2. Reasonable model size
3. Strong NLP representation
4. Hugging Face compatibility
5. PyTorch support
6. CPU inference possibility
7. Easy experimentation

A larger model may eventually provide better performance, but V1 should establish a strong baseline before increasing infrastructure requirements.

Potential future models:

```text
DeBERTa-v3-small
        |
        v
DeBERTa-v3-base
        |
        v
Larger encoder / ensemble
```

Model selection should ultimately be based on:

> Detection quality per dollar of infrastructure

rather than parameter count.

---

# 6. Dataset Strategy

AI-text detection is highly dependent on the training and evaluation data.

V1 should not depend on a single dataset.

The planned dataset sources are:

### RAID

Large-scale AI-generated text dataset containing multiple generators, domains, and adversarial transformations.

Repository:

https://github.com/liamdugan/raid

RAID should be sampled rather than blindly downloading and training on the entire dataset.

---

### CCKS26-AIGC

A recent AI-generated text detection dataset containing human, machine-generated, and human-machine collaborative examples.

Repository:

https://github.com/ASCII-LAB/CCKS26-Task6-LLM-Generated-Text-Detect-Trace

This is particularly useful for modern-generation examples and AI-assisted writing.

---

### TXD-22

A multi-class text dataset containing human, AI-generated and mixed-author text across multiple AI systems.

Source:

https://data.mendeley.com/datasets/prcjcggtjf/1

TXD-22 can help broaden generator and writing-style coverage.

---

### PaperLab Academic Dataset

PaperLab should eventually maintain its own dataset focused specifically on academic writing.

Potential categories:

```text
Human
├── Essays
├── Research papers
├── Technical reports
├── Lab reports
├── Literature reviews
└── Theses

AI
├── AI-generated essays
├── AI-generated research-style text
├── AI-generated reports
└── AI-generated technical writing

AI-Assisted
├── Grammar correction
├── Rewriting
├── Expansion
├── Summarization
└── Paraphrasing
```

Any dataset incorporated into a commercial PaperLab product must be reviewed for its license and permitted use.

---

# 7. Dataset Labels

V1 uses binary classification:

```text
0 = HUMAN
1 = AI
```

Example:

```json
{
  "text": "This study investigates...",
  "label": 1
}
```

Metadata should be preserved whenever available.

Recommended schema:

```json
{
  "id": "unique-id",
  "text": "document text",
  "label": 1,
  "source": "raid",
  "generator": "model-name",
  "domain": "academic",
  "language": "en"
}
```

Metadata is important for evaluation even when it is not directly passed to the model.

---

# 8. Train / Validation / Test Strategy

A random 80/20 split is not sufficient for AI detection.

The detector must be tested against conditions it has not seen during training.

Example:

```text
TRAIN
-------------------------
Model A
Model B
Model C

VALIDATION
-------------------------
Different prompts
Different documents

TEST
-------------------------
Unseen generator
Unseen prompts
Unseen domains
AI paraphrasing
AI rewriting
Human editing
```

This is called a **generator-disjoint / condition-disjoint evaluation strategy**.

The goal is to measure generalization rather than memorization.

---

# 9. Recommended Project Structure

```text
paperlab-detector/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   └── v1.yaml
│
├── data/
│   ├── raw/
│   │   ├── raid/
│   │   ├── ccks/
│   │   └── txd/
│   │
│   ├── processed/
│   │
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── checkpoints/
│   └── paperlab-detector-v1/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── clean.py
│   │   ├── normalize.py
│   │   ├── chunk.py
│   │   └── split.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── tokenizer.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── predict.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_inference.py
│
└── notebooks/
    ├── dataset_analysis.ipynb
    └── model_evaluation.ipynb
```

---

# 10. Environment

Recommended:

```text
Python 3.11+
PyTorch
Hugging Face Transformers
Hugging Face Datasets
Accelerate
scikit-learn
pandas
numpy
sentencepiece
```

Optional:

```text
PEFT
bitsandbytes
evaluate
Weights & Biases
```

---

# 11. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify PyTorch:

```bash
python -c "import torch; print(torch.__version__)"
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this returns:

```text
True
```

PyTorch can access the NVIDIA GPU.

---

# 12. Example requirements.txt

```text
torch
transformers
datasets
accelerate
scikit-learn
pandas
numpy
sentencepiece
tqdm
pyyaml
```

Exact versions should be pinned once the V1 environment has been validated.

---

# 13. Data Processing Pipeline

Raw datasets should never be fed directly into training.

The pipeline should be:

```text
Raw Dataset
     |
     v
Load
     |
     v
Normalize
     |
     v
Remove invalid records
     |
     v
Remove duplicates
     |
     v
Validate labels
     |
     v
Filter extremely short text
     |
     v
Split
     |
     v
Chunk
     |
     v
Training Dataset
```

---

# 14. Text Cleaning

Basic cleaning may include:

* Removing empty samples
* Removing duplicate text
* Normalizing whitespace
* Removing malformed records
* Validating encoding
* Removing unusably short samples
* Detecting language where required

Do **not** aggressively normalize punctuation or grammar.

Those characteristics may contain useful detection signals.

---

# 15. Chunking

Academic documents can contain thousands of tokens.

DeBERTa-v3-small should therefore process chunks rather than an entire paper.

Example:

```text
Paper
 |
 +-- Chunk 1
 +-- Chunk 2
 +-- Chunk 3
 +-- Chunk 4
 +-- ...
```

Initial configuration:

```text
Maximum sequence length: 512 tokens
```

The exact chunking strategy should be evaluated.

Possible approaches:

* Sentence-based chunks
* Paragraph-based chunks
* Sliding token windows
* Paragraph + token fallback

For V1, paragraph-based chunking with a token-length fallback is a practical starting point.

---

# 16. Training Configuration

Initial experiment:

```yaml
model:
  name: microsoft/deberta-v3-small
  num_labels: 2

training:
  learning_rate: 2.0e-5
  epochs: 3
  max_length: 512
  batch_size: 16
  weight_decay: 0.01

optimization:
  fp16: true

output:
  directory: models/paperlab-detector-v1
```

These are starting values.

They should not be treated as final hyperparameters.

---

# 17. Training

Conceptually:

```bash
python scripts/prepare_data.py
```

Then:

```bash
python scripts/train.py
```

Training should produce:

```text
models/
└── paperlab-detector-v1/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

# 18. Evaluation

Run:

```bash
python scripts/evaluate.py
```

The evaluation should report at least:

```text
Accuracy
Precision
Recall
F1
AUROC
AUPRC
False Positive Rate
False Negative Rate
```

Example:

```text
==============================
PaperLab Detector V1
==============================

Accuracy:        XX.XX%
Precision:       XX.XX%
Recall:          XX.XX%
F1:              XX.XX%
AUROC:           XX.XX%
False Positive:  XX.XX%
```

The actual values should only be reported after testing.

---

# 19. False Positives

False positives are a major concern for an academic product.

A legitimate human-written paper incorrectly classified as AI-generated can cause significant harm.

Therefore, V1 should track:

```text
False Positive Rate
```

separately for:

* Academic writing
* Technical writing
* General writing
* Different authors
* Different domains
* Different lengths

Do not optimize solely for overall accuracy.

---

# 20. Document-Level Prediction

The model operates on chunks.

PaperLab needs a document-level score.

Example:

```text
Chunk 1 → 0.12
Chunk 2 → 0.21
Chunk 3 → 0.91
Chunk 4 → 0.83
Chunk 5 → 0.17
```

These scores can be aggregated into a document-level estimate.

Possible aggregation features:

```text
Mean
Median
Percentile
Fraction above threshold
Weighted mean
```

V1 should experiment with multiple aggregation strategies using the validation set.

---

# 21. Inference Output

The detector should return structured data.

Example:

```json
{
  "document_score": 0.74,
  "classification": "ai_like",
  "chunks": [
    {
      "index": 0,
      "score": 0.12
    },
    {
      "index": 1,
      "score": 0.88
    }
  ]
}
```

The API should preserve the raw probability.

Avoid hard-coding a statement such as:

```text
AI = TRUE
```

because detection is probabilistic.

---

# 22. Confidence Categories

A future PaperLab UI may expose categories such as:

```text
Low AI likelihood
Uncertain
High AI likelihood
```

Thresholds must be selected using validation data.

Do not arbitrarily assume:

```text
0–30 = human
30–70 = uncertain
70–100 = AI
```

without measuring the corresponding false-positive and false-negative rates.

---

# 23. Production Cost Strategy

PaperLab should not require a GPU server running continuously.

Training:

```text
Temporary GPU
      |
      v
Train model
      |
      v
Save model
      |
      v
Shutdown GPU
```

Production:

```text
PaperLab API
      |
      v
CPU inference worker
      |
      v
DeBERTa detector
```

If CPU performance becomes insufficient, a small GPU worker can be introduced later.

---

# 24. Quantization

After V1 accuracy is established, investigate:

```text
FP32
  ↓
FP16
  ↓
INT8
```

The goal is to reduce:

* RAM usage
* inference latency
* CPU usage
* hosting cost

Quantization should only be adopted after measuring whether it materially affects detection quality.

---

# 25. Evaluation Against Unseen Models

This is one of the most important parts of the project.

The detector should be evaluated against:

```text
Known generator
      +
Unseen generator
      +
Unseen prompts
      +
Different domains
      +
AI paraphrasing
      +
Human editing
```

Example:

```text
Training:
GPT-A
Claude-A
Llama-A

Testing:
GPT-B
Claude-B
New generator
Human-edited AI
```

A model that performs well only on generators it has seen during training is not sufficient.

---

# 26. Adversarial Evaluation

AI text can be modified.

Test cases should include:

```text
Original AI text
      |
      +-- Paraphrased
      |
      +-- Rewritten
      |
      +-- Grammar corrected
      |
      +-- Human edited
      |
      +-- Sentence reordered
```

This should be part of the evaluation pipeline.

---

# 27. Data Leakage Prevention

The following must be checked:

* Duplicate documents
* Near-duplicate documents
* Same prompt in train/test
* Same generated output in train/test
* Same source document in train/test
* Generator leakage
* Metadata leakage

A model can appear extremely accurate if train/test contamination exists.

---

# 28. Reproducibility

Every experiment should record:

```text
Model version
Dataset version
Dataset hash
Training configuration
Random seed
Python version
PyTorch version
Transformers version
GPU
Training duration
Evaluation results
```

Recommended:

```text
experiments/
└── 2026-XX-XX_v1/
    ├── config.yaml
    ├── metrics.json
    ├── dataset_info.json
    └── notes.md
```

---

# 29. Model Versioning

Use explicit versions:

```text
paperlab-detector-v1.0
paperlab-detector-v1.1
paperlab-detector-v2.0
```

Example:

```text
v1.0
Binary human/AI detector

v1.1
Improved dataset

v1.2
Improved unseen-model generalization

v2.0
Human / AI / AI-assisted / Mixed
```

---

# 30. V1 Success Criteria

V1 should not be considered successful merely because it reaches high accuracy on a random test split.

A successful V1 should demonstrate:

* Strong performance on held-out data.
* Low false-positive rate on human academic writing.
* Reasonable performance on unseen generators.
* Reasonable performance after AI paraphrasing.
* Reproducible training.
* Affordable inference.
* Model small enough for practical deployment.

The exact target metrics should be established after the first baseline experiment.

---

# 31. Future Versions

## V1

```text
Human vs AI
DeBERTa-v3-small
```

## V2

```text
Human
AI
AI-assisted
Mixed
```

## V3

```text
Sentence-level detection
Paragraph-level evidence
```

## V4

```text
Generator/source attribution
```

## V5

```text
Multilingual detection
```

## V6

```text
Continuous evaluation
Continuous dataset updates
Model refresh pipeline
```

---

# 32. Important Limitations

AI-text detection is inherently difficult.

A detector cannot guarantee that:

> "This text was definitely written by AI."

It is possible for:

* Human text to be classified as AI-like.
* AI text to be classified as human-like.
* New generators to behave differently from training data.
* Human editing to change detector performance.
* Translation or paraphrasing to change detector performance.

Therefore PaperLab should treat the detector as an **analytical signal**, not definitive proof of authorship.

---

# 33. Recommended V1 Development Order

```text
1. Set up Python/PyTorch
        ↓
2. Download selected datasets
        ↓
3. Inspect dataset distributions
        ↓
4. Normalize datasets
        ↓
5. Remove duplicates
        ↓
6. Design generator-disjoint split
        ↓
7. Build baseline classifier
        ↓
8. Train DeBERTa-v3-small
        ↓
9. Evaluate
        ↓
10. Analyze false positives
        ↓
11. Test unseen generators
        ↓
12. Test paraphrased/edited text
        ↓
13. Optimize model
        ↓
14. Export V1
        ↓
15. Integrate into PaperLab
```

---

# 34. Final V1 Architecture

```text
                    ┌─────────────────────┐
                    │      PaperLab       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  PDF/DOCX Parser    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Chunker      │
                    │     512 tokens      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DeBERTa-v3-small    │
                    │                     │
                    │ PyTorch             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Chunk probabilities │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Score Aggregator    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Document AI Score   │
                    └─────────────────────┘
```

---

# 35. V1 Principle

The most important principle for PaperLab V1 is:

> **Optimize for generalization and low false-positive rates, not an impressive accuracy number on an easy test set.**

A detector that achieves 99% accuracy on contaminated or unrealistic test data is less useful than one that performs honestly and consistently on unseen modern AI-generated text.

The first milestone is therefore not:

**"Build the world's best AI detector."**

It is:

**"Build a reproducible, inexpensive baseline and determine exactly where it works and where it fails."**

That baseline will give PaperLab a solid foundation for subsequent detector versions.
