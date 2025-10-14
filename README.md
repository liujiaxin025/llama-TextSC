# Llama3-TextSC: Llama 3-based Textual Semantic Communication System

A semantic communication system that uses Llama 3 for semantic understanding and generation, combined with neural channel coding for robust text transmission over noisy channels.

## System Architecture

### Transmitter
1. **Semantic Condenser** - Uses Llama 3 to extract core semantic meaning
2. **Semantic Vectorizer** - Converts semantic JSON to embeddings using Sentence-BERT
3. **Channel Encoder** - MLP-based neural encoder for channel transmission

### Channel
- **AWGN Channel** - Additive White Gaussian Noise channel simulation

### Receiver
1. **Channel Decoder** - MLP-based neural decoder for noise recovery
2. **Semantic De-Vectorizer** - Converts vectors back to semantic text
3. **Semantic Generator** - Uses Llama 3 to reconstruct full text from semantics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.core.semantic_comm_system import SemanticCommSystem

# Initialize the system
system = SemanticCommSystem()

# Transmit text through noisy channel
original_text = "Your message here"
recovered_text = system.transmit(original_text, snr_db=10)
```

## Training

```bash
python src/training/train.py
```

## Evaluation

```bash
python src/evaluation/evaluate.py
```