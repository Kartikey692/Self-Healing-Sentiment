# Self-Healing Sentiment Classification DAG

## Project Overview

This project demonstrates a self-healing text classification pipeline for sentiment analysis, designed for reliability in human-in-the-loop workflows. It leverages:

- **DistilBERT**: A lightweight transformer model for text classification.
- **LoRA**: Efficient fine-tuning to minimize resource usage.
- **LangGraph DAG**: A Directed Acyclic Graph workflow that manages inference, confidence checking, and fallback logic.
- **CLI Interface**: For interactive classification, clarification, and logging.

## Model & Dataset Details

- **Model**: distilbert-base-uncased
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation) via the peft library.
- **Dataset**: IMDB Movie Reviews
- **Task**: Sentiment analysis (binary classification: Positive/Negative)
- **Sample Size**: 200 random samples for quick demonstration

## Fine-Tuning Method

LoRA injects small, trainable adapter layers into the transformer model, enabling efficient fine-tuning with minimal computation and memory overhead.

Only the LoRA adapters are trained, while the base model weights remain frozen. This approach is ideal for resource-constrained environments and rapid experimentation.

## Project Structure

```
.
├── .env                   # Hugging Face token (not committed)
├── finetune_lora.py       # Fine-tune DistilBERT with LoRA adapters
├── model.py               # Model loading and inference
├── dag.py                 # LangGraph DAG node logic
├── cli.py                 # Command-line interface
├── logger.py              # Structured logging
├── run.log                # Generated log file (after running)
└── README.md              # This file
```

## Workflow & Node Descriptions

The workflow is implemented as a LangGraph DAG with the following nodes:

- **InferenceNode**: Runs the fine-tuned model on user input and returns the predicted label and confidence score.
- **ConfidenceCheckNode**: Checks if the model’s confidence is above a defined threshold (default: 70%). If not, triggers the fallback node.
- **FallbackNode**: Prevents unreliable predictions by either:
  - Asking the user for clarification via the CLI, or
  - (Extensible) Escalating to a backup model or alternative strategy.

## File-by-File Explanation

| File              | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `.env`            | Stores your Hugging Face API token (for secure, authenticated model download) |
| `finetune_lora.py`| Fine-tunes DistilBERT on IMDB with LoRA and saves the model to `lora-distilbert-imdb/` |
| `model.py`        | Loads the fine-tuned model and provides a `predict()` function          |
| `dag.py`          | Implements the DAG logic: inference, confidence check, fallback, and final decision |
| `cli.py`          | Runs the CLI loop for user input, triggers the DAG, and displays results |
| `logger.py`       | Logs all predictions, confidence scores, fallback events, and final decisions |
| `run.log`         | Auto-generated log file with timestamps (after running the CLI)         |
| `README.md`       | Project documentation                                                   |

## Setup & Installation

### Clone the repository

```
git clone https://github.com/Kartikey692/Self-Healing-Sentiment
cd self-healing-classification-dag
```

### Install dependencies

```
pip install torch transformers datasets peft langgraph langchain python-dotenv
```

### Create a `.env` file

```
HF_TOKEN=your_actual_huggingface_token_here
```


## Running Fine-Tuning

Fine-tune the model (only required once, or if you want to retrain):

```
python finetune_lora.py
```

- Downloads the dataset and model
- Fine-tunes with LoRA
- Saves the result to `lora-distilbert-imdb/`

## Running the CLI & Interacting with the DAG

Launch the CLI to classify text and interact with the DAG:

```
python cli.py
```

### Workflow

1. **InferenceNode**: Predict label and confidence.
2. **ConfidenceCheckNode**: If confidence < threshold, trigger fallback.
3. **FallbackNode**: Ask for user clarification.
4. **Final Decision**: Show label (corrected if clarified).

### CLI Flow Example

```
Loading fine-tuned model...
Ready! Type your review (or 'exit' to quit):

Input: The movie was painfully slow and boring.
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: Yes
Final Label: Negative (Confidence: 54.3%) [Fallback used]

Input: I loved the cinematography and story!
Final Label: Positive (Confidence: 92.1%)

Input: exit
```

## Logging

All actions are logged in `run.log` with timestamps:

- Initial predictions and confidence scores
- Fallback activations and user clarifications
- Final classification decisions
