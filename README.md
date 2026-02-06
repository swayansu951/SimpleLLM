# TinyLlama Customer Support Chatbot (LoRA Fine-Tuning on CPU)

This project fine-tunes **TinyLlama/TinyLlama-1.1B-Chat-v1.0** using
**LoRA (PEFT)** on a custom conversational dataset and provides an
interactive command-line chatbot for customer--agent style
conversations.

The training is done using **TRL SFTTrainer** and runs fully on **CPU**.

------------------------------------------------------------------------

## Project Overview

-   Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0\
-   Fine-tuning method: LoRA (PEFT)\
-   Training framework: TRL (Supervised Fine-Tuning)\
-   Inference: HuggingFace pipeline\
-   Hardware: CPU only

------------------------------------------------------------------------

## Folder Structure

    .
    ├── train.py
    ├── chat.py
    ├── requirements.txt
    ├── Conversational_Transcript_Dataset.json
    └── hackathon/
        ├── outputs/
        └── my_model_trained/

------------------------------------------------------------------------

## Dataset Format

The training file must be named:

    Conversational_Transcript_Dataset.json

Format:

``` json
{
  "transcripts": [
    {
      "domain": "...",
      "intent": "...",
      "reason_for_call": "...",
      "conversation": [
        { "speaker": "Customer", "text": "..." },
        { "speaker": "Agent", "text": "..." }
      ]
    }
  ]
}
```

------------------------------------------------------------------------

## Installation

    pip install -r requirements.txt

### requirements.txt

    transformers
    datasets
    trl
    peft
    torch
    sentence_transformers
    difflib

------------------------------------------------------------------------

## Training

    python train.py

The trained LoRA adapter will be saved to:

    hackathon/my_model_trained

------------------------------------------------------------------------

## Training Configuration

-   Training type: Supervised Fine-Tuning (SFT)
-   Device: CPU
-   Precision: FP32
-   Epochs: 1
-   Batch size: 1
-   Max sequence length: 128
-   Learning rate: 2e-5
-   LoRA target modules:
    -   q_proj
    -   v_proj

------------------------------------------------------------------------

## Run the Chatbot

    python chat.py

------------------------------------------------------------------------

## Prompt Format

    ### conversation
    Customer: ...
    Agent: ...

------------------------------------------------------------------------

## Metadata-aware Chat

The chatbot finds the closest transcript using **difflib** and
automatically selects:

-   domain
-   intent
-   reason_for_call

to build better prompts.

------------------------------------------------------------------------

## Output

    hackathon/my_model_trained

------------------------------------------------------------------------

## Important Notes

Only LoRA adapters are saved.\
The base model is always loaded from HuggingFace.

------------------------------------------------------------------------

## Hardware

CPU only (8 GB RAM recommended)

------------------------------------------------------------------------

## Limitations

-   Small model (1.1B parameters)
-   Short training (1 epoch)
-   No automatic evaluation
-   Short context length

------------------------------------------------------------------------

## License

Same as **TinyLlama/TinyLlama-1.1B-Chat-v1.0**.

------------------------------------------------------------------------

## Author

**Swayansu Sahu**
