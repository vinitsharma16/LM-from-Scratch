# GPT-2 from Scratch: Pretraining, Fine-Tuning and Instruction Tuning

**DA 624 – NLP with LLMs · Project by Ekansh Aggarwal (264161002)**
MTech Data Science, IIT Guwahati · Instructor: Dr. Chiranjib Sur

In this project I implemented the GPT-2 architecture from scratch in PyTorch and took it through the full LLM pipeline: tokenization, pretraining, loading OpenAI's pretrained weights, classification fine-tuning, instruction fine-tuning and evaluation. I then extended it with LoRA (written from scratch) to instruction-tune GPT-2 large on the Stanford Alpaca dataset on a single free Kaggle GPU, and built a chat interface for the final model.

## What's inside
- **Tokenization:** byte-pair encoding with tiktoken, a sliding-window data loader, token and positional embeddings
- **Model:** causal multi-head self-attention, layer norm, GELU feed-forward, shortcut connections and transformer blocks, assembled into GPT-2 (124M parameters)
- **Pretraining:** next-token prediction with cross-entropy loss, plus text generation with temperature and top-k sampling
- **Pretrained weights:** loading OpenAI's GPT-2 weights into my own implementation
- **Spam classification:** fine-tuned GPT-2 to classify SMS spam (97.21% train, 97.32% validation, 95.67% test accuracy)
- **Instruction tuning:** GPT-2 medium (355M) fine-tuned on instruction–response pairs in the Alpaca format
- **Extension (LoRA + Alpaca):** GPT-2 large (774M) tuned on 46,436 Alpaca-cleaned examples with LoRA (r=16, α=32). Only 1.6% of the weights are trained, using fp16 mixed precision and gradient checkpointing.
- **Evaluation:** Llama 3 8B (run with Ollama) scores each answer from 0 to 100 against the reference answer
- **Chat UI:** a Gradio interface with a repetition penalty and n-gram blocking

## Results
| Model | Llama 3 judge score (110 test instructions) |
|---|---|
| GPT-2 medium, full fine-tuning (~1.1k examples) | 48.92 / 100 |
| GPT-2 large + LoRA (46k Alpaca examples) | **57.19 / 100** |

## How to run
All notebooks run on Kaggle (GPU T4, Internet on). Run the cells top to bottom. Step-by-step instructions are in the project report (Appendix A).

## Acknowledgements
The core implementation follows Sebastian Raschka's *Build a Large Language Model (From Scratch)* and Vizuara's "Build LLM from Scratch" lecture series. The LoRA extension, Alpaca training, evaluation and chat UI were built on top of it.
