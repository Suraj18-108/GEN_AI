# GPT-2 Interactive Story Generator 📖✨

An end-to-end deep learning web application powered by **GPT-2** and **Hugging Face Transformers**, featuring an interactive **Gradio** interface. Users can input an opening sentence, select a narrative writing style, tune generation randomness via the softmax temperature slider, inspect precise token counts, and conduct side-by-side temperature comparisons between low-entropy (T=0.5) and high-entropy (T=1.2) outputs.

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Project Folder Structure](#project-folder-structure)
5. [Environment Setup & Installation (Windows)](#environment-setup--installation-windows)
6. [How to Run the Application](#how-to-run-the-application)
7. [How to Use the Application](#how-to-use-the-application)
8. [Temperature Comparison Feature](#temperature-comparison-feature)
9. [Technical Deep Dive](#technical-deep-dive)
   - [GPT-2 Architecture & Hugging Face Transformers](#gpt-2-architecture--hugging-face-transformers)
   - [Tokenization & Accurate Token Accounting](#tokenization--accurate-token-accounting)
   - [Temperature, Logits, and Stochastic Sampling](#temperature-logits-and-stochastic-sampling)
10. [Limitations of GPT-2](#limitations-of-gpt-2)
11. [Troubleshooting & FAQs](#troubleshooting--faqs)
12. [Automated Testing](#automated-testing)

---

## 1. Project Overview

The **GPT-2 Interactive Story Generator** demonstrates practical causal language modeling applied to interactive creative writing. Rather than relying on closed-source, paid commercial APIs, this application runs a local, open-weights GPT-2 model (`openai-community/gpt2`) natively via PyTorch.

Users provide a premise or opening sentence, select one of 10 narrative genres (e.g., Fantasy, Mystery, Science Fiction, Horror), and adjust the sampling temperature. The model completes the story naturally while preserving the user's opening text, reporting exact prompt token counts, newly generated token counts, and total sequence length.

---

## 2. Key Features

- **Adaptive Prompt Construction:** Dynamically adapts narrative cues for 10 distinct genres without exposing internal system instructions to the reader.
- **Preserved Opening Sentence:** Seamlessly merges the user's prompt with generated continuations, handling punctuation and paragraph breaks cleanly.
- **Configurable Generation Length (Max New Tokens):** Live slider allowing users to specify the exact target token limit (20 to 300 tokens, default 150).
- **Dynamic Randomness Control:** Interactive slider from 0.2 to 1.5 (default: 0.8) adjusting the softmax temperature in real time.
- **Rigorous Token Auditing:** Measures exact input prompt tokens vs. newly generated tokens using tokenizer tensor shapes, avoiding estimation errors.
- **Dynamic Reading Font Customizer:** Switch typography across story panels in real-time (Classic Serif, Modern Sans, Typewriter Monospace, Literary Novel, or Cursive Script).
- **Customizable Temperature Comparison Panel:** Simultaneously generates two stories with user-selectable temperatures (Temperature A vs. Temperature B, defaults: 0.5 and 1.2) and configurable token lengths, displaying them side-by-side with an evidence-based comparative summary and lexical diversity metrics.
- **Modern Responsive UI:** Dark-navy / charcoal theme with purple/blue accents, metric chips, loading animations, 1-click clipboard copying, and workspace reset.
- **Performance Optimized:** Thread-safe singleton model loader, GPU (CUDA) auto-detection with graceful CPU fallback, and PyTorch inference mode.

---

## 3. Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language |
| **User Interface** | Gradio | Modern, reactive web application UI |
| **Language Model** | GPT-2 (`openai-community/gpt2`) | 124M parameter autoregressive causal LM |
| **Deep Learning** | PyTorch | Tensor computation and neural execution |
| **NLP Framework** | Hugging Face Transformers | Model loading, generation pipeline, and tokenization |
| **Operating System** | Windows 10/11 Compatible | Cross-platform compatibility |

---

## 4. Project Folder Structure

```
gpt2-story-generator/
│
├── app.py                   # Gradio interface, event handlers, and responsive layout
├── model.py                 # Singleton GPT-2 engine, prompt formatting, token math & comparison
├── requirements.txt         # Core dependencies (torch, transformers, gradio)
├── test_story_generator.py  # Unit tests for validation, prompt construction, formatting, & metrics
├── .gitignore               # Excludes caches, venv, and checkpoints
└── README.md                # Comprehensive documentation and technical guide
```

---

## 5. Environment Setup & Installation (Windows)

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on Windows.
- Git (optional, for cloning).

### Step 1: Open Terminal / PowerShell
Open PowerShell or Command Prompt and navigate to the project directory:
```powershell
cd "C:\Users\SHANTHI GURJAR\.gemini\antigravity\scratch\gpt2-story-generator"
```

### Step 2: Create a Virtual Environment
Isolate dependencies using Python's `venv` module:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
# In PowerShell:
.\venv\Scripts\Activate.ps1

# Or in Command Prompt (cmd.exe):
.\venv\Scripts\activate.bat
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

> **Note on PyTorch Hardware Acceleration:**  
> If you have an NVIDIA GPU and wish to enable CUDA acceleration, install the CUDA-enabled PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> If CUDA is not detected, the application automatically defaults to CPU inference.

---

## 6. How to Run the Application

Launch the application using:
```powershell
python app.py
```

Upon launching:
1. The application initializes the singleton `GPT2StoryEngine`.
2. The GPT-2 tokenizer and model are loaded into memory (downloaded on the first run, cached thereafter).
3. The local web server starts at:
   ```
   * Running on local URL:  http://127.0.0.1:7860
   ```
4. Open your browser and navigate to `http://127.0.0.1:7860`.

---

## 7. How to Use the Application

1. **Enter an Opening Sentence:** Type or paste your story starter into the "Opening Sentence" text box (e.g., *"The old clock tower began to chime backwards."*).
2. **Select a Writing Style:** Choose from Fantasy, Mystery, Science Fiction, Horror, Adventure, Romance, Comedy, Thriller, Fairy Tale, or Historical Fiction.
3. **Set the Temperature & Max New Tokens:**
   - **Temperature:** 0.2 – 0.6 (focused/predictable), 0.7 – 0.9 (balanced default), 1.0 – 1.5 (creative/varied).
   - **Max New Tokens:** Adjust the token slider between 20 and 300 (default: 150) to control length.
4. **Click "Generate Story":** Watch the story continue smoothly.
5. **Inspect Token Counts:** View the metric chips showing Newly Generated Tokens, Prompt Tokens, Total Sequence Length, and Compute Device.
6. **Copy or Reset:** Use the built-in copy button or reset button as needed.

---

## 8. Temperature Comparison Feature

The **Temperature Comparison** section allows direct empirical exploration of how different temperature parameters impact generative text:
- **Select Temperature A & B:** Adjust the sliders to any desired values (default: 0.5 for A, 1.2 for B).
- **Select Comparison Token Length:** Specify the max new tokens to generate for both runs (default: 150).
- Click **"⚡ Run Temperature Comparison"**.
- The model generates two independent continuations from your exact sentence and genre using the chosen settings.
- A side-by-side comparison displays both stories, their individual token counts, and an evidence-based analysis:
  - **Lexical Diversity (Type-Token Ratio):** Ratio of unique words to total words.
  - **Syntactic Predictability vs. Divergence:** Comparative breakdown of probability sharpening vs. flattening.
  - **Probabilistic Tendencies Caveat:** Highlights that temperature changes probability distributions, meaning individual samples reflect tendencies rather than rigid guarantees.

---

## 9. Technical Deep Dive

### GPT-2 Architecture & Hugging Face Transformers
GPT-2 (Radford et al., 2019) is an autoregressive causal language model based on the Transformer decoder architecture:
- Employs masked multi-head self-attention to ensure tokens can only attend to preceding positions.
- Uses learned positional embeddings up to a maximum context window of 1,024 tokens.
- Hugging Face Transformers provides the `AutoModelForCausalLM` and `AutoTokenizer` interfaces, handling weight loading, attention masks, and past key-value caching during generation.

### Tokenization & Accurate Token Accounting
GPT-2 uses **Byte-Pair Encoding (BPE)** with a vocabulary of 50,257 tokens:
- Words are broken into subword units (e.g., "unpredictable" $\rightarrow$ `['un', 'predict', 'able']`).
- **Prompt Token Count ($N_{prompt}$):** Number of tokens produced by tokenizing the prompt string.
- **Total Sequence Tokens ($N_{total}$):** Total tokens in the output tensor.
- **Newly Generated Tokens ($N_{new}$):** Calculated strictly as:
  $$N_{new} = N_{total} - N_{prompt}$$
  This isolates generated length from input prompt size.

### Temperature, Logits, and Stochastic Sampling
Given the unnormalized log-probabilities (logits) $z_i$ output by the language model for token $i$, the softmax probability $P(w_i)$ with temperature $T$ is given by:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **When $T \to 0$:** The distribution approaches an argmax (greedy decoding). High-probability tokens dominate, reducing randomness and vocabulary variance.
- **When $T = 1.0$:** Standard multinomial sampling according to original training probabilities.
- **When $T > 1.0$:** Logits are scaled down, flattening the probability distribution (higher entropy). Less common tokens have a significantly higher chance of selection.

---

## 10. Limitations of GPT-2

Users and evaluators should keep in mind the inherent characteristics of base GPT-2 (124M):
1. **Base Model vs. Instruction Tuning:** GPT-2 was pretrained on raw WebText rather than fine-tuned on instruction datasets (unlike ChatGPT or InstructGPT). Prompt framing guides thematic continuation, but strict instruction following cannot be guaranteed.
2. **Occasional Repetition:** Base language models can fall into repetitive phrase loops ("and then he saw, and then he saw..."). The application includes a mild repetition penalty ($1.1$) to reduce looping while preserving natural prose.
3. **Context Window Constraint:** The maximum context length is 1,024 tokens.
4. **Factual Hallucinations:** As a creative generative model, GPT-2 synthesizes plausible-sounding text without grounding in factual databases.

---

## 11. Troubleshooting & FAQs

### Q: "Failed to download model" or Network Timeout
**A:** Ensure your machine has an active internet connection on the first run so Hugging Face can download the model weights (~500 MB). If working behind a corporate firewall or proxy:
```powershell
$env:HF_HUB_ENABLE_HF_TRANSFER="0"
```
Weights are permanently cached in `~/.cache/huggingface/hub`. Once downloaded, the application runs entirely offline.

### Q: "Out of Memory" (OOM) on GPU
**A:** If your GPU has limited VRAM, GPT-2 easily runs on CPU. Force CPU mode by setting:
```powershell
$env:CUDA_VISIBLE_DEVICES=""
```

### Q: Port 7860 already in use
**A:** In `app.py`, change `demo.launch(server_port=7861)` or terminate the process occupying port 7860.

---

## 12. Automated Testing

Run the included unit test suite to verify input validation, prompt construction, formatting, lexical diversity calculations, and token accounting:

```powershell
python -m unittest test_story_generator.py -v
```

All tests execute cleanly without requiring GPU hardware or network access.

---

## 13. Deploying to Vercel (Free Serverless Tier)

The project includes pre-configured Vercel serverless deployment assets:
- `vercel.json`: Routing configuration.
- `public/index.html`: Responsive web frontend with font styling and temperature comparison.
- `api/generate.js`: Serverless API connecting to Hugging Face's free GPT-2 inference.

### Deploying via Vercel CLI (Quickest)
Run the following in PowerShell:
```powershell
npx vercel
```
1. Follow the browser prompt to log in to Vercel (free account).
2. Accept the defaults (`Set up and deploy? [Y]`).
3. Your live production URL will be ready in under 60 seconds!

### Deploying via GitHub to Vercel
1. Create a new GitHub repository and push your local commits:
   ```powershell
   git remote add origin https://github.com/<your-username>/gpt2-story-generator.git
   git branch -M main
   git push -u origin main
   ```
2. Go to [vercel.com/new](https://vercel.com/new) and click **Import** next to your repository.
3. Click **Deploy**. Vercel will build and host the application for free.
