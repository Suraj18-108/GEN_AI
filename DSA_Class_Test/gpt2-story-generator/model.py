"""
model.py: GPT-2 Model Management, Text Generation, and Temperature Comparison.

This module provides:
- Thread-safe singleton model and tokenizer loading (CPU/CUDA auto-selection).
- Prompt construction adapted to narrative styles.
- Story generation with Hugging Face Transformers.
- Accurate token counting (prompt, newly generated, total).
- Controlled temperature comparison between T=0.5 and T=1.2.
- In-depth, evidence-based comparative summary generation.
"""

import logging
import re
import threading
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure module logger
logger = logging.getLogger("gpt2_story_engine")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)

# Supported writing styles
SUPPORTED_STYLES = [
    "Fantasy",
    "Mystery",
    "Science Fiction",
    "Horror",
    "Adventure",
    "Romance",
    "Comedy",
    "Thriller",
    "Fairy Tale",
    "Historical Fiction",
]

# Style prompts designed for base GPT-2 instruction/framing
STYLE_PROMPTS: Dict[str, str] = {
    "Fantasy": "Write an epic fantasy story filled with magic, mythical creatures, and ancient mystery.",
    "Mystery": "Write a suspenseful noir mystery story steeped in secrets, investigation, and tension.",
    "Science Fiction": "Write a visionary science fiction story exploring future technology and space wonders.",
    "Horror": "Write a chilling psychological horror story where dread and eerie darkness lurk.",
    "Adventure": "Write a thrilling high-stakes adventure story full of danger and courageous exploration.",
    "Romance": "Write an emotionally evocative romance story highlighting deep affection and connection.",
    "Comedy": "Write a lighthearted and witty comedy story filled with humorous misunderstandings.",
    "Thriller": "Write a fast-paced psychological thriller filled with urgency, twists, and danger.",
    "Fairy Tale": "Write an enchanting fairy tale with magical charm, kingdoms, and timeless wonder.",
    "Historical Fiction": "Write an authentic historical fiction story reflecting the drama of a past era.",
}


def validate_story_inputs(
    opening_sentence: str, temperature: float, max_new_tokens: int = 150
) -> Tuple[bool, Optional[str]]:
    """
    Validate user inputs before passing to the model.

    Returns:
        (is_valid, error_message)
    """
    if not opening_sentence or not opening_sentence.strip():
        return False, "Opening sentence cannot be empty. Please provide an opening line for your story."

    clean_text = opening_sentence.strip()
    if len(clean_text) < 3:
        return False, "Opening sentence is too short. Please provide a complete phrase or sentence."

    if len(clean_text) > 1200:
        return False, "Opening sentence is too long (maximum 1,200 characters). Please provide a more concise opening."

    try:
        temp_val = float(temperature)
    except (TypeError, ValueError):
        return False, "Temperature must be a valid numerical value."

    if temp_val < 0.1 or temp_val > 2.0:
        return False, "Temperature must be between 0.2 and 1.5 for stable generation."

    try:
        tokens_val = int(max_new_tokens)
        if tokens_val < 10 or tokens_val > 512:
            return False, "Max new tokens must be an integer between 10 and 512."
    except (TypeError, ValueError):
        return False, "Max new tokens must be a valid integer."

    return True, None


def construct_prompt(opening_sentence: str, style: str) -> str:
    """
    Build an effective prompt incorporating both the opening sentence and the selected style.
    Encourages narrative continuation on base GPT-2.
    """
    style_instruction = STYLE_PROMPTS.get(
        style, f"Write a {style.lower()} story."
    )
    clean_opening = opening_sentence.strip()
    return f"{style_instruction} Continue the following opening sentence naturally and maintain a consistent narrative:\n\n{clean_opening}"


def format_story_output(opening_sentence: str, continuation: str) -> str:
    """
    Combines the user's preserved opening sentence with the generated continuation,
    ensuring clean spacing and preventing duplication or disjointed punctuation.
    """
    clean_opening = opening_sentence.strip()
    clean_continuation = continuation.strip()

    if not clean_continuation:
        return clean_opening

    # Prevent base GPT-2 from repeating the prompt opening line verbatim
    if clean_continuation.lower().startswith(clean_opening.lower()):
        clean_continuation = clean_continuation[len(clean_opening):].strip()

    if not clean_continuation:
        return clean_opening

    # Ensure natural punctuation and spacing
    if clean_continuation[0] in [".", ",", "!", "?", ";", ":", "'", '"', "\n"]:
        return f"{clean_opening}{clean_continuation}"

    return f"{clean_opening} {clean_continuation}"


def calculate_lexical_diversity(text: str) -> float:
    """
    Calculate Type-Token Ratio (TTR: unique words / total words) as a proxy
    for lexical diversity in text comparison.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


class GPT2StoryEngine:
    """
    Singleton wrapper for loading GPT-2 and executing causal generation.
    Maintains a single model instance in memory across requests.
    """

    _instance: Optional["GPT2StoryEngine"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPT2StoryEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "openai-community/gpt2"):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._initialized = True
        logger.info(
            f"Initialized GPT2StoryEngine instance. Device: {self.device}"
        )

    def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Load pretrained GPT-2 tokenizer and model once into memory.
        Thread-safe; reuses existing instance if already loaded.
        """
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return True, None

        with self._load_lock:
            # Double-check pattern
            if self.is_loaded:
                return True, None

            logger.info(
                f"Starting download/load of '{self.model_name}' on {self.device}..."
            )
            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load causal LM
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.model.to(self.device)
                self.model.eval()

                self.is_loaded = True
                self.load_error = None
                logger.info("GPT-2 model and tokenizer successfully loaded.")
                return True, None
            except Exception as exc:
                self.load_error = str(exc)
                logger.error(
                    f"Failed to load GPT-2 model '{self.model_name}': {exc}",
                    exc_info=True,
                )
                return False, f"Model initialization failed: {str(exc)}"

    def generate_story(
        self,
        opening_sentence: str,
        style: str = "Fantasy",
        temperature: float = 0.8,
        max_new_tokens: int = 150,
        top_k: int = 50,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a story continuation using the loaded GPT-2 model.

        Returns a detailed dictionary containing:
        - success: bool
        - story: complete formatted story text
        - continuation: newly generated text only
        - prompt: internal prompt string passed to model
        - style: selected style
        - temperature: selected temperature
        - prompt_tokens: count of prompt tokens
        - new_tokens: count of newly generated tokens
        - total_tokens: total sequence tokens
        - device: device used (cpu or cuda)
        - error: error message if any
        """
        # Validate inputs
        is_valid, validation_err = validate_story_inputs(
            opening_sentence, temperature, max_new_tokens
        )
        if not is_valid:
            return {
                "success": False,
                "error": validation_err,
                "story": "",
                "continuation": "",
                "prompt_tokens": 0,
                "new_tokens": 0,
                "total_tokens": 0,
            }

        # Ensure model is ready
        loaded, load_err = self.load_model()
        if not loaded or self.model is None or self.tokenizer is None:
            return {
                "success": False,
                "error": f"Model is not available: {load_err}",
                "story": "",
                "continuation": "",
                "prompt_tokens": 0,
                "new_tokens": 0,
                "total_tokens": 0,
            }

        try:
            # Set optional reproducible seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Build model prompt
            prompt = construct_prompt(opening_sentence, style)

            # Tokenize input
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            prompt_token_count = input_ids.shape[1]

            # Enforce GPT-2 maximum sequence length boundary (1024 tokens)
            available_tokens = 1024 - prompt_token_count
            if available_tokens <= 10:
                return {
                    "success": False,
                    "error": "The input prompt is too long for GPT-2's 1024-token context window.",
                    "story": "",
                    "continuation": "",
                    "prompt_tokens": prompt_token_count,
                    "new_tokens": 0,
                    "total_tokens": prompt_token_count,
                }

            actual_max_new_tokens = min(max_new_tokens, available_tokens)

            # Perform inference without gradient tracking
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=actual_max_new_tokens,
                    do_sample=True,
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=0.92,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Token counting using tensor dimensions
            total_token_count = int(output_ids.shape[1])
            new_token_count = total_token_count - prompt_token_count

            # Extract continuation tokens only
            continuation_ids = output_ids[0][prompt_token_count:]
            continuation_text = self.tokenizer.decode(
                continuation_ids, skip_special_tokens=True
            )

            # Format full story preserving user's opening sentence
            formatted_story = format_story_output(
                opening_sentence, continuation_text
            )

            return {
                "success": True,
                "story": formatted_story,
                "continuation": continuation_text,
                "prompt": prompt,
                "style": style,
                "temperature": float(temperature),
                "prompt_tokens": prompt_token_count,
                "new_tokens": new_token_count,
                "total_tokens": total_token_count,
                "device": self.device,
                "error": None,
            }

        except Exception as exc:
            logger.error(f"Story generation failed: {exc}", exc_info=True)
            return {
                "success": False,
                "error": f"An error occurred during text generation: {str(exc)}",
                "story": "",
                "continuation": "",
                "prompt_tokens": 0,
                "new_tokens": 0,
                "total_tokens": 0,
            }

    def compare_temperatures(
        self,
        opening_sentence: str,
        style: str = "Fantasy",
        temp_a: float = 0.5,
        temp_b: float = 1.2,
        max_new_tokens: int = 150,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Generate two stories from identical inputs and settings with selectable temperatures:
        Output A: Temperature A (e.g. 0.5 - focused/conservative)
        Output B: Temperature B (e.g. 1.2 - creative/divergent)

        Produces an evidence-based summary detailing observed coherence, repetition,
        predictability, and lexical variation.
        """
        # Validate inputs upfront for both temperatures and token count
        is_valid_a, err_a = validate_story_inputs(
            opening_sentence, temperature=temp_a, max_new_tokens=max_new_tokens
        )
        if not is_valid_a:
            return {
                "success": False,
                "error": f"Invalid Temperature A: {err_a}",
                "output_a": None,
                "output_b": None,
                "summary": "",
            }

        is_valid_b, err_b = validate_story_inputs(
            opening_sentence, temperature=temp_b, max_new_tokens=max_new_tokens
        )
        if not is_valid_b:
            return {
                "success": False,
                "error": f"Invalid Temperature B: {err_b}",
                "output_a": None,
                "output_b": None,
                "summary": "",
            }

        logger.info(
            f"Running temperature comparison: A={temp_a}, B={temp_b}, style={style}, max_new_tokens={max_new_tokens}"
        )

        # Generate Output A
        res_a = self.generate_story(
            opening_sentence=opening_sentence,
            style=style,
            temperature=temp_a,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if not res_a["success"]:
            return {
                "success": False,
                "error": f"Failed generating Output A: {res_a.get('error')}",
                "output_a": None,
                "output_b": None,
                "summary": "",
            }

        # Generate Output B
        res_b = self.generate_story(
            opening_sentence=opening_sentence,
            style=style,
            temperature=temp_b,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if not res_b["success"]:
            return {
                "success": False,
                "error": f"Failed generating Output B: {res_b.get('error')}",
                "output_a": res_a,
                "output_b": None,
                "summary": "",
            }

        # Analyze quantitative metrics
        div_a = calculate_lexical_diversity(res_a["story"])
        div_b = calculate_lexical_diversity(res_b["story"])

        # Construct comprehensive, balanced comparison summary
        summary = self._build_comparison_summary(
            temp_a=temp_a,
            temp_b=temp_b,
            res_a=res_a,
            res_b=res_b,
            diversity_a=div_a,
            diversity_b=div_b,
            seed=seed,
        )

        return {
            "success": True,
            "output_a": res_a,
            "output_b": res_b,
            "summary": summary,
            "error": None,
        }

    def _build_comparison_summary(
        self,
        temp_a: float,
        temp_b: float,
        res_a: Dict[str, Any],
        res_b: Dict[str, Any],
        diversity_a: float,
        diversity_b: float,
        seed: Optional[int],
    ) -> str:
        """
        Generates an evidence-based comparison summary explaining tendencies
        without claiming absolute outcomes.
        """
        seed_note = (
            f"Both outputs were initialized with seed `{seed}` for baseline reproducibility; "
            "however, temperature directly alters the logits softmax distribution prior to multinomial sampling."
            if seed is not None
            else "Outputs were produced with stochastic random sampling."
        )

        summary_lines = [
            "### 🔬 Temperature Comparison Analysis",
            "",
            f"| Metric / Property | Output A (Temperature: {temp_a}) | Output B (Temperature: {temp_b}) |",
            "| :--- | :--- | :--- |",
            f"| **Generated Tokens** | {res_a['new_tokens']} tokens | {res_b['new_tokens']} tokens |",
            f"| **Total Tokens** | {res_a['total_tokens']} tokens | {res_b['total_tokens']} tokens |",
            f"| **Lexical Diversity (TTR)** | {diversity_a} (unique words / total) | {diversity_b} (unique words / total) |",
            f"| **Relative Sampling Entropy** | {'Lower' if temp_a <= temp_b else 'Higher'} (Sharper probabilities) | {'Higher' if temp_b >= temp_a else 'Lower'} (Flatter probabilities) |",
            f"| **Primary Characteristic** | {'More focused & predictable' if temp_a <= temp_b else 'More adventurous & varied'} | {'More adventurous & varied' if temp_b >= temp_a else 'More focused & predictable'} |",
            "",
            "#### 💡 Key Observed Tendencies:",
            f"1. **Coherence vs. Entropy:** At **T={temp_a}**, sampling is {'more conservative and concentrated' if temp_a < temp_b else 'more dispersed and exploratory'}, whereas at **T={temp_b}**, sampling is {'more exploratory and dispersed' if temp_b > temp_a else 'more conservative and concentrated'}.",
            f"2. **Vocabulary Variation:** Output A achieved a lexical diversity score of **{diversity_a}**, whereas Output B achieved **{diversity_b}**. Higher temperatures typically encourage a wider vocabulary breadth.",
            f"3. **Predictability vs. Divergence:** Lower temperatures tightly preserve local syntactic consistency, whereas elevated temperatures encourage creative vocabulary and surprising plot twists with a higher risk of narrative drift.",
            "",
            f"> **Note on Probabilistic Outcomes:** {seed_note} Temperature sets a probability distribution rather than a deterministic outcome. Higher temperatures do not always guarantee superior creativity, nor does low temperature guarantee absolute grammatical coherence.",
        ]

        return "\n".join(summary_lines)


# Global engine accessor
_engine_instance: Optional[GPT2StoryEngine] = None


def get_story_engine(model_name: str = "openai-community/gpt2") -> GPT2StoryEngine:
    """Return the shared GPT2StoryEngine singleton instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = GPT2StoryEngine(model_name=model_name)
    return _engine_instance
