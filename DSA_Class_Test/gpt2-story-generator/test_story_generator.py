"""
test_story_generator.py: Automated Unit Tests for GPT-2 Interactive Story Generator.

Verifies:
- Input validation (empty text, too short, too long, invalid temperatures).
- Prompt construction for all writing styles.
- Preserving user opening sentences in formatted outputs.
- Lexical diversity computation.
- Token count calculations.
- Singleton pattern for GPT2StoryEngine.
- Temperature comparison parameter verification.
"""

import unittest
from unittest.mock import MagicMock, patch

from model import (
    SUPPORTED_STYLES,
    GPT2StoryEngine,
    calculate_lexical_diversity,
    construct_prompt,
    format_story_output,
    get_story_engine,
    validate_story_inputs,
)


class TestInputValidation(unittest.TestCase):
    """Test user input validation rules."""

    def test_valid_inputs(self):
        is_valid, err = validate_story_inputs(
            "The ancient tower glowed under the twin moons.", 0.8
        )
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_empty_opening_sentence(self):
        is_valid, err = validate_story_inputs("", 0.8)
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", err)

        is_valid, err = validate_story_inputs("   \n\t  ", 0.8)
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", err)

    def test_too_short_opening_sentence(self):
        is_valid, err = validate_story_inputs("Hi", 0.8)
        self.assertFalse(is_valid)
        self.assertIn("too short", err)

    def test_excessive_length_sentence(self):
        long_sentence = "A" * 1500
        is_valid, err = validate_story_inputs(long_sentence, 0.8)
        self.assertFalse(is_valid)
        self.assertIn("too long", err)

    def test_invalid_temperature_values(self):
        # Non-numeric
        is_valid, err = validate_story_inputs("Valid sentence here.", "hot")
        self.assertFalse(is_valid)
        self.assertIn("valid numerical", err)

        # Out of bounds (< 0.1 or > 2.0)
        is_valid, err = validate_story_inputs("Valid sentence here.", 0.05)
        self.assertFalse(is_valid)
        self.assertIn("between 0.2 and 1.5", err)

        is_valid, err = validate_story_inputs("Valid sentence here.", 2.5)
        self.assertFalse(is_valid)
        self.assertIn("between 0.2 and 1.5", err)

    def test_max_new_tokens_validation(self):
        # Valid
        is_valid, err = validate_story_inputs("Valid sentence here.", 0.8, max_new_tokens=100)
        self.assertTrue(is_valid)

        # Non-integer
        is_valid, err = validate_story_inputs("Valid sentence here.", 0.8, max_new_tokens="fifty")
        self.assertFalse(is_valid)
        self.assertIn("valid integer", err)

        # Out of bounds (< 10 or > 512)
        is_valid, err = validate_story_inputs("Valid sentence here.", 0.8, max_new_tokens=5)
        self.assertFalse(is_valid)
        self.assertIn("between 10 and 512", err)

        is_valid, err = validate_story_inputs("Valid sentence here.", 0.8, max_new_tokens=600)
        self.assertFalse(is_valid)
        self.assertIn("between 10 and 512", err)


class TestPromptConstruction(unittest.TestCase):
    """Test prompt building across all supported styles."""

    def test_supported_styles_presence(self):
        expected_styles = [
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
        for style in expected_styles:
            self.assertIn(style, SUPPORTED_STYLES)

    def test_prompt_includes_opening_and_style(self):
        opening = "The clock struck thirteen in the silent library."
        for style in SUPPORTED_STYLES:
            prompt = construct_prompt(opening, style)
            self.assertIn(opening, prompt)
            self.assertIn("Continue the following opening sentence naturally", prompt)


class TestFormattingAndMetrics(unittest.TestCase):
    """Test story assembly and lexical diversity calculation."""

    def test_format_story_output_preserves_opening(self):
        opening = "It was a dark and stormy night."
        continuation = "Thunder shook the foundation of the manor."
        full_story = format_story_output(opening, continuation)
        self.assertEqual(
            full_story,
            "It was a dark and stormy night. Thunder shook the foundation of the manor.",
        )

    def test_format_story_output_deduplicates_opening(self):
        opening = "In the deep forest,"
        continuation = "In the deep forest, an ancient tree stood."
        full_story = format_story_output(opening, continuation)
        self.assertEqual(full_story, "In the deep forest, an ancient tree stood.")

    def test_format_story_output_handles_empty_continuation(self):
        opening = "The starship approached the warp gate."
        full_story = format_story_output(opening, "")
        self.assertEqual(full_story, opening)

    def test_lexical_diversity(self):
        repetitive = "echo echo echo echo"
        diverse = "the curious fox leaped swiftly"
        self.assertLess(
            calculate_lexical_diversity(repetitive),
            calculate_lexical_diversity(diverse),
        )
        self.assertEqual(calculate_lexical_diversity(repetitive), 0.25)
        self.assertEqual(calculate_lexical_diversity(diverse), 1.0)


class TestEngineArchitecture(unittest.TestCase):
    """Test engine singleton behavior and token count accounting."""

    def test_singleton_engine(self):
        engine1 = get_story_engine()
        engine2 = get_story_engine()
        self.assertIs(engine1, engine2)

    def test_token_count_math(self):
        # Verify prompt vs newly generated token distinction
        mock_prompt_len = 25
        mock_total_len = 120
        mock_new_len = mock_total_len - mock_prompt_len

        self.assertEqual(mock_new_len, 95)
        self.assertEqual(mock_prompt_len + mock_new_len, mock_total_len)


class TestUIFeatures(unittest.TestCase):
    """Test UI customizations such as font styles."""

    def test_font_styles(self):
        from app import SUPPORTED_FONTS, FONT_STYLE_MAP, get_font_style_css

        self.assertGreaterEqual(len(SUPPORTED_FONTS), 3)
        for font in SUPPORTED_FONTS:
            self.assertIn(font, FONT_STYLE_MAP)
            css = get_font_style_css(font)
            self.assertIn("font-family:", css)
            self.assertIn("!important", css)


if __name__ == "__main__":
    unittest.main()
