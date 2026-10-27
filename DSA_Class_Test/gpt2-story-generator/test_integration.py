"""
test_integration.py: Comprehensive Integration Tests for GPT-2 Story Generator.

Validates:
- Real GPT-2 model and tokenizer inference.
- Non-empty output incorporating the user's opening sentence.
- Token count accounting (new_tokens + prompt_tokens == total_tokens).
- Prompt tokens excluded from newly generated tokens.
- All 10 writing styles accepted and prompt constructed properly.
- Custom selectable max new tokens limit.
- Temperature comparison generation with selectable temperatures (e.g. 0.4 and 1.1).
- Comparison summary formatting and metrics.
- Gradio app instantiation without UI errors.
"""

import unittest
from model import get_story_engine, SUPPORTED_STYLES
from app import create_app, generate_story_handler, compare_temperatures_handler


class TestStoryGeneratorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_story_engine()
        loaded, err = cls.engine.load_model()
        assert loaded, f"Model failed to load: {err}"

    def test_01_story_generation_and_custom_tokens(self):
        opening = "The forgotten library held books that wrote themselves."
        requested_tokens = 35
        result = self.engine.generate_story(
            opening_sentence=opening,
            style="Fantasy",
            temperature=0.8,
            max_new_tokens=requested_tokens,
        )
        self.assertTrue(result["success"])
        self.assertIn(opening, result["story"])
        self.assertGreater(result["new_tokens"], 0)
        self.assertLessEqual(result["new_tokens"], requested_tokens)
        self.assertGreater(result["prompt_tokens"], 0)
        self.assertEqual(
            result["total_tokens"],
            result["prompt_tokens"] + result["new_tokens"]
        )

    def test_02_all_writing_styles_accepted(self):
        opening = "A mysterious signal echoed from the deep cavern."
        for style in SUPPORTED_STYLES:
            prompt = self.engine.generate_story(
                opening_sentence=opening,
                style=style,
                temperature=0.7,
                max_new_tokens=10,
            )
            self.assertTrue(prompt["success"], f"Failed for style: {style}")

    def test_03_temperature_comparison_with_selectable_temperatures(self):
        opening = "The clock struck midnight as the fog rolled in."
        custom_temp_a = 0.4
        custom_temp_b = 1.1
        cmp = self.engine.compare_temperatures(
            opening_sentence=opening,
            style="Mystery",
            temp_a=custom_temp_a,
            temp_b=custom_temp_b,
            max_new_tokens=25,
            seed=42,
        )
        self.assertTrue(cmp["success"])
        out_a = cmp["output_a"]
        out_b = cmp["output_b"]
        self.assertIsNotNone(out_a)
        self.assertIsNotNone(out_b)
        self.assertIn(opening, out_a["story"])
        self.assertIn(opening, out_b["story"])
        self.assertEqual(out_a["temperature"], custom_temp_a)
        self.assertEqual(out_b["temperature"], custom_temp_b)
        self.assertLessEqual(out_a["new_tokens"], 25)
        self.assertLessEqual(out_b["new_tokens"], 25)
        self.assertIn("Temperature Comparison Analysis", cmp["summary"])
        self.assertIn(f"Output A (Temperature: {custom_temp_a})", cmp["summary"])
        self.assertIn(f"Output B (Temperature: {custom_temp_b})", cmp["summary"])

    def test_04_gradio_handlers(self):
        # Test empty input validation in generate handler
        story, metrics, status = generate_story_handler("", "Fantasy", 0.8, 150)
        self.assertIn("Validation Error", status)
        self.assertEqual(story, "")

        # Test empty input in comparison handler
        story_a, stats_a, story_b, stats_b, summary, cmp_status = compare_temperatures_handler(
            "", "Fantasy", 0.5, 1.2, 150
        )
        self.assertIn("Validation Error", cmp_status)

        # Test invalid temperature validation in comparison handler
        story_a, stats_a, story_b, stats_b, summary, cmp_status = compare_temperatures_handler(
            "Valid sentence.", "Fantasy", -0.5, 1.2, 150
        )
        self.assertIn("Validation Error", cmp_status)

    def test_05_gradio_app_builds(self):
        demo = create_app()
        self.assertIsNotNone(demo)


if __name__ == "__main__":
    unittest.main()
