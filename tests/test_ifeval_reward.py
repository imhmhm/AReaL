"""Unit tests for IFEval reward function."""

from areal.reward import ifeval_reward_fn
from areal.reward.ifeval import _remove_thinking_section, _parse_ground_truth


class TestIFEvalRewardBasic:
    """Test cases for basic IFEval reward function."""

    def test_keywords_existence_correct(self):
        """Test reward when required keywords are present."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is transforming the world.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 1.0

    def test_keywords_existence_missing(self):
        """Test reward when required keywords are missing."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="Technology is advancing rapidly.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 0.0

    def test_keywords_existence_case_insensitive(self):
        """Test keyword matching is case insensitive."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="ai is transforming the world.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 1.0

    def test_keywords_existence_multiple_keywords(self):
        """Test multiple keywords all present."""
        reward = ifeval_reward_fn(
            prompt="Write about AI and future",
            completions="AI will shape the future of technology.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI", "future"]}]}]',
        )
        assert reward == 1.0

    def test_keywords_existence_partial_keywords(self):
        """Test partial keywords present."""
        reward = ifeval_reward_fn(
            prompt="Write about AI and future",
            completions="AI is important.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI", "future"]}]}]',
        )
        assert reward == 0.0


class TestIFEvalForbiddenWords:
    """Test forbidden words constraint."""

    def test_forbidden_words_not_present(self):
        """Test reward when forbidden words are absent."""
        reward = ifeval_reward_fn(
            prompt="Write a response",
            completions="This is a good response.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad", "evil"]}]}]',
        )
        assert reward == 1.0

    def test_forbidden_words_present(self):
        """Test reward when forbidden words are present."""
        reward = ifeval_reward_fn(
            prompt="Write a response",
            completions="This is a bad response.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad"]}]}]',
        )
        assert reward == 0.0

    def test_forbidden_words_case_insensitive(self):
        """Test forbidden word detection is case insensitive."""
        reward = ifeval_reward_fn(
            prompt="Write a response",
            completions="This is BAD.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:forbidden_words"], "kwargs": [{"forbidden_words": ["bad"]}]}]',
        )
        assert reward == 0.0


class TestIFEvalMultipleConstraints:
    """Test cases with multiple constraints."""

    def test_multiple_constraints_all_pass(self):
        """Test reward with multiple constraints, all passing."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is great. AI will change the world.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence", "keywords:frequency"], "kwargs": [{"keywords": ["AI"]}, {"word": "AI", "frequency": 2}]}]',
        )
        assert reward == 1.0

    def test_multiple_constraints_partial_pass(self):
        """Test reward with multiple constraints, partial passing."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI is great. Machine learning is also important.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence", "keywords:frequency"], "kwargs": [{"keywords": ["AI"]}, {"word": "AI", "frequency": 2}]}]',
        )
        assert reward == 0.5  # Only keyword existence passes

    def test_multiple_constraints_all_fail(self):
        """Test reward with multiple constraints, all failing."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="Technology is evolving.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence", "keywords:frequency"], "kwargs": [{"keywords": ["AI"]}, {"word": "AI", "frequency": 2}]}]',
        )
        assert reward == 0.0


class TestIFEvalEdgeCases:
    """Test edge cases for IFEval reward function."""

    def test_empty_completion(self):
        """Test reward for empty completion."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["test"]}]}]',
        )
        assert reward == 0.0

    def test_none_ground_truth(self):
        """Test reward when ground_truth is None."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="Test output.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth=None,
        )
        assert reward == 0.0

    def test_empty_instruction_list(self):
        """Test reward with empty instruction_id list (regression test for PR #1655)."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="Test output.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": [], "kwargs": []}]',
        )
        assert reward == 0.0

    def test_unknown_instruction_key(self):
        """Test reward with unknown instruction key."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="Test output.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["unknown:instruction"], "kwargs": []}]',
        )
        assert reward == 0.0


class TestIFEvalFormatConstraints:
    """Test format-related constraints."""

    def test_uppercase_constraint_correct(self):
        """Test reward for uppercase constraint."""
        reward = ifeval_reward_fn(
            prompt="Write in uppercase",
            completions="THIS IS ALL UPPERCASE",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["change_case:english_capital"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_uppercase_constraint_wrong(self):
        """Test reward fails for lowercase when uppercase required."""
        reward = ifeval_reward_fn(
            prompt="Write in uppercase",
            completions="this is lowercase",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["change_case:english_capital"], "kwargs": []}]',
        )
        assert reward == 0.0

    def test_lowercase_constraint_correct(self):
        """Test reward for lowercase constraint."""
        reward = ifeval_reward_fn(
            prompt="Write in lowercase",
            completions="this is all lowercase",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["change_case:english_lowercase"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_json_format_correct(self):
        """Test reward for JSON format constraint."""
        reward = ifeval_reward_fn(
            prompt="Return JSON",
            completions='{"status": "success", "value": 42}',
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["detectable_format:json_format"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_json_format_invalid(self):
        """Test reward for invalid JSON."""
        reward = ifeval_reward_fn(
            prompt="Return JSON",
            completions='not valid json',
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["detectable_format:json_format"], "kwargs": []}]',
        )
        assert reward == 0.0

    def test_quotation_correct(self):
        """Test reward for quotation wrapping."""
        reward = ifeval_reward_fn(
            prompt="Wrap in quotes",
            completions='"This is quoted"',
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["startend:quotation"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_quotation_wrong(self):
        """Test reward fails without quotes."""
        reward = ifeval_reward_fn(
            prompt="Wrap in quotes",
            completions="This is not quoted",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["startend:quotation"], "kwargs": []}]',
        )
        assert reward == 0.0


class TestIFEvalPunctuationConstraints:
    """Test punctuation-related constraints."""

    def test_no_comma_correct(self):
        """Test reward for no comma constraint."""
        reward = ifeval_reward_fn(
            prompt="No commas allowed",
            completions="This sentence has no commas at all",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["punctuation:no_comma"], "kwargs": []}]',
        )
        assert reward == 1.0

    def test_no_comma_with_commas(self):
        """Test reward fails when commas are present."""
        reward = ifeval_reward_fn(
            prompt="No commas allowed",
            completions="This, sentence, has commas",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["punctuation:no_comma"], "kwargs": []}]',
        )
        assert reward == 0.0


class TestIFEvalEndConstraints:
    """Test end phrase constraints."""

    def test_end_phrase_correct(self):
        """Test reward for end phrase constraint."""
        reward = ifeval_reward_fn(
            prompt="End with specific phrase",
            completions="This is my response. Thank you for asking!",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["startend:end_checker"], "kwargs": [{"end_phrase": "Thank you for asking!"}]}]',
        )
        assert reward == 1.0

    def test_end_phrase_wrong(self):
        """Test reward fails without correct end phrase."""
        reward = ifeval_reward_fn(
            prompt="End with specific phrase",
            completions="This is my response. Goodbye!",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["startend:end_checker"], "kwargs": [{"end_phrase": "Thank you for asking!"}]}]',
        )
        assert reward == 0.0


class TestRemoveThinkingSection:
    """Test cases for _remove_thinking_section helper."""

    def test_remove_thinking_tags(self):
        """Test removing thinking tags."""
        result = _remove_thinking_section("Let me analyzeThe answer is 42")
        assert "The answer is 42" in result

    def test_remove_answer_tags(self):
        """Test removing answer tags."""
        result = _remove_thinking_section("42")
        assert result == "42"

    def test_combined_tags(self):
        """Test removing both thinking and answer tags."""
        result = _remove_thinking_section(
            "Analysis42"
        )
        assert result == "42"

    def test_no_tags(self):
        """Test input without any tags."""
        result = _remove_thinking_section("Just plain text")
        assert result == "Just plain text"

    def test_empty_input(self):
        """Test empty input."""
        result = _remove_thinking_section("")
        assert result == ""

    def test_thinking_end_tag_only(self):
        """Test removing  tag."""
        result = _remove_thinking_section("Some analysis hereThe answer")
        assert result == "The answer"

    def test_thinking_start_tag_only(self):
        """Test handling start tag without end."""
        result = _remove_thinking_section("Let me think...")
        # Should remove everything before the last
        assert "think" not in result or result.strip() == ""

    def test_alternative_format(self):
        """Test alternative  and  tags."""
        result = _remove_thinking_section("AnalysisThe answer")
        assert "The answer" in result


class TestParseGroundTruth:
    """Test cases for _parse_ground_truth helper."""

    def test_parse_json_string(self):
        """Test parsing JSON string."""
        result = _parse_ground_truth('[{"instruction_id": ["test"], "kwargs": []}]')
        assert result["instruction_id"] == ["test"]

    def test_parse_python_dict_string(self):
        """Test parsing Python dict string (ast.literal_eval)."""
        result = _parse_ground_truth("{'instruction_id': ['test'], 'kwargs': []}")
        assert result["instruction_id"] == ["test"]

    def test_parse_nested_string(self):
        """Test parsing nested/double-encoded JSON string.

        This handles the case where ground_truth was accidentally JSON-encoded twice:
        - Original: {"instruction_id": ["test"]}
        - After json.dumps: '{"instruction_id": ["test"]}'
        - After another json.dumps: '"{\\"instruction_id\\": [\\"test\\"]}"'

        The Python string value contains: " followed by { followed by backslash-quote etc.
        When parsed by json.loads, returns: {"instruction_id": ["test"]} (a string)
        Then another json.loads returns: {"instruction_id": ["test"]} (a dict)
        """
        # Correct format: use json.dumps to create proper double-encoded string
        # to avoid confusion with Python escape sequences in source code
        import json
        original = {"instruction_id": ["test"]}
        first_encode = json.dumps(original)  # '{"instruction_id": ["test"]}'
        second_encode = json.dumps(first_encode)  # '"{\\"instruction_id\\": [\\"test\\"]}"'
        result = _parse_ground_truth(second_encode)
        assert result["instruction_id"] == ["test"]

    def test_parse_list_format(self):
        """Test parsing list format, taking first element."""
        result = _parse_ground_truth('[{"instruction_id": ["test"]}, {"instruction_id": ["other"]}]')
        assert result["instruction_id"] == ["test"]

    def test_parse_empty_list(self):
        """Test parsing empty list."""
        result = _parse_ground_truth('[]')
        assert result["instruction_id"] == []
        assert result["kwargs"] == []


class TestIFEvalLengthConstraints:
    """Test length-related constraints."""

    def test_word_count_at_least(self):
        """Test word count constraint with at least."""
        reward = ifeval_reward_fn(
            prompt="Write at least 10 words",
            completions="This response has more than ten words in it.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["length_constraints:number_words"], "kwargs": [{"num_words": 10, "relation": "at least"}]}]',
        )
        assert reward == 1.0

    def test_word_count_at_most(self):
        """Test word count constraint with at most."""
        reward = ifeval_reward_fn(
            prompt="Write at most 5 words",
            completions="Short response here.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["length_constraints:number_words"], "kwargs": [{"num_words": 5, "relation": "at most"}]}]',
        )
        assert reward == 1.0

    def test_word_count_too_long(self):
        """Test word count constraint fails when too long."""
        reward = ifeval_reward_fn(
            prompt="Write at most 5 words",
            completions="This is a very long response with many words.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["length_constraints:number_words"], "kwargs": [{"num_words": 5, "relation": "at most"}]}]',
        )
        assert reward == 0.0


class TestIFEvalBulletConstraints:
    """Test bullet list constraints."""

    def test_bullet_list_correct_count(self):
        """Test reward for correct bullet count."""
        reward = ifeval_reward_fn(
            prompt="Write 3 bullet points",
            completions="* Point one\n* Point two\n* Point three",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["detectable_format:number_bullet_lists"], "kwargs": [{"num_bullets": 3}]}]',
        )
        assert reward == 1.0

    def test_bullet_list_wrong_count(self):
        """Test reward fails for wrong bullet count."""
        reward = ifeval_reward_fn(
            prompt="Write 3 bullet points",
            completions="* Only one point",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["detectable_format:number_bullet_lists"], "kwargs": [{"num_bullets": 3}]}]',
        )
        assert reward == 0.0


class TestIFEvalRobustness:
    """Test robustness of IFEval reward function."""

    def test_malformed_ground_truth(self):
        """Test handling of malformed ground truth."""
        reward = ifeval_reward_fn(
            prompt="Write something",
            completions="Test output.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='invalid json',
        )
        assert reward == 0.0

    def test_unicode_content(self):
        """Test handling of unicode content."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI é transforming the world. 你好",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 1.0

    def test_very_long_completion(self):
        """Test handling of very long completion."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI " + "is " * 10000 + "important.",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 1.0

    def test_special_characters_in_completion(self):
        """Test handling of special characters."""
        reward = ifeval_reward_fn(
            prompt="Write about AI",
            completions="AI\t\n\r\0is here",
            prompt_ids=[],
            completion_ids=[],
            ground_truth='[{"instruction_id": ["keywords:existence"], "kwargs": [{"keywords": ["AI"]}]}]',
        )
        assert reward == 1.0