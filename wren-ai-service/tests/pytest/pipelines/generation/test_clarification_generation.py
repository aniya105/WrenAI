import importlib.util
import json
import sys
from pathlib import Path

import pytest


# Heavy dependencies are not available in this bare environment;
# we load the module under test directly from file to bypass
# __init__.py imports that pull in hamilton/haystack/etc.
_spec = importlib.util.spec_from_file_location(
    "clarification_generation",
    Path(__file__).parents[4] / "src" / "pipelines" / "generation" / "clarification_generation.py",
)
clarification_generation = importlib.util.module_from_spec(_spec)

# Minimal mocks required by the module itself
sys.modules["hamilton"] = type(sys)("hamilton")
sys.modules["hamilton.async_driver"] = type(sys)("hamilton.async_driver")
sys.modules["hamilton.async_driver"].AsyncDriver = object
sys.modules["hamilton.base"] = type(sys)("hamilton.base")
sys.modules["hamilton.base"].DictResult = object
sys.modules["haystack.components.builders.prompt_builder"] = type(sys)("haystack.components.builders.prompt_builder")
sys.modules["haystack.components.builders.prompt_builder"].PromptBuilder = object
sys.modules["langfuse.decorators"] = type(sys)("langfuse.decorators")
sys.modules["langfuse.decorators"].observe = lambda *a, **kw: (lambda f: f)
sys.modules["src.core.pipeline"] = type(sys)("src.core.pipeline")
sys.modules["src.core.pipeline"].BasicPipeline = object
sys.modules["src.core.provider"] = type(sys)("src.core.provider")
sys.modules["src.core.provider"].LLMProvider = object
sys.modules["src.pipelines.common"] = type(sys)("src.pipelines.common")
sys.modules["src.pipelines.common"].clean_up_new_lines = lambda x: x
sys.modules["src.utils"] = type(sys)("src.utils")
sys.modules["src.utils"].trace_cost = lambda f: f
sys.modules["src.web.v1.services.ask"] = type(sys)("src.web.v1.services.ask")
sys.modules["src.web.v1.services.ask"].AskHistory = type(sys)("AskHistory")

_spec.loader.exec_module(clarification_generation)

ClarificationQuestion = clarification_generation.ClarificationQuestion
ClarificationResult = clarification_generation.ClarificationResult
post_process = clarification_generation.post_process


class TestClarificationModels:
    def test_clarification_result_valid(self):
        result = ClarificationResult(
            needs_clarification=True,
            ambiguity_type="table_ambiguity",
            reasoning="Multiple tables match 'sales'",
            clarification_questions=[
                ClarificationQuestion(
                    question="Which sales table?",
                    reasoning="Schema has both online_sales and offline_sales",
                )
            ],
        )
        assert result.needs_clarification is True
        assert result.ambiguity_type == "table_ambiguity"
        assert len(result.clarification_questions) == 1
        assert result.clarification_questions[0].question == "Which sales table?"

    def test_clarification_result_no_clarification(self):
        result = ClarificationResult(needs_clarification=False)
        assert result.needs_clarification is False
        assert result.clarification_questions is None
        assert result.ambiguity_type is None

    def test_clarification_question_with_reasoning(self):
        question = ClarificationQuestion(
            question="What time range?",
            reasoning="User didn't specify a date filter",
        )
        assert question.question == "What time range?"
        assert question.reasoning == "User didn't specify a date filter"

    def test_invalid_ambiguity_type_rejected(self):
        with pytest.raises(ValueError):
            ClarificationResult(
                needs_clarification=True,
                ambiguity_type="invalid_type",
            )


class TestPostProcess:
    def test_post_process_valid_json(self):
        reply = json.dumps(
            {
                "needs_clarification": True,
                "ambiguity_type": "column_ambiguity",
                "reasoning": "Multiple revenue columns found",
                "clarification_questions": [
                    {
                        "question": "Which revenue?",
                        "reasoning": "Both exist in schema",
                    }
                ],
            }
        )
        result = post_process({"replies": [reply]})
        assert result["needs_clarification"] is True
        assert result["ambiguity_type"] == "column_ambiguity"
        assert len(result["clarification_questions"]) == 1
        assert result["clarification_questions"][0]["question"] == "Which revenue?"

    def test_post_process_no_replies(self):
        result = post_process({"replies": []})
        assert result["needs_clarification"] is False
        assert result["clarification_questions"] == []

    def test_post_process_invalid_json_fallback(self):
        result = post_process({"replies": ["not valid json"]})
        assert result["needs_clarification"] is False
        assert result["clarification_questions"] == []

    def test_post_process_missing_keys(self):
        reply = json.dumps({"needs_clarification": False})
        result = post_process({"replies": [reply]})
        assert result["needs_clarification"] is False
        assert result["ambiguity_type"] is None
        assert result["reasoning"] is None
        assert result["clarification_questions"] == []

    def test_post_process_malformed_question_still_parses(self):
        reply = json.dumps(
            {
                "needs_clarification": True,
                "clarification_questions": [
                    {
                        "question": "Bad question",
                        # missing reasoning is ok since it's Optional in old model but required in new
                    }
                ],
            }
        )
        result = post_process({"replies": [reply]})
        assert result["needs_clarification"] is True
        assert len(result["clarification_questions"]) == 1
        assert result["clarification_questions"][0]["question"] == "Bad question"


class TestJSONSchemaOutput:
    def test_model_json_schema_structure(self):
        schema = ClarificationResult.model_json_schema()
        assert "needs_clarification" in schema["properties"]
        assert schema["properties"]["needs_clarification"]["type"] == "boolean"
        assert "ambiguity_type" in schema["properties"]
        assert "clarification_questions" in schema["properties"]
        assert "reasoning" in schema["properties"]

        # Check nested question schema
        question_props = schema["$defs"]["ClarificationQuestion"]["properties"]
        assert "question" in question_props
        assert "reasoning" in question_props

    def test_model_json_schema_enum_values(self):
        schema = ClarificationResult.model_json_schema()
        # Pydantic v2 wraps Optional[Literal] in anyOf
        ambiguity_anyof = schema["properties"]["ambiguity_type"]["anyOf"]
        ambiguity_enum = ambiguity_anyof[0]["enum"]
        assert "table_ambiguity" in ambiguity_enum
        assert "column_ambiguity" in ambiguity_enum
        assert "filter_missing" in ambiguity_enum
        assert "metric_ambiguity" in ambiguity_enum
        assert "instruction_conflict" in ambiguity_enum
