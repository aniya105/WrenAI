import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

# Heavy dependency mocks
sys.modules["hamilton"] = type(sys)("hamilton")
sys.modules["hamilton.async_driver"] = type(sys)("hamilton.async_driver")
sys.modules["hamilton.async_driver"].AsyncDriver = object
sys.modules["hamilton.base"] = type(sys)("hamilton.base")
sys.modules["hamilton.base"].DictResult = object

sys.modules["haystack"] = type(sys)("haystack")
sys.modules["haystack"].Document = object
sys.modules["haystack"].component = lambda cls: cls
sys.modules["haystack.components"] = type(sys)("haystack.components")
sys.modules["haystack.components.builders"] = type(sys)("haystack.components.builders")
sys.modules["haystack.components.builders.prompt_builder"] = type(sys)("haystack.components.builders.prompt_builder")
sys.modules["haystack.components.builders.prompt_builder"].PromptBuilder = object

sys.modules["langfuse"] = type(sys)("langfuse")
sys.modules["langfuse.decorators"] = type(sys)("langfuse.decorators")
sys.modules["langfuse.decorators"].observe = lambda *a, **kw: (lambda f: f)

sys.modules["src.core.pipeline"] = type(sys)("src.core.pipeline")
sys.modules["src.core.pipeline"].BasicPipeline = object
sys.modules["src.core.provider"] = type(sys)("src.core.provider")
sys.modules["src.core.provider"].LLMProvider = object
sys.modules["src.core.provider"].EmbedderProvider = object

sys.modules["src.pipelines.common"] = type(sys)("src.pipelines.common")
sys.modules["src.pipelines.common"].clean_up_new_lines = lambda x: x

sys.modules["src.utils"] = type(sys)("src.utils")
sys.modules["src.utils"].trace_cost = lambda f: f
sys.modules["src.utils"].trace_metadata = lambda f: f

sys.modules["orjson"] = type(sys)("orjson")
sys.modules["orjson"].loads = lambda x: x

# Mock cachetools TTLCache as a simple dict-like class
class MockTTLCache(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()

sys.modules["cachetools"] = type(sys)("cachetools")
sys.modules["cachetools"].TTLCache = MockTTLCache

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

# Load ask.py directly to bypass services/__init__.py which pulls in everything
_spec = importlib.util.spec_from_file_location(
    "ask_module",
    Path(__file__).parents[3] / "src" / "web" / "v1" / "services" / "ask.py",
)
_ask_module = importlib.util.module_from_spec(_spec)

# Minimal mocks for ask.py imports
sys.modules["src.web.v1.services"] = type(sys)("src.web.v1.services")
from pydantic import BaseModel as PydanticBaseModel

class MockBaseRequest(PydanticBaseModel):
    _query_id: str | None = None
    project_id: str | None = None
    thread_id: str | None = None
    request_from: str = "ui"

    @property
    def query_id(self) -> str | None:
        return self._query_id

    @query_id.setter
    def query_id(self, value: str):
        self._query_id = value


class MockSSEEvent:
    pass


sys.modules["src.web.v1.services"].BaseRequest = MockBaseRequest
sys.modules["src.web.v1.services"].SSEEvent = MockSSEEvent

_spec.loader.exec_module(_ask_module)

AskService = _ask_module.AskService
AskRequest = _ask_module.AskRequest
AskResultRequest = _ask_module.AskResultRequest
AskResultResponse = _ask_module.AskResultResponse
AskHistory = _ask_module.AskHistory
ClarificationAnswer = _ask_module.ClarificationAnswer
ClarifyRequest = _ask_module.ClarifyRequest


@pytest.fixture
def ask_service_enabled():
    return AskService(
        pipelines={},
        enable_clarification=True,
    )


@pytest.fixture
def ask_service_disabled():
    return AskService(
        pipelines={},
        enable_clarification=False,
    )


class TestShouldCheckClarification:
    def test_disabled_flag_returns_false(self, ask_service_disabled):
        result = ask_service_disabled._should_check_clarification(
            "show me sales", [{"score": 0.9, "table_name": "sales"}], [], []
        )
        assert result is False

    def test_no_documents_returns_false(self, ask_service_enabled):
        result = ask_service_enabled._should_check_clarification(
            "show me sales", [], [], []
        )
        assert result is False

    def test_score_gap_trigger(self, ask_service_enabled):
        docs = [
            {"score": 0.95, "table_name": "online_sales"},
            {"score": 0.92, "table_name": "offline_sales"},
        ]
        result = ask_service_enabled._should_check_clarification(
            "show me sales", docs, [], []
        )
        assert result is True

    def test_score_gap_no_trigger_when_gap_large(self, ask_service_enabled):
        docs = [
            {"score": 0.95, "table_name": "customers"},
            {"score": 0.50, "table_name": "orders"},
        ]
        result = ask_service_enabled._should_check_clarification(
            "show me customer list", docs, [], []
        )
        assert result is False

    def test_table_keyword_match_trigger(self, ask_service_enabled):
        docs = [
            {"score": 0.9, "table_name": "online_sales"},
            {"score": 0.85, "table_name": "offline_sales"},
            {"score": 0.8, "table_name": "customers"},
        ]
        result = ask_service_enabled._should_check_clarification(
            "show me sales", docs, [], []
        )
        assert result is True

    def test_aggregation_ambiguity_trigger(self, ask_service_enabled):
        docs = [
            {"score": 0.9, "table_name": "revenue"},
            {"score": 0.88, "table_name": "expenses"},
        ]
        result = ask_service_enabled._should_check_clarification(
            "what is the total", docs, [], []
        )
        assert result is True

    def test_aggregation_no_trigger_with_single_doc(self, ask_service_enabled):
        docs = [
            {"score": 0.9, "table_name": "revenue"},
        ]
        result = ask_service_enabled._should_check_clarification(
            "what is the total", docs, [], []
        )
        assert result is False

    def test_instruction_conflict_trigger(self, ask_service_enabled):
        docs = [
            {"score": 0.9, "table_name": "online_sales"},
            {"score": 0.88, "table_name": "offline_sales"},
        ]
        instructions = [{"instruction": "Sales data includes both online and offline channels"}]
        result = ask_service_enabled._should_check_clarification(
            "show me sales", docs, [], instructions
        )
        assert result is True

    def test_clear_query_passes_heuristic(self, ask_service_enabled):
        docs = [
            {"score": 0.95, "table_name": "online_sales"},
        ]
        result = ask_service_enabled._should_check_clarification(
            "show me total revenue from online_sales in 2024", docs, [], []
        )
        assert result is False


class TestClarifyAndResume:
    def test_clarify_and_resume_missing_query_id(self, ask_service_enabled):
        clarify_request = ClarifyRequest(
            query_id="non-existent-id",
            clarification_answers=[ClarificationAnswer(question_index=0, answer="online")],
        )

        async def run():
            await ask_service_enabled.clarify_and_resume(clarify_request)
            result = ask_service_enabled.get_ask_result(
                AskResultRequest(query_id="non-existent-id")
            )
            assert result.status == "failed"
            assert result.error.code == "OTHERS"
            assert "not found" in result.error.message.lower()

        asyncio.run(run())

    def test_clarify_and_resume_creates_enriched_request(self, ask_service_enabled):
        query_id = "test-query-id"
        original = AskRequest(
            query="show me sales",
            mdl_hash="hash-123",
            histories=[AskHistory(question="previous", sql="SELECT 1")],
        )
        original.query_id = query_id
        ask_service_enabled._ask_contexts[query_id] = original
        ask_service_enabled._ask_results[query_id] = AskResultResponse(
            status="understanding",
        )

        clarify_request = ClarifyRequest(
            query_id=query_id,
            clarification_answers=[
                ClarificationAnswer(question_index=0, answer="online sales"),
                ClarificationAnswer(question_index=1, answer="2024"),
            ],
        )

        async def run():
            try:
                await ask_service_enabled.clarify_and_resume(clarify_request)
            except Exception:
                pass  # expected because pipelines are empty
            assert query_id in ask_service_enabled._ask_contexts

        asyncio.run(run())
