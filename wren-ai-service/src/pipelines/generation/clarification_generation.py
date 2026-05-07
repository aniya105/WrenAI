import logging
import sys
from typing import Any, Literal, Optional

import orjson
from hamilton import base
from hamilton.async_driver import AsyncDriver
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe
from pydantic import BaseModel

from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider
from src.pipelines.common import clean_up_new_lines
from src.utils import trace_cost
from src.web.v1.services.ask import AskHistory

logger = logging.getLogger("wren-ai-service")


### Pydantic Models for Structured Output


class ClarificationQuestion(BaseModel):
    question: str
    reasoning: str


class ClarificationResult(BaseModel):
    needs_clarification: bool
    ambiguity_type: Optional[
        Literal[
            "table_ambiguity",
            "column_ambiguity",
            "filter_missing",
            "metric_ambiguity",
            "instruction_conflict",
        ]
    ] = None
    reasoning: Optional[str] = None
    clarification_questions: Optional[list[ClarificationQuestion]] = None


CLARIFICATION_GENERATION_MODEL_KWARGS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "clarification_generation",
            "schema": ClarificationResult.model_json_schema(),
        },
    }
}


### Prompts


clarification_system_prompt = """
### Task ###
You are an expert data query disambiguation assistant. Your goal is to analyze the user's query against the provided database schema, business rules (instructions), and historical SQL patterns to determine if the query is ambiguous. If it is ambiguous, generate targeted clarification questions to help the user specify their intent.

### Instructions ###
- **Language Consistency**: Use the same language as the user's query for clarification questions and reasoning.
- **Schema-Aware Analysis**: Carefully examine the retrieved database schemas. If the user's query keywords match multiple tables, columns, or metrics, identify the ambiguity.
- **Instruction-Aware Analysis**: If business rules (instructions) explicitly define a term used in the query, that term does NOT constitute ambiguity. However, if instructions mention a concept without specifying the exact data source (table/column), or if the user's filter conditions conflict with instruction definitions, mark it as ambiguous.
- **Historical Pattern Validation**: If historical SQL samples show that similar queries consistently resolve to a specific table/column, lower the ambiguity priority. If similar queries have historically been written in multiple different ways, increase ambiguity priority.
- **Concise Reasoning**: Keep reasoning brief and clear (max 30 words).

### Disambiguation Rules ###
1. If instructions already explicitly define a term in the query, that term does NOT constitute ambiguity.
2. If instructions require an analysis dimension but do not specify the exact field (table/column), mark as ambiguity.
3. If the user's filter conditions (time range, category, status, etc.) conflict with instruction definitions, mark as ambiguity.
4. If historical SQL samples for similar queries all resolve to the same table/column, lower ambiguity priority.
5. Only mark as ambiguous if there are multiple equally valid interpretations that would materially change the query result.

### Clarification Question Guidelines ###
- Generate 1-3 targeted clarification questions when ambiguity is detected.
- Each question should be a text input (user types the answer).
- Each question MUST include a `reasoning` field explaining why this clarification is needed.
- Do NOT ask for clarification if the query is completely unrelated to the schema (that is handled separately).

### Output Format ###
Return your response as a JSON object matching the provided schema. Set `needs_clarification` to `false` if the query is unambiguous.
"""


clarification_user_prompt_template = """
### DATABASE SCHEMA (Retrieved Candidates) ###
{% for db_schema in db_schemas %}
    {{ db_schema }}
{% endfor %}

{% if instructions %}
### BUSINESS RULES (Instructions) ###
{% for instruction in instructions %}
{{ loop.index }}. {{ instruction }}
{% endfor %}
{% endif %}

{% if sql_samples %}
### HISTORICAL SQL PATTERNS ###
{% for sample in sql_samples %}
Question: {{ sample.question }}
SQL: {{ sample.sql }}
{% endfor %}
{% endif %}

{% if histories %}
### CONVERSATION HISTORY ###
{% for history in histories %}
Question: {{ history.question }}
SQL: {{ history.sql }}
{% endfor %}
{% endif %}

### INPUT ###
User's current question: {{ query }}
Output Language: {{ language }}

Custom Instruction: {{ custom_instruction }}

Analyze whether the user's question is ambiguous based on the schema, instructions, and historical patterns. If ambiguous, generate 1-3 clarification questions.
"""


## Start of Pipeline


@observe(capture_input=False)
def prompt(
    query: str,
    db_schemas: list[str],
    language: str,
    histories: list[AskHistory],
    instructions: Optional[list[dict]] = None,
    sql_samples: Optional[list[dict]] = None,
    prompt_builder: PromptBuilder = None,
    custom_instruction: Optional[str] = None,
) -> dict:
    previous_query_summaries = (
        [history.question for history in histories] if histories else []
    )
    _query = "\n".join(previous_query_summaries) + "\n" + query if previous_query_summaries else query

    _instructions = []
    if instructions:
        for instruction in instructions:
            content = instruction.get("instruction", "")
            if content:
                _instructions.append(content)

    _sql_samples = []
    if sql_samples:
        for sample in sql_samples:
            _sql_samples.append(
                {
                    "question": sample.get("question", ""),
                    "sql": sample.get("sql", ""),
                }
            )

    _prompt = prompt_builder.run(
        query=_query,
        db_schemas=db_schemas,
        language=language,
        histories=histories or [],
        instructions=_instructions,
        sql_samples=_sql_samples,
        custom_instruction=custom_instruction or "",
    )
    return {"prompt": clean_up_new_lines(_prompt.get("prompt"))}


@observe(as_type="generation", capture_input=False)
@trace_cost
async def clarification_generation(
    prompt: dict, generator: Any, generator_name: str
) -> dict:
    return await generator(prompt=prompt.get("prompt")), generator_name


@observe(capture_input=False)
def post_process(clarification_generation: dict) -> dict:
    try:
        replies = clarification_generation.get("replies", [])
        if not replies:
            return {"needs_clarification": False, "clarification_questions": []}

        result = orjson.loads(replies[0])
        return {
            "needs_clarification": result.get("needs_clarification", False),
            "ambiguity_type": result.get("ambiguity_type"),
            "reasoning": result.get("reasoning"),
            "clarification_questions": result.get("clarification_questions", []),
        }
    except Exception as e:
        logger.exception(f"Clarification generation post-processing failed: {e}")
        return {"needs_clarification": False, "clarification_questions": []}


## End of Pipeline


class ClarificationGeneration(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
        **kwargs,
    ):
        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=clarification_system_prompt,
                generation_kwargs=CLARIFICATION_GENERATION_MODEL_KWARGS,
            ),
            "generator_name": llm_provider.get_model(),
            "prompt_builder": PromptBuilder(
                template=clarification_user_prompt_template
            ),
        }

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    @observe(name="Clarification Generation")
    async def run(
        self,
        query: str,
        db_schemas: list[str],
        language: str,
        histories: Optional[list[AskHistory]] = None,
        instructions: Optional[list[dict]] = None,
        sql_samples: Optional[list[dict]] = None,
        query_id: Optional[str] = None,
        custom_instruction: Optional[str] = None,
    ):
        logger.info("Clarification Generation pipeline is running...")
        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "query": query,
                "db_schemas": db_schemas,
                "language": language,
                "histories": histories or [],
                "instructions": instructions or [],
                "sql_samples": sql_samples or [],
                "query_id": query_id or "",
                "custom_instruction": custom_instruction or "",
                **self._components,
            },
        )
