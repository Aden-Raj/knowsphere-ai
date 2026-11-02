import pytest

from app.services.conversation_memory import ConversationHistoryMessage as Message
from app.services.follow_up_resolver import resolve_follow_up


@pytest.mark.parametrize(
    "question",
    ["What technologies are used in KnowSphere AI?", "Summarize my professional experience."],
)
def test_standalone_question_bypasses_resolution(question):
    result = resolve_follow_up(question, [Message("assistant", "KnowSphere AI uses RAG.")])

    assert result.original_question == question
    assert result.retrieval_question == question
    assert result.resolved is False
    assert result.status == "standalone"


def test_reference_without_history_is_left_unchanged():
    question = "What technologies does that project use?"

    result = resolve_follow_up(question, [])

    assert result.retrieval_question == question
    assert result.resolved is False
    assert result.status == "no_history"


def test_those_projects_resolves_from_most_recent_assistant_turn():
    question = "Which of those projects uses RAG?"
    result = resolve_follow_up(
        question,
        [Message("assistant", "The resume mentions two projects: AI-Powered Quiz Generator and KnowSphere AI.")],
    )

    assert result.retrieval_question == "Which of these projects (AI-Powered Quiz Generator and KnowSphere AI) uses RAG?"
    assert result.original_question == question
    assert result.resolved is True
    assert result.status == "resolved_plural_reference"


def test_that_project_resolves_when_history_has_one_clear_project():
    question = "What technologies does that project use?"
    result = resolve_follow_up(
        question,
        [Message("assistant", "The project is KnowSphere AI."), Message("user", "What about its backend?")],
    )

    assert result.retrieval_question == "What technologies does KnowSphere AI use?"
    assert result.original_question == question
    assert result.status == "resolved_singular_reference"


def test_its_backend_resolves_without_duplicate_component_word():
    result = resolve_follow_up(
        "What about its backend?", [Message("assistant", "Project: KnowSphere AI.")],
    )

    assert result.retrieval_question == "What about the backend of KnowSphere AI?"


def test_ordinal_reference_selects_requested_project_from_recent_list():
    result = resolve_follow_up(
        "Tell me more about the second one?",
        [Message("assistant", "Projects: AI-Powered Quiz Generator and KnowSphere AI.")],
    )

    assert result.retrieval_question == "Tell me more about KnowSphere AI?"
    assert result.original_question == "Tell me more about the second one?"
    assert result.status == "resolved_ordinal_reference"


def test_ambiguous_singular_reference_is_not_guessed():
    question = "What does that project use?"
    result = resolve_follow_up(
        question, [Message("assistant", "Projects: AI-Powered Quiz Generator and KnowSphere AI.")],
    )

    assert result.retrieval_question == question
    assert result.resolved is False
    assert result.status == "ambiguous_reference"


def test_reference_without_extractable_antecedent_is_unresolved():
    question = "What about it?"
    result = resolve_follow_up(question, [Message("assistant", "The answer is not available.")])

    assert result.retrieval_question == question
    assert result.status == "unresolved_reference"


def test_employer_or_framework_mentions_are_not_treated_as_project_antecedents():
    question = "What does that project use?"
    result = resolve_follow_up(
        question,
        [Message("assistant", "At TCS, I used Angular and REST APIs.")],
    )

    assert result.retrieval_question == question
    assert result.status == "unresolved_reference"


@pytest.mark.parametrize(
    "assistant_content",
    [
        "The UI uses Angular Material and React Hooks.",
        "Technologies: Python, Angular Material, React Hooks, and FastAPI.",
        "Skills: Angular Material; React Hooks; FastAPI.",
    ],
)
def test_technology_or_skill_lists_do_not_create_project_entities(assistant_content):
    result = resolve_follow_up(
        "What does that project use?", [Message("assistant", assistant_content)],
    )

    assert result.resolved is False
    assert result.status == "unresolved_reference"
    assert result.retrieval_question == "What does that project use?"


def test_technology_list_ordinal_is_not_resolved_as_a_project():
    question = "What does the second one use?"
    result = resolve_follow_up(
        question, [Message("assistant", "Technologies: Angular Material and React Hooks.")],
    )

    assert result.resolved is False
    assert result.status == "unresolved_reference"
    assert result.retrieval_question == question


def test_skills_list_does_not_satisfy_that_project_reference():
    question = "Tell me more about that project."
    result = resolve_follow_up(
        question, [Message("assistant", "Skills: Angular Material and React Hooks.")],
    )

    assert result.resolved is False
    assert result.status == "unresolved_reference"
    assert result.retrieval_question == question


def test_mixed_project_and_technology_lists_resolve_only_project_names():
    result = resolve_follow_up(
        "What technologies does that project use?",
        [Message(
            "assistant",
            "Projects: KnowSphere AI and Invoice Copilot. Technologies: Angular Material and React Hooks.",
        )],
    )

    assert result.resolved is False
    assert result.status == "ambiguous_reference"
    assert result.retrieval_question == "What technologies does that project use?"


def test_mixed_project_and_technology_lists_support_plural_and_ordinal_project_refs():
    history = [Message(
        "assistant",
        "Projects: KnowSphere AI and Invoice Copilot. Technologies: Angular Material and React Hooks.",
    )]

    plural = resolve_follow_up("Which of those projects uses RAG?", history)
    ordinal = resolve_follow_up("Tell me more about the second one.", history)

    assert plural.retrieval_question == "Which of these projects (KnowSphere AI and Invoice Copilot) uses RAG?"
    assert ordinal.retrieval_question == "Tell me more about Invoice Copilot."


def test_clarification_reply_resolves_the_pending_question_to_a_standalone_query():
    original_reply = "I mean KnowSphere AI."
    history = [
        Message("assistant", "The resume mentions two projects: AI-Powered Quiz Generator and KnowSphere AI."),
        Message("user", "What technologies does that project use?"),
        Message(
            "assistant", "I couldn't determine what you're referring to. Could you clarify?",
            is_clarification=True,
        ),
    ]

    result = resolve_follow_up(original_reply, history)

    assert result.original_question == original_reply
    assert result.retrieval_question == "What technologies does KnowSphere AI use?"
    assert result.resolved is True
    assert result.status == "resolved_clarification"


def test_short_ordinal_clarification_selects_from_the_prior_explicit_project_list():
    result = resolve_follow_up(
        "The second one.",
        [
            Message("assistant", "Projects: AI-Powered Quiz Generator and KnowSphere AI."),
            Message("user", "What technologies does that project use?"),
            Message("assistant", "Could you clarify?", is_clarification=True),
        ],
    )

    assert result.retrieval_question == "What technologies does KnowSphere AI use?"
    assert result.original_question == "The second one."
    assert result.status == "resolved_clarification"


def test_explicit_user_clarification_can_name_the_subject_when_prior_list_was_unparseable():
    result = resolve_follow_up(
        "I mean KnowSphere AI.",
        [
            Message("assistant", "The resume includes AI-Powered Quiz Generator and KnowSphere AI."),
            Message("user", "Which of those projects uses RAG?"),
            Message(
                "assistant", "I couldn't determine what you're referring to. Could you clarify?",
                is_clarification=True,
            ),
        ],
    )

    assert result.retrieval_question == "Does KnowSphere AI use RAG?"
    assert result.status == "resolved_clarification"


def test_unrelated_question_after_clarification_remains_standalone():
    question = "What is BM25?"
    result = resolve_follow_up(
        question,
        [
            Message("assistant", "Projects: AI-Powered Quiz Generator and KnowSphere AI."),
            Message("user", "What technologies does that project use?"),
            Message("assistant", "Could you clarify?", is_clarification=True),
        ],
    )

    assert result.retrieval_question == question
    assert result.status == "standalone"


def test_clarification_reply_must_match_an_explicit_project_candidate():
    question = "I mean Angular."
    result = resolve_follow_up(
        question,
        [
            Message("assistant", "Projects: Invoice Copilot and KnowSphere AI."),
            Message("user", "What technologies does that project use?"),
            Message("assistant", "Could you clarify?", is_clarification=True),
        ],
    )

    assert result.retrieval_question == question
    assert result.status == "unresolved_reference"


def test_multiple_clarification_exchanges_use_only_the_latest_exchange():
    question = "The second one."
    result = resolve_follow_up(
        question,
        [
            Message("assistant", "Projects: Old Project and Old Portal."),
            Message("user", "What does that project use?"),
            Message("assistant", "Could you clarify?", is_clarification=True),
            Message("user", "Old Portal."),
            Message("assistant", "Old Portal uses an API."),
            Message("assistant", "Projects: Invoice Copilot and KnowSphere AI."),
            Message("user", "What technologies does that project use?"),
            Message("assistant", "Could you clarify?", is_clarification=True),
        ],
    )

    assert result.retrieval_question == "What technologies does KnowSphere AI use?"
    assert result.status == "resolved_clarification"
