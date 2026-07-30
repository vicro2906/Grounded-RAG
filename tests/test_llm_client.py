"""Every OpenAI call must go through one endpoint.

CLAUDE.md promises that "every LLM call is encapsulated so it can be switched to a private EU
model without friction". That promise is only true while the clients are built in ONE place —
it was not, and the drift was invisible: a `ChatOpenAI(...)` added straight into a new
retrieval mode keeps working perfectly against the default region, and nothing fails until the
day residency actually matters. Hence the architectural guard below.
"""
import pathlib
import re

import rag

ROOT = pathlib.Path(__file__).resolve().parent.parent
# rag.py legitimately constructs the clients (it IS the factory); the ingestion scripts stay
# standalone by design and read OPENAI_BASE_URL themselves, which the second test checks.
FACTORY = ROOT / "rag.py"
INGESTION = ROOT / "ingestion"

# No space before the paren: prose says «Azure OpenAI (EU, for GDPR)», code says «OpenAI(...)».
_DIRECT_CLIENT = re.compile(r"\b(ChatOpenAI|OpenAIEmbeddings|AsyncOpenAI|OpenAI)\(")


def _project_sources():
    """Every module the guard applies to: the whole tree minus the factory itself, the
    standalone ingestion scripts and the tests.

    Dot-directories are skipped wholesale, not just `.venv`. They hold COPIES of this repo —
    `.claude/worktrees/` when a task runs in its own worktree, `.langgraph_api/` caches — and a
    copy carries a legitimate `rag.py`, which made the guard report the factory as its own
    offender. A guard that cries wolf gets disabled, which is worse than not having it."""
    for path in ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        if path == FACTORY or INGESTION in path.parents or path.parent.name == "tests":
            continue
        yield path


def test_no_module_builds_its_own_openai_client():
    offenders = []
    for path in _project_sources():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _DIRECT_CLIENT.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, (
        "build these through rag.chat_model() / rag.embeddings_model(), or they will ignore "
        "OPENAI_BASE_URL:\n" + "\n".join(offenders))


def test_ingestion_scripts_honour_the_same_endpoint():
    """They are standalone (no app imports), so the guard is that they read the variable."""
    for path in INGESTION.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if _DIRECT_CLIENT.search(source):
            assert "OPENAI_BASE_URL" in source, f"{path.name} builds a client ignoring the endpoint"


def test_factory_passes_the_configured_endpoint(monkeypatch):
    monkeypatch.setattr(rag, "OPENAI_BASE_URL", "https://eu.api.openai.com/v1")
    assert str(rag.chat_model("gpt-4o").openai_api_base) == "https://eu.api.openai.com/v1"
    assert str(rag.embeddings_model().openai_api_base) == "https://eu.api.openai.com/v1"


def test_factory_does_not_impose_sampling_on_its_callers():
    """Centralizing the connection must not quietly change how any model behaves: each caller
    keeps owning its own temperature (0 for the structured steps, 0.2 for generation, the
    library default for the RAGAS judge)."""
    assert rag.chat_model("gpt-4o", temperature=0.2).temperature == 0.2
    assert rag.chat_model("gpt-4o", temperature=0).temperature == 0
