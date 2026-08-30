"""
DevFit test suite root.

Test organisation mirrors the source layout under ``src/devfit/``.

Directories
-----------
tests/
├── conftest.py          shared fixtures
├── test_schema.py       Pydantic model validation
├── verifier/
│   └── test_rules.py    deterministic rule layer (no LLM, no network)
├── github/
│   └── test_bundle.py   ArtefactBundle unit tests
│   └── test_client.py   GitHubClient unit tests (httpx mock transport)
└── integration/         marked with @pytest.mark.integration
    └── test_collector.py real GitHub API smoke test

Run commands
------------
uv run pytest                           # all unit tests
uv run pytest -m "not integration"      # skip network tests
uv run pytest tests/verifier/test_rules.py::test_date_rule_contradicts
uv run pytest --cov=src/devfit --cov-report=term-missing
"""
