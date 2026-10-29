# Contributing to AgentXRay

Thank you for your interest in contributing to **AgentXRay**! AgentXRay is an open-source diagnostic instrument designed to isolate and minimize causal failure conditions in tool-calling AI agents using black-box Delta Debugging and counterfactual verification.

We welcome bug fixes, documentation improvements, benchmark testbed contributions, and SDK enhancements.

---

## Core Design Principles

When contributing to AgentXRay, please align your changes with our architectural foundations:

1. **Strict Black-Box Boundary**: The diagnostic engine treats agents and tools as black boxes. We inspect only observable inputs, tool calls, and outputs. Never add logic requiring access to internal model weights, hidden token activations, or proprietary system instructions.
2. **Zero Paid Cloud Spend in Core / Testbeds**: The default testbeds, offline demonstrations, unit test suites, and documentation examples must run 100% offline with zero external network requests and $0.00 spend.
3. **Formal Minimality & Counterfactual Proof**: An isolated failure condition must be 1-minimal. Always verify that removing the isolated candidate directive restores clean task success ($c \setminus \{e\}$).
4. **Immutability & Typing**: Domain records and telemetry use frozen Pydantic models with strict validation (`extra="forbid"`). All public APIs require full Python type annotations.
5. **Restrained Presentation**: CLI commands prioritize clean, legible monochrome/ANSI cards and strict POSIX process exit codes (`0`, `1`, `2`, `3`).

---

## Development Environment Setup

### Prerequisites

- Python `>= 3.12`
- Git

### 1. Clone & Set Up Virtual Environment

```bash
# Clone repository
git clone https://github.com/yash-mishraa/AgentXRay.git
cd AgentXRay

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install editable package with development and demo dependencies
pip install -e ".[dev,demo]"
```

*(Alternatively, using `uv`: `uv venv && source .venv/bin/activate && uv pip install -e ".[dev,demo]"`)*

---

## Running the Test Suite

AgentXRay maintains a comprehensive test suite across unit, integration, and statistical conformance tests.

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_api.py tests/test_cli.py

# Run with verbose output
pytest -v
```

Before submitting any Pull Request, ensure that all tests pass cleanly:
```bash
pytest tests/
# Output: 640 passed in ~2.8s
```

---

## Running the Developer Tools

### 1. Developer CLI

```bash
# Run help
agentxray --help

# Run the deterministic offline verification demo
agentxray demo

# Diagnose an example declarative scenario
agentxray diagnose examples/scenario_example.json

# Check package version
agentxray version
```

### 2. Standalone Product Website

```bash
# Build the multi-page static site
npx --yes vite build web

# Or preview locally with standard Python HTTP server
python3 -m http.server --directory web 3000
# Open http://localhost:3000 in your browser
```

### 3. Gradio Web App (Hugging Face Spaces Default)

```bash
python app.py
# Open http://localhost:7860 in your browser
```

---

## Contribution Workflow

1. **Fork & Branch**: Create a descriptive branch from `main`:
   ```bash
   git checkout -b fix/evaluator-timeout-handling
   ```
2. **Implement**: Keep your changes focused and minimal. Avoid adding external dependencies for trivial helpers.
3. **Test**: Write tests covering your changes in `tests/test_<feature>.py`.
4. **Format & Verify**: Ensure type annotations are valid and existing test coverage passes.
5. **Commit**: Use clear, conventional commit messages:
   - `feat(...)`: New feature or SDK capability
   - `fix(...)`: Bug fix
   - `docs(...)`: Documentation updates
   - `test(...)`: Test suite improvements
   - `refactor(...)`: Code cleanup without semantic change
6. **Pull Request**: Open a PR against `main`. Include a clear description of:
   - The problem or use case
   - The minimal change implemented
   - How it was verified

---

## License

By contributing to AgentXRay, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
