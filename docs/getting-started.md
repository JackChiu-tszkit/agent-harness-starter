# Getting started

Use Python 3.11 or newer. The automated tests in this repository were run with
Python 3.13. The Docker image uses Python 3.11.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python examples/calculator_agent.py
```

The examples always construct a mock-model runtime. They do not read an API key.

To start the API with the same default:

```bash
uvicorn agent_harness.api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI UI.

Copy `.env.example` if you want to change ports, limits, or the model provider.
Export the variables before starting the process. The server reads the
environment when it is imported.

The OpenAI-compatible provider is optional. Leave
`AGENT_HARNESS_MODEL_PROVIDER=mock` unless `AGENT_HARNESS_OPENAI_API_KEY` is
set in the environment. The key is not written into the repository.
