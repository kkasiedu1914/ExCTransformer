# Contributing

Open an issue with a minimal example, expected and observed behavior, Python and
dependency versions, and whether you used CPU or GPU. Include the relevant
configuration and a small synthetic input when possible.

For development, use Python 3.11, install `pip install -e ".[dev]"`, and run
`python -m pytest tests`. Run `python examples/cpu_quickstart.py --output /tmp/exct-demo`
for changes affecting graph construction, training, or prediction; use a fresh
output directory each time. Keep datasets, credentials, checkpoints, and generated
outputs out of commits. Add focused tests for behavior changes.

Pull requests should describe the problem, the resulting behavior, and validation.
Contributions are provided under the repository's license, with upstream notices
preserved for adapted code.
