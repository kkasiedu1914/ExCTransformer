# Release validation — 2026-10-25

The release is prepared for Python 3.11 on Linux. Validation includes:

- Building a wheel and source distribution and passing `twine check`.
- Verifying packaged atom features, asset manifests, compatibility imports, and upstream licenses.
- 15 passing helper tests covering manifests, downloads, checksums, CLI calls, and archive extraction.
- Installing the wheel with the documented constraints in a fresh virtual environment; `pip check` reports no broken requirements.
- Running the installed package outside the checkout: CIF generation, graph construction, one-epoch CPU training, checkpoint reload, prediction, WLS_Ridge GraphSVX attribution, and plot export.
- Running build/install/tests and the full CPU tutorial independently on GitHub Actions.
- Checking repository text for embedded credentials and validating local documentation links.

The test CSV fixtures contain only three data rows each. The repository retains
small visualization examples; full research datasets and model checkpoints are
managed through the external asset manifest.

This validates software installation and workflow operation. It does not re-run
the manuscript benchmarks or validate GPU performance. The supplied CPU tutorial
uses synthetic structures and targets.

The GitHub repository, Zenodo software archives, and Figshare research assets are
public. Users can install the package and run the asset download commands without
an account or credentials. The current release passes 15 helper tests, package and
citation validation, and the installed CLI/CPU workflow on GitHub Actions.
