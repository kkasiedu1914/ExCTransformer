# ExCTransformer v1.0.1 — 25 October 2026

- Includes the installed `exct-download` argument-forwarding fix and regression tests merged in PR #1.
- Makes the fixed code the stable release, with the public research-asset instructions and citation metadata.
- Adds a v1.0.1 source/packaged asset manifest so default downloads work after the software version bump. The research files and checksums are unchanged.
- Updates example cache paths and pins the installation instructions to v1.0.1.
- Archives this release at https://doi.org/10.5281/zenodo.23648584. The project concept DOI is https://doi.org/10.5281/zenodo.20777272. The associated paper DOI will be added when assigned.

The bounded download audit verified all nine public assets, with five complete checksum-verified small/example downloads, four 64 KB transfers for larger files, archive extraction, cache reuse, checksum rejection, and batch selection. Full checkpoint loading and training were outside that audit.
