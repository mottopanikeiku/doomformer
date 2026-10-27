# Security

Doomformer runs locally. It reads model files shipped in the package (NumPy `.npz` and JSON), writes maps and a cache
under `~/.cache/doomformer`, and on first use clones and builds the ACS compiler ([ACC](https://github.com/rheit/acc))
at a pinned commit. `doomformer mnist --all` downloads the MNIST files over HTTPS.

Report a vulnerability with [GitHub's private reporting form](https://github.com/mottopanikeiku/doomformer/security/advisories/new).
Only the latest release is supported.
