# AlphaZero CPU inference dependencies

The default API image stays torch-free. `ALPHAZERO_RUNTIME=true` opts into an
inference runtime; `ALPHAZERO_ENABLED=true` is a separate runtime choice. The
production deployment script overrides both according to its explicit mode.

For CPython 3.11 on **linux/arm64**, `requirements-alphazero-arm64.lock` pins
CPU-only torch 2.10.0+cpu and all eight applicable transitive dependencies.
The versions equal the Phase 4D.4K-B tested staging runtime; this phase does not
upgrade them. Hashes were obtained from an actual pip resolver report running
inside the native ARM64 image, then validated by downloading/reinstalling with
`--only-binary=:all: --force-reinstall --require-hashes` in the image build.
Force reinstall also verifies the wheels for dependencies already present from
the ordinary API install (Jinja2 and MarkupSafe).

The torch wheel is `torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl`.
Its SHA-256 is `1cfcb9b1558c6e52dffd0d4effce83b13c5ae5d97338164c372048c21f9cfccb`.
The [official CPU index](https://download.pytorch.org/whl/cpu/torch/) resolves to
PyTorch's `download-r2.pytorch.org` delivery host. The additional
[PyPI index](https://pypi.org/simple/) supplies the exact existing MarkupSafe
3.0.4 ARM64 wheel and pure-Python dependencies. Every accepted candidate must
match its pinned digest; adding an index does not permit an unlisted wheel.

## Updating the lock

1. Use an isolated CPython 3.11 **native linux/arm64** container, with no
   production credentials or training/research files. Record the old runtime
   versions before choosing an update. Do not guess platform hashes.
2. Resolve explicit versions with:

   ```sh
   python -m pip install --dry-run --ignore-installed \
     --report /tmp/alphazero-arm64-resolution.json \
     --index-url https://download.pytorch.org/whl/cpu \
     --extra-index-url https://pypi.org/simple \
     'torch==2.10.0+cpu' 'MarkupSafe==3.0.4' 'filelock==3.32.3' \
     'fsspec==2026.7.0' 'networkx==3.6.1' 'sympy==1.14.0' \
     'mpmath==1.3.0' 'Jinja2==3.1.6' 'typing_extensions==4.16.0'
   ```

3. For each report `install` entry, verify the name/version, upstream URL,
   wheel filename and `download_info.archive_info.hashes.sha256`. Binary
   torch/MarkupSafe wheels must target cp311/aarch64; other wheels are
   platform-independent. Translate all nine entries into exact pinned
   requirements with `--hash=sha256:<digest>`; retain wheel filenames in comments.
   Independently compare downloaded bytes with `python -m pip hash <wheel>`.
4. Build both ARM64 modes. Check no torch in lite, CPU-only torch/import/thread
   settings in enabled mode, artifact integrity, tactical guard and API tests,
   real bounded moves, and the browser/export suites. The dedicated workflow's
   path filters include the lock. Request review before adopting changed bytes.

## Limits

This lock is intentionally specific to Python 3.11 Linux ARM64. The existing
optional AMD64 build retains `requirements-alphazero.txt` and version-only CPU
resolution; it is not the Mac mini release target. Neither this lock nor a Git
commit-specific image tag makes the entire OS image reproducible: the existing
`python:3.11-slim` base, apt packages, and ordinary API dependency set are not
fully hash/digest locked. That broader dependency-management rewrite is outside
this phase. Hashes establish expected downloaded bytes, not an independent
signature or security audit of upstream packages.
