# Publishing `pybayescointur` to PyPI

A short, repeatable checklist. Build artifacts already validated with
`twine check` (both the wheel and the sdist PASSED).

## 0. One-time setup

1. **Check the name is free.** Open <https://pypi.org/project/pybayescointur/>.
   A `404` means the name is available.
2. **Create accounts** on <https://pypi.org> and (for the dry run)
   <https://test.pypi.org>.
3. **Create an API token**: PyPI → *Account settings* → *API tokens* →
   *Add API token* (scope "Entire account" for the first upload, then narrow it
   to the project afterwards). Copy the token (starts with `pypi-...`).

## 1. (Re)build the distributions

> **Always delete the old `dist/` first.** Uploading a stale artifact built
> before the metadata was finalised causes
> `InvalidDistribution: ... unrecognized or malformed field 'license-file'`.
> A clean rebuild with up-to-date `build`/`twine`/`setuptools` fixes it.

```bash
cd C:\Users\HP\Documents\xtpmg\pybayesurcoint
python -m pip install --upgrade build twine setuptools packaging
# PowerShell:
Remove-Item -Recurse -Force dist, build, pybayescointur.egg-info -ErrorAction SilentlyContinue
# bash:
# rm -rf dist build pybayescointur.egg-info
python -m build
python -m twine check dist/*        # must say PASSED for both files
```

The metadata should read `Metadata-Version: 2.4` / `License-Expression: MIT`
(this package uses the modern PEP 639 license declaration).

## 2. Dry run on TestPyPI (recommended)

```bash
python -m twine upload --repository testpypi dist/*
# username:  __token__
# password:  <your TestPyPI token>

# then install from TestPyPI into a clean environment to confirm it works:
python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ pybayescointur
python -c "import pybayescointur as p; print(p.__version__)"
```

## 3. Real upload to PyPI

```bash
python -m twine upload dist/*
# username:  __token__
# password:  <your PyPI token>
```

Then verify:

```bash
pip install pybayescointur
python -c "import pybayescointur as p; print(p.__version__)"
```

## Tips / gotchas

- **A version number can never be reused.** If you need to fix anything after a
  release, bump the version in **both** `pyproject.toml` and
  `pybayescointur/__init__.py` (e.g. `0.1.1`) and rebuild.
- Store the token in `~/.pypirc` or the `TWINE_USERNAME=__token__` /
  `TWINE_PASSWORD=pypi-...` environment variables to avoid retyping.
- The README images use absolute `raw.githubusercontent.com` URLs so they render
  on the PyPI project page **once the GitHub repo exists and `main` is pushed**.
- Optional: add a GitHub Actions workflow to publish automatically on tag using
  PyPI *Trusted Publishing* (no token needed).
```

### Optional: trusted-publishing GitHub Action
Create `.github/workflows/publish.yml`, register the repo as a Trusted Publisher
on PyPI, and a `git tag v0.1.0 && git push --tags` will build and publish.
