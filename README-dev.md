## Developer README
* Run `$ make documentation` to build the documentation as well as the coverage badge.
* Check `docs/docstr-coverage-output.txt` for the output by the [coverage checker](https://github.com/HunterMcGushion/docstr_coverage).

## Setup history
* Applied sphinx-apidoc template based on [this SO question](https://stackoverflow.com/questions/50361218/remove-the-word-module-from-sphinx-documentation).
  The template files are stored in `docs/apidoc_templates`.
* `--no-toc` option is used for sphinx-apidoc. This suppresses overwriting `modules.rst`.
  We maintain the file (containing the table of contents) by hand.
* `docs/index.rst` and `docs/modules.rst` are copied to `source/` and then used. Make sure to edit the ones in `docs/`.

## Requirements
* `$ pip install -r requirements-dev.txt`
