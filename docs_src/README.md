## Install
```bash
$ pip install sphinx
$ pip install sphinx_rtd_theme
```

## Generate docs
The following commands are run by the Makefile in the root directory of this package.

```bash
$ sphinx-apidoc -f -o ./docs .
$ sphinx-build -b html ./docs ./docs/_build
```

## Extensions
* Autodoc
* Napoleon https://www.sphinx-doc.org/en/1.5/ext/napoleon.html
* Mathjax https://www.sphinx-doc.org/en/1.5/ext/math.html#module-sphinx.ext.mathjax
* Doctest https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html
* (Theme: `sphinx_rtd_theme`)

## Setup history
* Followed the instruction here for setup: https://qiita.com/futakuchi0117/items/4d3997c1ca1323259844
