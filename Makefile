documentation:
	sphinx-apidoc -f -o ./docs .
	sphinx-build -b html ./docs ./docs/_build
	docstr-coverage causal_da --badge ./docs/coverage_badge.svg

open-docs:
	xdg-open docs/_build/index.html
