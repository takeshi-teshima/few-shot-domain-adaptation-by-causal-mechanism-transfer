documentation:
	mkdir -p docs/_static
	sphinx-apidoc --templatedir ./docs/apidoc_templates -f -o ./docs . setup.py
	sphinx-build -b html ./docs ./docs/_build
	docstr-coverage causal_da --failunder=0 --skipmagic --skipfiledoc --badge ./docs/coverage_badge.svg  2>&1 | tee docs/docstr-coverage-output.txt

open-docs:
	xdg-open docs/_build/index.html
