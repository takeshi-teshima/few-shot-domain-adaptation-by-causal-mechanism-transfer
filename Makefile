documentation:
	mkdir -p docs/source/_static
	cp -f docs/conf.py docs/source/
	cp -f docs/index.rst docs/source/
	cp -f docs/modules.rst docs/source/
	sphinx-apidoc --no-toc --templatedir ./docs/apidoc_templates -f -o ./docs/source . setup.py
	sphinx-build -b html ./docs/source ./docs/_build
	docstr-coverage causal_da --failunder=0 --skipmagic --skipfiledoc --badge ./docs/coverage_badge.svg  2>&1 | tee docs/docstr-coverage-output.txt

open-docs:
	xdg-open docs/_build/index.html
