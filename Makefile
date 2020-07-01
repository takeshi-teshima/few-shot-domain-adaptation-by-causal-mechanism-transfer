documentation:
	rm -r docs/temp_src &
	mkdir docs/temp_src; mkdir docs/temp_src/_static
	cp -f docs/conf.py docs/temp_src/
	cp -f docs/index.rst docs/temp_src/
	cp -f docs/modules.rst docs/temp_src/
	sphinx-apidoc --no-toc --templatedir ./docs/apidoc_templates -f -o ./docs/temp_src . setup.py
	sphinx-build -b html ./docs/temp_src ./docs/_build
	docstr-coverage causal_da --failunder=0 --skipmagic --skipfiledoc --badge ./docs/coverage_badge.svg  2>&1 | tee docs/docstr-coverage-output.txt

install-dev:
	pip install .

open-docs:
	xdg-open docs/_build/index.html
