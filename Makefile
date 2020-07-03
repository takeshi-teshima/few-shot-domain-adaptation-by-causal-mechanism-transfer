build-full-documentation: install-dev documentation

documentation:
	rm -r docs_src/temp_src &
	mkdir docs_src/temp_src; mkdir docs_src/temp_src/_static
	cp -f docs_src/conf.py docs_src/temp_src/
	cp -f docs_src/index.rst docs_src/temp_src/
	cp -f docs_src/modules.rst docs_src/temp_src/
	sphinx-apidoc --no-toc --templatedir ./docs_src/apidoc_templates -f -o ./docs_src/temp_src . setup.py
	sphinx-build -b html ./docs_src/temp_src ./docs
	docstr-coverage causal_da --failunder=0 --skipmagic --exclude=".*/__init__.py" --skipfiledoc --badge ./docs_src/coverage_badge.svg  2>&1 | tee docs_src/docstr-coverage-output.txt

install-dev:
	pip install .

open-docs:
	xdg-open docs/index.html
