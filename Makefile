documentation:
	sphinx-build -b html ./docs ./docs/_build
open-docs:
	xdg-open docs/_build/index.html
