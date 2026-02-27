.PHONY: all clean wheel pub test docs docs-md

wheel:
	@echo "Building wheel..."
	tox -e build -- --wheel
	@echo "Done."

test:
	@echo "Testing DataStore..."
	cd datastore && python3 -m pytest tests/ -v --tb=short

pub:
	@echo "Publishing wheel..."
	tox -e publish
	@echo "Done."

clean:
	@echo "Cleaning..."
	tox -e clean
	@echo "Done."

docs:
	@echo "Building documentation..."
	@PYTHONPATH=src sphinx-build -b html docs buildlib/docs --keep-going -v -E
	@echo "Documentation built in buildlib/docs/"
	@echo "Starting documentation server on port 8000..."
	@(sleep 1; python3 -c "import webbrowser; webbrowser.open('http://127.0.0.1:8001/')" 2>/dev/null &)
	@cd buildlib/docs && python3 -m http.server 8001

docs-md:
	@echo "Building markdown documentation..."
	@PYTHONPATH=src sphinx-build -b markdown docs buildlib/markdowndocs --keep-going -v -E
	@echo "Markdown documentation built in buildlib/markdowndocs/"
