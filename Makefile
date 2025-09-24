.PHONY: all clean buildlib wheel pub mac-arm64 docs serve-docs docs-serve

buildlib:
	@echo "Building library..."
	@cd chdb && bash build.sh
	@echo "Done."

wheel:
	@echo "Building wheel..."
	tox -e build -- --wheel
	@echo "Done."

test:
	@echo "Testing..."
	cd tests && python3 run_all.py

pub:
	@echo "Publishing wheel..."
	tox -e publish
	@echo "Done."

clean:
	@echo "Cleaning..."
	tox -e clean
	@echo "Done."

mac-arm64:
	@echo "Make macOS arm64 whl"
	chdb/build_mac_arm64.sh
	@echo "Done."

linux-arm64:
	@echo "Make linux arm64 whl"
	chdb/build_linux_arm64.sh
	@echo "Done."

build: clean buildlib wheel

docs:
	@echo "Building documentation..."
	@PYTHONPATH=src sphinx-build -b html docs buildlib/docs --keep-going -v -E
	@echo "Documentation built in buildlib/docs/"
	@echo "Starting documentation server on port 8000..."
	@(sleep 1; python3 -c "import webbrowser; webbrowser.open('http://127.0.0.1:8001/')" 2>/dev/null &)
	@cd buildlib/docs && python3 -m http.server 8001
