.PHONY: all clean buildlib wheel pub

buildlib:
	@echo "Building library..."
	@cd chdb && bash build.sh
	@echo "Done."

wheel:
	@echo "Building wheel..."
	tox -e build -- --wheel
	@echo "Done."

pub:
	@echo "Publishing wheel..."
	tox -e publish
	@echo "Done."

clean:
	@echo "Cleaning..."
	tox -e clean
	@echo "Done."


build: clean buildlib wheel
