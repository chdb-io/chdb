.PHONY: all clean buildlib wheel pub mac-arm64

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

libchdb-mac-arm64:
	@echo "Make macOS arm64 libchdb.so"
	chdb/build_libchdb_mac_arm64.sh
	@echo "Done."

linux-arm64:
	@echo "Make linux arm64 whl"
	chdb/build_linux_arm64.sh
	@echo "Done."

build: clean buildlib wheel
