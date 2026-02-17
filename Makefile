# Convenience Makefile for common tasks

BUILD_DIR := build
CMAKE := cmake
CMAKE_BUILD := $(CMAKE) --build $(BUILD_DIR) -- -j

.PHONY: all configure build test run format tidy clean distclean

all: build

configure:
	@mkdir -p $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR) -DEXECUTORCH_VISION_ENABLE_TESTS=ON

build: configure
	@$(CMAKE_BUILD)

test: build
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	@$(BUILD_DIR)/executorch_vision_bin

format:
	@if command -v clang-format >/dev/null 2>&1; then \
	  clang-format -i $(shell git ls-files '*.cpp' '*.h'); \
	else \
	  echo "clang-format not found"; exit 1; \
	fi

tidy:
	@if command -v clang-tidy >/dev/null 2>&1; then \
	  clang-tidy src/*.cpp -- -Iinclude -p $(BUILD_DIR) || true; \
	else \
	  echo "clang-tidy not found"; exit 1; \
	fi

clean:
	@rm -rf $(BUILD_DIR)/*

distclean: clean
	@rm -rf $(BUILD_DIR)
