# Convenience Makefile for the public PYTHIA analysis.
#
# The thesis work was run with explicit local compile/run commands. This file
# wraps those commands so a new reader can build and run a small smoke test with
# one command.

SHELL := /bin/bash

REPO_DIR := $(CURDIR)

ifeq ($(origin CXX),default)
CXX := g++
endif
CXXFLAGS ?= -O2 -std=c++17

PYTHIA ?= /Users/shanesweetman/Downloads/pythia/pythia8315
FJ ?= /Users/shanesweetman/Downloads/fastjet

ROOT_CONFIG ?= root-config
FASTJET_CONFIG ?= $(FJ)/bin/fastjet-config

SRC ?= src/pythia/Pythia1.cc
BIN ?= TMD
OUT ?= output.root
PROGRESS ?= progress.txt

EVENTS ?= 10000
SEED ?= 12345

ROOT_CFLAGS := $(shell $(ROOT_CONFIG) --cflags 2>/dev/null)
ROOT_LIBS := $(shell $(ROOT_CONFIG) --libs 2>/dev/null)
FASTJET_CXXFLAGS := $(shell "$(FASTJET_CONFIG)" --cxxflags 2>/dev/null)
FASTJET_LIBS := $(shell "$(FASTJET_CONFIG)" --libs 2>/dev/null)

.DEFAULT_GOAL := help

.PHONY: help check-deps build run test smoke open clean print-config

help:
	@echo "TMD-Analysis Make targets"
	@echo
	@echo "  make build        Compile the PYTHIA analysis"
	@echo "  make test         Build and run a small 10k-event smoke test"
	@echo "  make run          Build and run with EVENTS=$${EVENTS:-$(EVENTS)}"
	@echo "  make open         Open OUT=$${OUT:-$(OUT)} in a ROOT TBrowser"
	@echo "  make clean        Remove local build/output files"
	@echo "  make print-config Show the paths and run settings"
	@echo
	@echo "Common overrides:"
	@echo "  make test PYTHIA=/path/to/pythia8315 FJ=/path/to/fastjet"
	@echo "  make run EVENTS=100000 SEED=12345 OUT=my_output.root"

check-deps:
	@command -v "$(ROOT_CONFIG)" >/dev/null 2>&1 || { \
		echo "ERROR: $(ROOT_CONFIG) not found. Load ROOT or set ROOT_CONFIG=/path/to/root-config."; \
		exit 1; \
	}
	@test -x "$(FASTJET_CONFIG)" || { \
		echo "ERROR: FastJet config not found: $(FASTJET_CONFIG)"; \
		echo "Set FJ=/path/to/fastjet."; \
		exit 1; \
	}
	@test -d "$(PYTHIA)/include" || { \
		echo "ERROR: PYTHIA include directory not found: $(PYTHIA)/include"; \
		echo "Set PYTHIA=/path/to/pythia8315."; \
		exit 1; \
	}
	@test -d "$(PYTHIA)/lib" || { \
		echo "ERROR: PYTHIA library directory not found: $(PYTHIA)/lib"; \
		echo "Set PYTHIA=/path/to/pythia8315."; \
		exit 1; \
	}

build: $(BIN)

$(BIN): $(SRC) | check-deps
	@echo "Compiling $(SRC) -> $(BIN)"
	@unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES; \
	$(CXX) $(CXXFLAGS) \
		-I"$(PYTHIA)/include" \
		$(ROOT_CFLAGS) \
		$(FASTJET_CXXFLAGS) \
		"$(SRC)" -o "$(BIN)" \
		-L"$(PYTHIA)/lib" -lpythia8 \
		$(ROOT_LIBS) \
		$(FASTJET_LIBS) \
		-Wl,-rpath,"$(PYTHIA)/lib" -Wl,-rpath,"$(FJ)/lib"

run: build
	@echo "Running $(EVENTS) events with seed $(SEED)"
	@env DYLD_LIBRARY_PATH="$(PYTHIA)/lib:$(FJ)/lib" \
		DYLD_FALLBACK_LIBRARY_PATH="$(PYTHIA)/lib:$(FJ)/lib" \
		"./$(BIN)" "$(EVENTS)" "$(SEED)" "$(OUT)" "$(PROGRESS)"
	@echo "Wrote $(OUT)"

test: EVENTS = 10000
test: OUT = output.root
test: PROGRESS = progress.txt
test: run

smoke: test

open:
	@test -f "$(OUT)" || { echo "ERROR: ROOT file not found: $(OUT)"; exit 1; }
	@root -l -e 'TFile::Open("$(OUT)"); new TBrowser();'

print-config:
	@echo "REPO_DIR        = $(REPO_DIR)"
	@echo "CXX             = $(CXX)"
	@echo "CXXFLAGS        = $(CXXFLAGS)"
	@echo "PYTHIA          = $(PYTHIA)"
	@echo "FJ              = $(FJ)"
	@echo "ROOT_CONFIG     = $(ROOT_CONFIG)"
	@echo "FASTJET_CONFIG  = $(FASTJET_CONFIG)"
	@echo "SRC             = $(SRC)"
	@echo "BIN             = $(BIN)"
	@echo "OUT             = $(OUT)"
	@echo "EVENTS          = $(EVENTS)"
	@echo "SEED            = $(SEED)"

clean:
	rm -f "$(BIN)" "$(OUT)" "$(PROGRESS)"
