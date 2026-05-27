# Makefile for the public PYTHIA analysis.
#
# The original thesis runs used explicit local ROOT/PYTHIA/FastJet commands.
# This file keeps the same build logic in one place so the repository can be
# cloned, configured once, and run from a normal terminal.

SHELL := /bin/bash

REPO_DIR := $(CURDIR)
LOCAL_CONFIG ?= local.mk

-include $(LOCAL_CONFIG)

ifeq ($(origin CXX),default)
CXX := g++
endif
CXXFLAGS ?= -O2 -std=c++17

ROOT_CONFIG ?= root-config
FASTJET_CONFIG ?= $(firstword $(wildcard $(HOME)/Downloads/fastjet/bin/fastjet-config) $(wildcard $(HOME)/downloads/fastjet/bin/fastjet-config) fastjet-config)
PYTHIA8_CONFIG ?= $(firstword $(wildcard $(HOME)/Downloads/pythia/pythia*/bin/pythia8-config) $(wildcard $(HOME)/downloads/pythia/pythia*/bin/pythia8-config) pythia8-config)

SRC ?= src/pythia/Pythia1.cc
BIN ?= tmd-pythia
OUT ?= output.root
PROGRESS ?= progress.txt

EVENTS ?= 10000
SEED ?= 12345
PLOT_TAG ?= public
THEORY_FILE ?= data/theory/epemCrossSection_z0p70.dat

ROOT_CFLAGS := $(shell $(ROOT_CONFIG) --cflags 2>/dev/null)
ROOT_LIBS := $(shell $(ROOT_CONFIG) --libs 2>/dev/null)
FASTJET_CXXFLAGS := $(shell "$(FASTJET_CONFIG)" --cxxflags 2>/dev/null)
FASTJET_LIBS := $(shell "$(FASTJET_CONFIG)" --libs 2>/dev/null)
FASTJET_PREFIX := $(shell "$(FASTJET_CONFIG)" --prefix 2>/dev/null)
PYTHIA_CXXFLAGS_RAW := $(shell "$(PYTHIA8_CONFIG)" --cxxflags 2>/dev/null)
PYTHIA_CXXFLAGS := $(filter-out -O% -std=% -pedantic -W -Wall -Wshadow,$(PYTHIA_CXXFLAGS_RAW))
PYTHIA_LIBS := $(shell "$(PYTHIA8_CONFIG)" --libs 2>/dev/null)
PYTHIA_PREFIX := $(shell "$(PYTHIA8_CONFIG)" --prefix 2>/dev/null)

.DEFAULT_GOAL := all

.PHONY: all install help check-deps build run open plots theory-overlay clean print-config

all: $(BIN)

install: all

help:
	@echo "TMD-Analysis Make targets"
	@echo
	@echo "  make              Compile the PYTHIA analysis"
	@echo "  make build        Same as make"
	@echo "  make run          Build and run with EVENTS=$${EVENTS:-$(EVENTS)}"
	@echo "  make open         Open OUT=$${OUT:-$(OUT)} in a ROOT TBrowser"
	@echo "  make plots        Regenerate selected ROOT plots from OUT=$${OUT:-$(OUT)}"
	@echo "  make theory-overlay  Overlay PYTHIA histograms with included TMD theory"
	@echo "  make clean        Remove local build/output files"
	@echo "  make print-config Show the paths and run settings"
	@echo
	@echo "First-time local setup:"
	@echo "  cp config/local.mk.example local.mk"
	@echo "  edit local.mk if your dependency paths differ"
	@echo
	@echo "Standard run:"
	@echo "  make"
	@echo "  make run EVENTS=10000 SEED=12345 OUT=output.root PROGRESS=progress.txt"
	@echo
	@echo "Common overrides:"
	@echo "  make run PYTHIA8_CONFIG=/path/to/pythia8315/bin/pythia8-config"
	@echo "  make run EVENTS=100000 SEED=12345 OUT=my_output.root"

check-deps:
	@test -x "$(ROOT_CONFIG)" || command -v "$(ROOT_CONFIG)" >/dev/null 2>&1 || { \
		echo "ERROR: $(ROOT_CONFIG) not found. Load ROOT or set ROOT_CONFIG=/path/to/root-config in local.mk."; \
		exit 1; \
	}
	@test -x "$(FASTJET_CONFIG)" || command -v "$(FASTJET_CONFIG)" >/dev/null 2>&1 || { \
		echo "ERROR: FastJet config not found: $(FASTJET_CONFIG)"; \
		echo "Set FASTJET_CONFIG=/path/to/fastjet-config in local.mk."; \
		exit 1; \
	}
	@test -x "$(PYTHIA8_CONFIG)" || command -v "$(PYTHIA8_CONFIG)" >/dev/null 2>&1 || { \
		echo "ERROR: PYTHIA 8 config not found: $(PYTHIA8_CONFIG)"; \
		echo "Set PYTHIA8_CONFIG=/path/to/pythia8-config in local.mk."; \
		exit 1; \
	}
	@test -n "$(PYTHIA_PREFIX)" || { \
		echo "ERROR: Could not determine PYTHIA prefix from $(PYTHIA8_CONFIG)."; \
		exit 1; \
	}
	@test -n "$(FASTJET_PREFIX)" || { \
		echo "ERROR: Could not determine FastJet prefix from $(FASTJET_CONFIG)."; \
		exit 1; \
	}

build: all

$(BIN): $(SRC) | check-deps
	@echo "Compiling $(SRC) -> $(BIN)"
	@unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES; \
	$(CXX) \
		$(PYTHIA_CXXFLAGS) \
		$(ROOT_CFLAGS) \
		$(FASTJET_CXXFLAGS) \
		$(CXXFLAGS) \
		"$(SRC)" -o "$(BIN)" \
		$(PYTHIA_LIBS) \
		$(ROOT_LIBS) \
		$(FASTJET_LIBS)

run: build
	@echo "Running $(EVENTS) events with seed $(SEED)"
	@unset DYLD_FRAMEWORK_PATH DYLD_INSERT_LIBRARIES; \
	env DYLD_LIBRARY_PATH="$(PYTHIA_PREFIX)/lib:$(FASTJET_PREFIX)/lib" \
		DYLD_FALLBACK_LIBRARY_PATH="$(PYTHIA_PREFIX)/lib:$(FASTJET_PREFIX)/lib" \
		"./$(BIN)" "$(EVENTS)" "$(SEED)" "$(OUT)" "$(PROGRESS)"
	@echo "Wrote $(OUT)"

open:
	@test -f "$(OUT)" || { echo "ERROR: ROOT file not found: $(OUT)"; exit 1; }
	@root -l -e 'TFile::Open("$(OUT)"); new TBrowser();'

plots:
	@test -f "$(OUT)" || { echo "ERROR: ROOT file not found: $(OUT)"; exit 1; }
	@root -l -b -q 'tools/plotting/make_100M_canvases.C("$(OUT)","$(PLOT_TAG)")'
	@root -l -b -q 'tools/plotting/ratio_vs_cut_from_tree.C("$(OUT)","$(PLOT_TAG)")'

theory-overlay:
	@test -f "$(OUT)" || { echo "ERROR: ROOT file not found: $(OUT)"; exit 1; }
	@test -f "$(THEORY_FILE)" || { echo "ERROR: Theory file not found: $(THEORY_FILE)"; exit 1; }
	@root -l -b -q 'tools/plotting/figure20_peakmatch_overlay.C("$(OUT)","$(THEORY_FILE)","$(PLOT_TAG)")'

print-config:
	@echo "REPO_DIR        = $(REPO_DIR)"
	@echo "LOCAL_CONFIG    = $(LOCAL_CONFIG)"
	@echo "CXX             = $(CXX)"
	@echo "CXXFLAGS        = $(CXXFLAGS)"
	@echo "ROOT_CONFIG     = $(ROOT_CONFIG)"
	@echo "FASTJET_CONFIG  = $(FASTJET_CONFIG)"
	@echo "FASTJET_PREFIX  = $(FASTJET_PREFIX)"
	@echo "PYTHIA8_CONFIG  = $(PYTHIA8_CONFIG)"
	@echo "PYTHIA_PREFIX   = $(PYTHIA_PREFIX)"
	@echo "SRC             = $(SRC)"
	@echo "BIN             = $(BIN)"
	@echo "OUT             = $(OUT)"
	@echo "PROGRESS        = $(PROGRESS)"
	@echo "EVENTS          = $(EVENTS)"
	@echo "SEED            = $(SEED)"
	@echo "PLOT_TAG        = $(PLOT_TAG)"
	@echo "THEORY_FILE     = $(THEORY_FILE)"

clean:
	rm -f "$(BIN)" "$(OUT)" "$(PROGRESS)"
	rm -f c_qT_OSSS_4cuts_pion_counts_*.pdf c_qT_OSSS_4cuts_pion_counts_*.png
	rm -f c_qT_OSSS_4cuts_pion_norm_*.pdf c_qT_OSSS_4cuts_pion_norm_*.png
	rm -f ratio_vs_cut_from_tree*.root ratio_vs_cut_from_tree*.pdf ratio_vs_cut_from_tree*.png
	rm -f tmd_theory_overlay_*.root tmd_theory_overlay_*.pdf tmd_theory_overlay_*.png
	rm -f tmd_theory_overlay_chi2_*.txt
	rm -f figure20_peakmatch_overlay_*.root figure20_peakmatch_overlay_*.pdf figure20_peakmatch_overlay_*.png
	rm -f figure20_peakmatch_overlay_chi2_*.txt
