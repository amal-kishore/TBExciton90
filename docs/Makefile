# Makefile for TBExciton90 documentation

MAIN = TBExciton90_Manual
TEX = $(MAIN).tex
PDF = $(MAIN).pdf

# Default target
all: $(PDF)

# Compile PDF
$(PDF): $(TEX)
	@echo "Compiling LaTeX document..."
	pdflatex $(TEX)
	pdflatex $(TEX)  # Second pass for references
	@echo "PDF generated: $(PDF)"

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.toc *.out *.fdb_latexmk *.fls *.synctex.gz

# Clean everything including PDF
distclean: clean
	rm -f $(PDF)

# Quick build (single pass)
quick:
	pdflatex $(TEX)

# View PDF
view: $(PDF)
	@if command -v evince > /dev/null; then \
		evince $(PDF) & \
	elif command -v okular > /dev/null; then \
		okular $(PDF) & \
	elif command -v xpdf > /dev/null; then \
		xpdf $(PDF) & \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile PDF (default)"
	@echo "  quick    - Quick compile (single pass)"
	@echo "  clean    - Remove auxiliary files"
	@echo "  distclean- Remove all generated files"
	@echo "  view     - Open PDF in viewer"
	@echo "  help     - Show this help"

.PHONY: all clean distclean quick view help