#!/bin/bash
# Build script for TBExciton90 manual

echo "==============================================="
echo "Building TBExciton90 Technical Manual"
echo "==============================================="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found!"
    echo "Please install LaTeX (e.g., texlive-full on Ubuntu/Debian)"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  CentOS/RHEL:   sudo yum install texlive-scheme-full"
    echo "  macOS:         brew install --cask mactex"
    exit 1
fi

echo "✓ pdflatex found"

# Check for required LaTeX packages
echo "Checking LaTeX installation..."

# Create a minimal test document
cat > test_latex.tex << 'EOF'
\documentclass{article}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}
\begin{document}
Test
\end{document}
EOF

# Test compilation
if pdflatex -interaction=nonstopmode test_latex.tex > /dev/null 2>&1; then
    echo "✓ LaTeX packages available"
    rm -f test_latex.*
else
    echo "⚠ Some LaTeX packages may be missing"
    echo "Consider installing additional packages if compilation fails"
    rm -f test_latex.*
fi

# Build the manual
echo ""
echo "Building manual..."
echo "=================="

if make all; then
    echo ""
    echo "✓ Manual built successfully!"
    echo "📖 Output: TBExciton90_Manual.pdf"
    
    # Check file size
    if [ -f "TBExciton90_Manual.pdf" ]; then
        SIZE=$(du -h TBExciton90_Manual.pdf | cut -f1)
        echo "📊 File size: $SIZE"
    fi
    
    echo ""
    echo "To view the manual:"
    echo "  make view"
    echo "  # or open TBExciton90_Manual.pdf manually"
    
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the LaTeX log for errors:"
    echo "  cat TBExciton90_Manual.log"
    exit 1
fi

echo ""
echo "==============================================="