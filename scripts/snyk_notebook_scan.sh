#!/bin/bash

# Snyk Jupyter Notebook Scanner
# Scans Jupyter notebooks for security vulnerabilities and exports CVE reports
# 
# Features:
# - Recursive notebook discovery and conversion
# - Dependency vulnerability scanning
# - Code vulnerability scanning  
# - CVE report export with timestamps
# - Proper cleanup while preserving reports

# --- Prerequisites ---
# 1. Snyk CLI: Ensure 'snyk' is installed and authenticated ('snyk auth').
# 2. Python tools: Ensure 'pipreqsnb' and 'jupyter' are installed ('pip install pipreqsnb jupyter').

set -e  # Exit on any error

# --- Setup: Dynamically find the project's root directory ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORTS_DIR="$PROJECT_DIR/security_reports_$TIMESTAMP"
TEMP_DIR=$(mktemp -d)

# Create reports directory
mkdir -p "$REPORTS_DIR"

# --- Function: Add user-site bin directory to PATH ---
add_user_site_to_path() {
    # Dynamically find user-site bin directory and add to PATH if it exists
    local user_bin_dir=""
    
    if command -v python3 &> /dev/null; then
        user_bin_dir=$(python3 -m site --user-base)/bin
    elif command -v python &> /dev/null; then
        user_bin_dir=$(python -m site --user-base)/bin
    fi
    
    if [ -n "$user_bin_dir" ] && [ -d "$user_bin_dir" ]; then
        export PATH="$PATH:$user_bin_dir"
    fi
}

echo "📂 Script located in: $SCRIPT_DIR"
echo "🎯 Project root set to: $PROJECT_DIR"
echo "📊 Reports will be saved to: $REPORTS_DIR"
echo "🚀 Starting recursive Jupyter Notebook security scan..."

# --- Step 1: Find and convert notebooks ---
echo ""
echo "🔎 Step 1/4: Finding and converting notebooks to Python scripts..."

# Find all notebooks and store paths in array (compatible with older bash)
notebooks=()
while IFS= read -r -d $'\0' notebook; do
    notebooks+=("$notebook")
done < <(find "$PROJECT_DIR" -name "*.ipynb" -print0)

if [ ${#notebooks[@]} -eq 0 ]; then
    echo "⚠️ No .ipynb files found in $PROJECT_DIR"
    echo "🛑 Exiting - nothing to scan."
    rm -rf "$TEMP_DIR"
    exit 0
fi

echo "📓 Found ${#notebooks[@]} notebook(s):"
for notebook in "${notebooks[@]}"; do
    echo "   - $notebook"
done

# Convert notebooks to Python scripts
echo ""
echo "🔄 Converting notebooks to Python scripts..."
for notebook in "${notebooks[@]}"; do
    # Calculate relative path by removing PROJECT_DIR prefix and replacing slashes with underscores
    rel_path="${notebook#$PROJECT_DIR/}"
    flat_name=$(echo "${rel_path%.ipynb}" | tr '/' '_').py
    
    echo "   - Converting $rel_path to $flat_name"
    if ! jupyter nbconvert --to script "$notebook" --output="${flat_name%.py}" --output-dir="$TEMP_DIR" 2>/dev/null; then
        echo "   ⚠️ Failed to convert $rel_path, skipping..."
    fi
done

# --- Step 2: Generate requirements and scan dependencies ---
echo ""
echo "🔎 Step 2/4: Scanning dependencies for vulnerabilities..."

# Try pipreqsnb first, fallback to regular pipreqs
if command -v pipreqsnb &> /dev/null; then
    echo "📋 Generating requirements.txt from notebooks using pipreqsnb..."
    if pipreqsnb "$PROJECT_DIR" --force --savepath "$TEMP_DIR/requirements.txt" 2>/dev/null; then
        echo "✅ Requirements generated with pipreqsnb"
    else
        echo "⚠️ pipreqsnb failed, trying regular pipreqs on converted files..."
        add_user_site_to_path
        pipreqs "$TEMP_DIR" --force --savepath "$TEMP_DIR/requirements.txt" 2>/dev/null || true
    fi
else
    echo "📋 pipreqsnb not found, using pipreqs on converted Python files..."
    add_user_site_to_path
    pipreqs "$TEMP_DIR" --force --savepath "$TEMP_DIR/requirements.txt" 2>/dev/null || true
fi

# Scan dependencies if requirements.txt exists
if [ -f "$TEMP_DIR/requirements.txt" ]; then
    echo "🔬 Scanning dependencies for CVEs with Snyk..."
    echo "Dependencies found:"
    cat "$TEMP_DIR/requirements.txt" | sed 's/^/   - /'
    
    # Run dependency scan and save report
    if snyk test --file="$TEMP_DIR/requirements.txt" --command=python3 --skip-unresolved \
        --json > "$REPORTS_DIR/dependency_vulnerabilities.json" 2>/dev/null; then
        echo "✅ Dependency scan completed - no vulnerabilities found"
    else
        echo "⚠️ Vulnerabilities found in dependencies - check $REPORTS_DIR/dependency_vulnerabilities.json"
    fi
    
    # Also create text report
    snyk test --file="$TEMP_DIR/requirements.txt" --command=python3 --skip-unresolved \
        > "$REPORTS_DIR/dependency_vulnerabilities.txt" 2>&1 || true
else
    echo "⚠️ No requirements.txt generated - skipping dependency scan"
fi

# --- Step 3: Scan converted Python code ---
echo ""
echo "🔎 Step 3/4: Scanning converted Python code for vulnerabilities..."

if [ "$(ls -A "$TEMP_DIR"/*.py 2>/dev/null)" ]; then
    echo "💻 Running Snyk Code analysis on converted scripts..."
    
    # Run code scan and save report
    if snyk code test "$TEMP_DIR" --json > "$REPORTS_DIR/code_vulnerabilities.json" 2>/dev/null; then
        echo "✅ Code scan completed - no vulnerabilities found"
    else
        echo "⚠️ Code vulnerabilities found - check $REPORTS_DIR/code_vulnerabilities.json"
    fi
    
    # Also create textreport  
    snyk code test "$TEMP_DIR" > "$REPORTS_DIR/code_vulnerabilities.txt" 2>&1 || true
else
    echo "⚠️ No Python files to scan"
fi

# --- Step 4: Generate summary and cleanup ---
echo ""
echo "🔎 Step 4/4: Generating scan summary..."

# Create scan summary
cat > "$REPORTS_DIR/scan_summary.txt" << EOF
Snyk Jupyter Notebook Security Scan Summary
===========================================
Scan Date: $(date)
Project: $PROJECT_DIR
Notebooks Scanned: ${#notebooks[@]}

Files Generated:
- dependency_vulnerabilities.json (JSON format)
- dependency_vulnerabilities.txt (Human readable)
- code_vulnerabilities.json (JSON format)  
- code_vulnerabilities.txt (Human readable)
- scan_summary.txt (This file)

Notebooks Processed:
EOF

for notebook in "${notebooks[@]}"; do
    echo "- $notebook" >> "$REPORTS_DIR/scan_summary.txt"
done

echo ""
echo "🧹 Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Scan completed successfully!"
echo "📊 Reports saved to: $REPORTS_DIR"
echo ""
echo "📋 Summary:"
echo "   - Notebooks processed: ${#notebooks[@]}"
echo "   - Reports location: $REPORTS_DIR"
echo "   - Check *_vulnerabilities.txt files for results in text format"
echo "   - Check *_vulnerabilities.json files for programmatic analysis"