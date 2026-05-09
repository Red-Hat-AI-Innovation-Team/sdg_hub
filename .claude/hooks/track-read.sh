#!/bin/bash
# PreToolUse hook for Read tool
# Tracks when evidence files are read

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)

# Check if the file matches evidence patterns
case "$FILE_PATH" in
  *screenshots/*|*-console.txt|*-result.txt|*.png|*coverage*|*test-results*)
    echo "$FILE_PATH" >> .claude/.evidence-reads
    ;;
esac

# Always allow the read
exit 0
