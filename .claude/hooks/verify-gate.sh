#!/bin/bash
# PreToolUse hook for Write|Edit tools
# Blocks writes to results files unless evidence has been read

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)

# Only gate writes to test-results or quality files
case "$FILE_PATH" in
  *test-results.json|*QUALITY.md|*performance_report.json)
    if [ ! -s .claude/.evidence-reads ]; then
      echo '{"decision":"block","reason":"No test output or execution evidence has been Read this session. Run the tests and read the output before updating results."}'
      exit 1
    fi
    # Clear evidence log after successful gate pass
    > .claude/.evidence-reads
    ;;
esac

exit 0
