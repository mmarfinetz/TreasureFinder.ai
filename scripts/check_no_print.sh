#!/bin/bash
# Check for print statements in production code
# Should use logger instead

ERROR_FOUND=0

for file in "$@"; do
    # Skip test files and specific allowed files
    if [[ "$file" == *test*.py ]] || [[ "$file" == tests/* ]] || [[ "$file" == quick_test.py ]]; then
        continue
    fi
    
    # Check for print statements (but allow if it's in a comment or string)
    if grep -E "^\s*print\(" "$file" > /dev/null 2>&1; then
        echo "❌ ERROR: Found print statement in $file (use logger instead)"
        grep -n -E "^\s*print\(" "$file"
        ERROR_FOUND=1
    fi
done

if [ $ERROR_FOUND -eq 1 ]; then
    echo ""
    echo "Production code should use structured logging instead of print statements."
    echo "Replace with: logger.info(), logger.warning(), logger.error(), etc."
    exit 1
fi

exit 0