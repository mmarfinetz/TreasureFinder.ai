#!/bin/bash
# Check for banned patterns in production code
# Exit with non-zero status if found

BANNED_PATTERNS=(
    "np\.random\."
    "torch\.randn"
    "random\.random"
    "random\.uniform"
    "random\.normal"
    "random\.randint"
    "simulated satellite data"
    "synthetic data"
    "# Simulated"
    "# Synthetic"
    "NIR simulated"
)

ERROR_FOUND=0

for file in "$@"; do
    # Skip test files and notebooks
    if [[ "$file" == *test*.py ]] || [[ "$file" == tests/* ]] || [[ "$file" == *.ipynb ]]; then
        continue
    fi
    
    for pattern in "${BANNED_PATTERNS[@]}"; do
        if grep -E "$pattern" "$file" > /dev/null 2>&1; then
            echo "❌ ERROR: Found banned pattern '$pattern' in $file"
            grep -n -E "$pattern" "$file"
            ERROR_FOUND=1
        fi
    done
done

if [ $ERROR_FOUND -eq 1 ]; then
    echo ""
    echo "Production code must not contain synthetic/random data generation."
    echo "Please use real data sources or return explicit error states."
    exit 1
fi

exit 0