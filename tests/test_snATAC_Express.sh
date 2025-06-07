#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test directory setup
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DATA_DIR="${TEST_DIR}/test_data"
TEST_OUTPUT_DIR="${TEST_DIR}/test_output"

echo -e "${YELLOW}Setting up test environment...${NC}"

# Create test directories
mkdir -p "${TEST_DATA_DIR}"
mkdir -p "${TEST_OUTPUT_DIR}"

# Function to check if a command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1${NC}"
        exit 1
    fi
}

# Function to create test data
create_test_data() {
    echo -e "${YELLOW}Creating test data...${NC}"
    
    # Create a small gene list
    cat > "${TEST_DATA_DIR}/test_genes.txt" << EOF
GENE1
GENE2
GENE3
EOF
    check_status "Created test gene list"

    # Create a small peak matrix (simplified for testing)
    cat > "${TEST_DATA_DIR}/test_peaks.txt" << EOF
peak1\t1\t0\t1
peak2\t0\t1\t1
peak3\t1\t1\t0
EOF
    check_status "Created test peak matrix"

    # Create a small RNA matrix (simplified for testing)
    cat > "${TEST_DATA_DIR}/test_rna.txt" << EOF
gene1\t10\t20\t15
gene2\t5\t15\t10
gene3\t20\t10\t25
EOF
    check_status "Created test RNA matrix"
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"

    # Test 1: Check if get_gene_coords.sh exists and is executable
    if [ -f "../../get_gene_coords.sh" ]; then
        chmod +x "../../get_gene_coords.sh"
        check_status "get_gene_coords.sh is executable"
    else
        echo -e "${RED}Error: get_gene_coords.sh not found${NC}"
        exit 1
    fi

    # Test 2: Run get_gene_coords.sh with test data
    echo -e "${YELLOW}Testing gene coordinate extraction...${NC}"
    ../../get_gene_coords.sh \
        -g "${TEST_DATA_DIR}/test_genes.txt" \
        -a "../../gencode.v41.annotation.gtf.gz" \
        -r "tss" \
        -w "2000" \
        -o "${TEST_OUTPUT_DIR}"
    check_status "Gene coordinate extraction"

    # Test 3: Check if output files were created
    if [ -f "${TEST_OUTPUT_DIR}/gene_list.txt" ]; then
        check_status "Output file creation"
    else
        echo -e "${RED}Error: Output file not created${NC}"
        exit 1
    fi

    # Test 4: Validate output format
    echo -e "${YELLOW}Validating output format...${NC}"
    if grep -q "chr" "${TEST_OUTPUT_DIR}/gene_list.txt"; then
        check_status "Output format validation"
    else
        echo -e "${RED}Error: Output format is incorrect${NC}"
        exit 1
    fi
}

# Function to clean up test data
cleanup() {
    echo -e "${YELLOW}Cleaning up test data...${NC}"
    rm -rf "${TEST_DATA_DIR}"
    rm -rf "${TEST_OUTPUT_DIR}"
    check_status "Cleanup completed"
}

# Main execution
echo -e "${YELLOW}Starting snATAC-Express tests${NC}"
echo "----------------------------------------"

create_test_data
run_tests

echo "----------------------------------------"
echo -e "${GREEN}All tests completed successfully!${NC}"

# Uncomment the following line to clean up test data after running
# cleanup 