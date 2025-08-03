#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Source common utilities
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/common.sh"

# Navigate to project root directory
navigate_to_project_root "${SCRIPT_PATH}"

print_info "Starting TestPyPI publishing process for exotools..."

# Clean up previous builds
clean_previous_builds

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade build tools
print_info "Installing/upgrading build tools..."
pip install --upgrade pip
pip install --upgrade build twine

# Build the package
print_info "Building the package..."
python -m build

# Check the package with twine
print_info "Checking the package with twine..."
twine check dist/*

# Upload to TestPyPI - only the newly built packages
print_info "Uploading to TestPyPI..."
# Get current version from pyproject.toml
VERSION=$(grep -m 1 'version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
print_info "Uploading version ${VERSION} to TestPyPI..."
twine upload --repository testpypi dist/*${VERSION}* --verbose

# Display installation instructions
print_success "Package successfully uploaded to TestPyPI!"
print_success "You can install it using:"
echo -e "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ exotools==${VERSION}"

# Deactivate virtual environment
deactivate

# Clean up virtual environment
print_info "Cleaning up virtual environment..."
rm -rf venv/

print_success "TestPyPI publishing process completed!"
