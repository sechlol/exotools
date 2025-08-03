#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Source common utilities
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/common.sh"

# Navigate to project root directory
navigate_to_project_root "${SCRIPT_PATH}"

print_info "Setting up pre-commit hooks for exotools..."

# Check if pre-commit is installed
ensure_package_installed "pre-commit"

# Check if import-linter is installed (needed for the import-linter hook)
ensure_package_installed "import-linter"

# Check if ruff is installed
ensure_package_installed "ruff"

# Install the pre-commit hooks
print_info "Installing pre-commit hooks..."
pre-commit install --install-hooks
pre-commit install --hook-type pre-push

print_success "Pre-commit hooks have been successfully installed!"
print_success "The hooks will run automatically on git commit and git push."
print_info "To run the hooks manually on all files, use:"
echo -e "  pre-commit run --all-files"
print_info "To run a specific hook, use:"
echo -e "  pre-commit run <hook-id> --all-files"
print_info "Example:"
echo -e "  pre-commit run ruff --all-files"
pre-commit run --all-files
