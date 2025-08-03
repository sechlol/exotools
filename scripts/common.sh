#!/bin/bash
# Common utilities for exotools scripts

# Colors for better readability
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export RED='\033[0;31m'
export NC='\033[0m' # No Color

# Get the script directory and project root
get_project_root() {
    local script_path="$1"
    local script_dir="$( cd "$( dirname "${script_path}" )" &> /dev/null && pwd )"
    local project_root="$( dirname "${script_dir}" )"
    echo "${project_root}"
}

# Navigate to project root directory
navigate_to_project_root() {
    local script_path="$1"
    local project_root=$(get_project_root "${script_path}")
    cd "${project_root}"
    echo -e "${YELLOW}Project root: ${project_root}${NC}"
    return 0
}

# Print a message with color
print_message() {
    local color="$1"
    local message="$2"
    echo -e "${color}${message}${NC}"
}

# Print info message (yellow)
print_info() {
    print_message "${YELLOW}" "$1"
}

# Print success message (green)
print_success() {
    print_message "${GREEN}" "$1"
}

# Print error message (red)
print_error() {
    print_message "${RED}" "$1"
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Install a package if not already installed
ensure_package_installed() {
    local package="$1"
    if ! command_exists "${package}"; then
        print_info "${package} is not installed. Installing..."
        pip install "${package}"
    else
        print_success "${package} is already installed."
    fi
}

# Clean up previous builds
clean_previous_builds() {
    print_info "Cleaning up previous builds..."
    rm -rf build/ dist/ *.egg-info/
}
