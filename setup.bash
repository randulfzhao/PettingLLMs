#!/usr/bin/env bash
# Copyright (c) 2025 PettingLLMs Contributors
# SPDX-License-Identifier: MIT
#
# PettingLLMs Environment Setup Script
#
# This script automates the setup of a Conda environment and installs
# all required dependencies for the PettingLLMs project, including PyTorch and
# flash-attention with proper build isolation handling.
#
# Usage:
#   bash setup.bash
#
# Requirements:
#   - Conda
#   - Git (for submodule management)
#   - CUDA 12.8+ (for GPU support)
#   - At least 10GB free disk space

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly CONDA_ENV_NAME="${CONDA_ENV_NAME:-mas}"
readonly LEGACY_VENV_NAME="pettingllms_venv"
readonly PYTORCH_VERSION="2.7.1"
readonly TORCHVISION_VERSION="0.22.1"
readonly TORCHAUDIO_VERSION="2.7.1"
readonly FLASH_ATTN_VERSION="2.8.3"
readonly CUDA_VERSION="cu128"
readonly REQUIREMENTS_FILE="requirements_venv.txt"
readonly DEFAULT_PIP_CACHE_DIR="${HOME}/.cache/pip"

CONDA_BIN="${CONDA_BIN:-}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DEFAULT_PIP_CACHE_DIR}}"

resolve_conda_bin() {
    if [[ -n "${CONDA_BIN}" ]]; then
        return 0
    fi

    local candidate
    candidate=$(type -P conda 2>/dev/null || true)
    if [[ -n "${candidate}" ]]; then
        CONDA_BIN="${candidate}"
        return 0
    fi

    for candidate in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda"; do
        if [[ -x "${candidate}" ]]; then
            CONDA_BIN="${candidate}"
            return 0
        fi
    done

    return 1
}

run_conda() {
    "${CONDA_BIN}" "$@"
}

run_in_env() {
    run_conda run --no-capture-output -n "${CONDA_ENV_NAME}" "$@"
}

conda_env_exists() {
    run_conda run -n "${CONDA_ENV_NAME}" python --version &> /dev/null
}

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

print_header() {
    echo ""
    echo "========================================"
    echo "$*"
    echo "========================================"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! resolve_conda_bin; then
        log_error "Conda not found in PATH"
        log_info "Install Conda or set CONDA_BIN to your Conda executable"
        exit 1
    fi

    if [[ ! -x "${CONDA_BIN}" ]]; then
        log_error "Resolved Conda executable is not executable: ${CONDA_BIN}"
        exit 1
    fi

    # Check if git is available
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed or not in PATH"
        log_info "Please install Git: sudo apt install git"
        exit 1
    fi

    log_info "Found Conda at ${CONDA_BIN}"

    # Check if requirements.txt exists
    if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
        log_error "Requirements file not found: ${REQUIREMENTS_FILE}"
        exit 1
    fi

    log_success "Prerequisites check passed"
}
# Create Conda environment
setup_conda_env() {
    print_header "Step 1/6: Setting up Conda environment"

    if [[ -d "${LEGACY_VENV_NAME}" ]]; then
        log_info "Removing legacy virtual environment: ${LEGACY_VENV_NAME}"
        rm -rf "${LEGACY_VENV_NAME}"
        log_success "Removed legacy virtual environment"
    fi

    if conda_env_exists; then
        log_warning "Conda environment already exists: ${CONDA_ENV_NAME}. Reusing it."
    else
        log_info "Creating Conda environment: ${CONDA_ENV_NAME}"
        run_conda create -y -n "${CONDA_ENV_NAME}" python=3.12
        log_success "Conda environment created"
    fi

    local python_version
    python_version=$(run_in_env python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
    if [[ "${python_version}" != 3.12.* ]]; then
        log_error "Conda environment ${CONDA_ENV_NAME} is using Python ${python_version}, but Python 3.12 is required"
        log_info "Remove the environment with: conda env remove -n ${CONDA_ENV_NAME}"
        exit 1
    fi

    log_info "Using Conda environment ${CONDA_ENV_NAME} with Python ${python_version}"
    log_success "Conda environment ready"
}
# Initialize and update git submodules and install verl
init_submodules() {
    print_header "Step 2/6: Initializing git submodules and installing verl"

    log_info "Initializing and updating git submodules..."
    git submodule update --init --recursive
    log_success "Git submodules updated successfully"

    log_info "Installing verl in editable mode..."
    run_in_env python -m pip install -e ./verl
    log_success "Successfully installed verl"
}



# Upgrade pip tools
upgrade_pip() {
    print_header "Step 3/6: Upgrading pip tools"

    log_info "Upgrading pip, setuptools, and wheel..."
    run_in_env python -m pip install --upgrade pip setuptools wheel --quiet
    log_success "pip tools upgraded"
}

# Install PyTorch
install_pytorch() {
    print_header "Step 4/6: Installing PyTorch"

    log_info "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}"
    run_in_env python -m pip install \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
    log_success "PyTorch installation completed"
}

# Install flash-attn
install_flash_attn() {
    print_header "Step 5/6: Installing flash-attn"

    log_info "Installing flash-attn ${FLASH_ATTN_VERSION}"
    local pip_tmp_dir="${PIP_CACHE_DIR}/tmp"
    mkdir -p "${pip_tmp_dir}"

    env PIP_CACHE_DIR="${PIP_CACHE_DIR}" run_in_env python -m pip install ninja --quiet || true
    env TMPDIR="${pip_tmp_dir}" PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
        run_in_env python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
    log_success "flash-attn installation completed"
}

# Install remaining requirements
install_requirements() {
    print_header "Step 6/6: Installing project dependencies"

    log_info "Installing dependencies from ${REQUIREMENTS_FILE}"
    run_in_env python -m pip install -r "${REQUIREMENTS_FILE}"
    log_success "All dependencies installed successfully"
}

# Print completion message
print_completion() {
    print_header "Installation Complete"

    echo ""
    log_success "Environment setup completed successfully!"
    echo ""
    echo "To activate the Conda environment, run:"
    echo "  conda activate ${CONDA_ENV_NAME}"
    echo ""
    echo "To verify the installation, run:"
    echo "  conda run -n ${CONDA_ENV_NAME} python -c 'import torch; print(torch.__version__)'"
    echo ""
}

# Error handler
error_handler() {
    local exit_code=$?
    log_error "Setup failed with exit code ${exit_code}"
    echo ""
    echo "Common fixes:"
    echo "  conda activate base"
    echo "  conda env remove -n ${CONDA_ENV_NAME}"
    echo "  CONDA_BIN=$(command -v conda) bash setup.bash"
    echo "  git submodule update --init --recursive"
    echo ""
    exit "${exit_code}"
}

# Main function
main() {
    trap error_handler ERR

    print_header "PettingLLMs Environment Setup"
    check_prerequisites
    setup_conda_env
    init_submodules
    upgrade_pip
    install_pytorch
    install_flash_attn
    install_requirements
    run_in_env python -m pip install -e .
    print_completion
}

# Run main function
main "$@"
