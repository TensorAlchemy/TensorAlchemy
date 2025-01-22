#!/bin/bash

###########################################
# TensorAlchemy Runner
#
# This script manages the TensorAlchemy validator and miner processes,
# handling updates, process management, and graceful shutdowns.
#
# Usage: ./run.sh [validator|miner] [--auto-update true|false] [additional args...]
###########################################

set -eu

# Ensure we're in the correct directory before doing anything else
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Verify the directory exists and change to it
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Script directory not found: $SCRIPT_DIR" >&2
    exit 1
fi

cd "$SCRIPT_DIR" || {
    echo "Error: Failed to change to script directory: $SCRIPT_DIR" >&2
    exit 1
}

# Configuration
PYTHON_CMD="python3"
VALIDATOR_PATH="neurons/validator/main.py"
MINER_PATH="neurons/miners/Inpainter/main.py"
UPDATE_EXIT_CODE=42
WIDTH=80
AUTO_UPDATE=true

# Verify required paths exist
if [ ! -f "$VALIDATOR_PATH" ]; then
    echo "Error: Validator script not found at: $VALIDATOR_PATH" >&2
    exit 1
fi

if [ ! -f "$MINER_PATH" ]; then
    echo "Error: Miner script not found at: $MINER_PATH" >&2
    exit 1
fi

# Get current git branch
REPO_BRANCH=$(git rev-parse --abbrev-ref HEAD || echo "unknown")

# Check if terminal supports colors
if [ -t 1 ] && command -v tput >/dev/null && [ "$(tput colors)" -ge 8 ]; then
    COLOR_RED="$(printf '\x1B[0;31m')"
    COLOR_GREEN="$(printf '\x1B[0;32m')"
    COLOR_YELLOW="$(printf '\x1B[1;33m')"
    COLOR_BLUE="$(printf '\x1B[0;34m')"
    COLOR_NC="$(printf '\x1B[0m')"
else
    COLOR_RED=""
    COLOR_GREEN=""
    COLOR_YELLOW=""
    COLOR_BLUE=""
    COLOR_NC=""
fi

log() {
    _level="$1"
    _message="$2"
    _timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    _script_name=$(basename "$0")
    printf "%s [%s] %s: %s\n" "$_timestamp" "$_level" "$_script_name" "$_message"
}

log_info() {
    log "${COLOR_GREEN}INFO${COLOR_NC}" "$1"
}

log_warn() {
    log "${COLOR_YELLOW}WARN${COLOR_NC}" "$1"
}

log_error() {
    log "${COLOR_RED}ERROR${COLOR_NC}" "$1" >&2
}

print_banner() {
    _message="$1"
    _border="$(printf "%${WIDTH}s" | tr " " "-")"

    printf "${COLOR_BLUE}+%s+${COLOR_NC}\n" "$_border"

    printf "%b" "$_message" | while IFS= read -r _line; do
        [ -z "$_line" ] && _line=" "
        _line_length=${#_line}
        _padding=$(( (WIDTH - 3 - _line_length) / 2 ))
        _padding_left=$(( _padding + (WIDTH - 3 - _line_length) % 2 ))
        _padding_right=$_padding

        printf "${COLOR_BLUE}|${COLOR_NC} %*s%s%*s ${COLOR_BLUE}|${COLOR_NC}\n" \
            "$_padding_left" "" "$_line" "$_padding_right" ""
    done

    printf "${COLOR_BLUE}+%s+${COLOR_NC}\n" "$_border"
}

run_process() {
    case "$1" in
        validator)
            if [ "$AUTO_UPDATE" = true ]; then
                log_info "Launching validator with args: ${*:2} --alchemy.auto_update"
                "$PYTHON_CMD" "$VALIDATOR_PATH" "${@:2}" "--alchemy.auto_update"
            else
                log_info "Launching validator with args: ${*:2}"
                "$PYTHON_CMD" "$VALIDATOR_PATH" "${@:2}" "--alchemy.auto_update"
            fi
            ;;
        miner)
            if [ "$AUTO_UPDATE" = true ]; then
                log_info "Launching miner with args: ${*:2} --alchemy.auto_update"
                "$PYTHON_CMD" "$MINER_PATH" "${@:2}" "--alchemy.auto_update"
            else
                log_info "Launching miner with args: ${*:2}"
                "$PYTHON_CMD" "$MINER_PATH" "${@:2}"
            fi
            ;;
        *)
            log_error "Invalid process type: $1"
            return 1
            ;;
    esac
}

check_git_updates() {
    # First verify we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository. Directory: $(pwd)"
        return 1
    fi

    log_info "Checking for updates..."

    # Fetch the latest changes
    if ! git fetch origin "$REPO_BRANCH" --quiet 2>/dev/null; then
        log_error "Failed to fetch updates from remote repository"
        return 1
    fi

    _local_commit=$(git rev-parse HEAD)
    _remote_commit=$(git rev-parse "origin/$REPO_BRANCH")
    log_info "Commits - Local: $_local_commit Remote: $_remote_commit"

    if [ "$_local_commit" != "$_remote_commit" ]; then
        print_banner "Update Available!\n\nLocal:  ${_local_commit%????????*}\nRemote: ${_remote_commit%????????*}"
        return 0
    fi
    return 1
}

update_repository() {
    # First verify we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository. Directory: $(pwd)"
        return 1
    fi

    log_info "Forcing repository update..."

    # Debug: Show current git status
    log_info "Git status before update:"
    git status

    # Forcefully reset and update the repository
    if ! git fetch origin "$REPO_BRANCH" --quiet; then
        log_error "Failed to fetch from remote"
        return 1
    fi
    log_info "Fetch completed"

    if ! git reset --hard "origin/$REPO_BRANCH"; then
        log_error "Failed to reset to origin/$REPO_BRANCH"
        return 1
    fi
    log_info "Reset completed"

    if ! git clean -fd; then
        log_error "Failed to clean repository"
        return 1
    fi
    log_info "Clean completed"

    # Debug: Show git status after update
    log_info "Git status after update:"
    git status

    return 0
}

handle_exit_code() {
    _exit_code="$1"
    _process_type="$2"

    case $_exit_code in
        0)
            log_info "Process completed normally"
            return 0
            ;;
        "$UPDATE_EXIT_CODE")
            if [ "$AUTO_UPDATE" = true ]; then
                log_info "Update detected during runtime - restarting"
                return 2
            else
                log_info "Update detected but auto-update is disabled - exiting"
                return 0
            fi
            ;;
        *)
            log_error "Process $_process_type failed with exit code ${_exit_code}"
            return 1
            ;;
    esac
}

parse_args() {
    local args=("$@")
    local i=0
    _process_type=""

    while [ $i -lt ${#args[@]} ]; do
        case "${args[$i]}" in
            --auto-update)
                if [ $((i + 1)) -lt ${#args[@]} ]; then
                    if [ "${args[$((i + 1))]}" = "false" ] || [ "${args[$((i + 1))]}" = "0" ]; then
                        AUTO_UPDATE=false
                        log_info "Auto-update disabled via command line"
                    elif [ "${args[$((i + 1))]}" = "true" ] || [ "${args[$((i + 1))]}" = "1" ]; then
                        AUTO_UPDATE=true
                        log_info "Auto-update enabled via command line"
                    else
                        log_error "Invalid value for --auto-update: ${args[$((i + 1))]} (must be true/false or 1/0)"
                        exit 1
                    fi
                    i=$((i + 2))
                else
                    log_error "--auto-update requires a value"
                    exit 1
                fi
                ;;
            validator|miner)
                _process_type="${args[$i]}"
                i=$((i + 1))
                ;;
            *)
                i=$((i + 1))
                ;;
        esac
    done

    # Debug output
    log_info "Parsed arguments - Process type: $_process_type, Auto-update: $AUTO_UPDATE"
    return 0
}

main() {
    if [ $# -lt 1 ]; then
        print_banner "Usage: ./run.sh [validator|miner] [--auto-update true|false] [additional args...]"
        exit 1
    fi

    # First parse auto-update and process type
    parse_args "$@"

    if [ -z "${_process_type:-}" ]; then
        print_banner "Error: First argument must be 'validator' or 'miner'"
        exit 1
    fi

    log_info "Starting TensorAlchemy $_process_type (auto-update: $AUTO_UPDATE)"

    # Only update if auto-update is true
    if [ "$AUTO_UPDATE" = true ]; then
        log_info "Auto-update is enabled, performing initial update"
        update_repository || exit 1
    else
        log_info "Auto-update is disabled, skipping updates"
    fi

    log_info "Installing dependencies..."
    if ! pip install -e . --no-cache-dir; then
        log_error "Failed to install updated dependencies"
        return 1
    fi

    # Run the process once and handle its exit
    log_info "Running $_process_type process..."
    run_process "$@"
    _exit_code=$?

    handle_exit_code "$_exit_code" "$_process_type"
    _should_continue=$?

    # Exit codes:
    # 0 = normal completion
    # 1 = error
    # 2 = update needed
    if [ $_should_continue -eq 1 ]; then
        log_error "Process failed with error, exiting"
        exit 1
    elif [ $_should_continue -eq 2 ] && [ "$AUTO_UPDATE" = true ]; then
        log_info "Update requested and auto-update is enabled"
        update_repository && main "$@"  # Restart from beginning if update succeeds
    fi

    # Normal exit
    exit 0
}

main "$@"
