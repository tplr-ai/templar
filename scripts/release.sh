#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION_FILE="$PROJECT_ROOT/src/tplr/__init__.py"
BUMP_TYPE="patch"
GITHUB_API="https://api.github.com"
REMOTE=""
MAIN_BRANCH="main"
VERBOSE=false
SKIP_CONFIRM=false
DRY_RUN=false
ORIGINAL_BRANCH=""
STASHED=false
CUSTOM_VERSION=""

show_help() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Automate the release process:

Options:
  --major          Bump the major version instead of minor
  --minor          Bump the minor version
  --patch          Bump the patch version (default)
  --version VER    Specify a custom version number (e.g., 1.2.3)
  --remote NAME    Specify the git remote to use (default: origin or github)
  --branch NAME    Specify the git branch (default: main)
  --dry            Dry run mode - show what would be done without executing
  -y, --yes        Skip confirmation prompt
  -v, --verbose    Enable verbose output
  -h, --help       Display this help message and exit

Environment variables:
  GITHUB_TOKEN     GitHub API token for creating releases (optional)

EOF
}

print_header() {
    echo -e "\033[1;36m"
    cat << "EOF"
  _____      _
 |  __ \    | |
 | |__) |___| | ___  __ _ ___  ___
 |  _  // _ \ |/ _ \/ _` / __|/ _ \
 | | \ \  __/ |  __/ (_| \__ \  __/
 |_|  \_\___|_|\___|\__,_|___/\___|

 .--.      .--.      .--.      .--.
:::::.\::::::::.\::::::::.\::::::::.\
'      `--'      `--'      `--'      `
EOF
    echo -e "\033[0m"
    echo "========================================"
    echo "  Starting release process..."
    echo "========================================"
    echo ""
}

log() {
    if [[ "$VERBOSE" == true ]]; then
        echo "[$SCRIPT_NAME] $*"
    fi
}

dry_run_info() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would execute: $*"
    fi
}

get_current_version() {
    local version_line
    version_line=$(grep -E "__version__\s*=\s*['\"]" "$VERSION_FILE")
    if [[ -z "$version_line" ]]; then
        echo "Error: Version line not found in $VERSION_FILE" >&2
        exit 1
    fi

    if [[ $version_line =~ __version__[[:space:]]*=[[:space:]]*[\'\"](([0-9]+)\.([0-9]+)\.([0-9]+))[\'\"] ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "Error: Could not parse version from line: $version_line" >&2
        exit 1
    fi
}

calculate_new_version() {
    local current_version=$1
    local bump_type=$2

    local major minor patch
    IFS='.' read -r major minor patch <<< "$current_version"

    if [[ "$bump_type" == "major" ]]; then
        major=$((major + 1))
        minor=0
        patch=0
    elif [[ "$bump_type" == "minor" ]]; then
        minor=$((minor + 1))
        patch=0
    elif [[ "$bump_type" == "patch" ]]; then
        patch=$((patch + 1))
    else
        echo "Error: Unknown bump type: $bump_type" >&2
        exit 1
    fi

    echo "${major}.${minor}.${patch}"
}

update_version_file() {
    local current_version=$1
    local new_version=$2

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would update version from $current_version to $new_version in $VERSION_FILE"
        return 0
    fi

    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' "s/__version__[[:space:]]*=[[:space:]]*['\"]${current_version}['\"]/__version__ = \"${new_version}\"/" "$VERSION_FILE"
    else
        sed -i "s/__version__[[:space:]]*=[[:space:]]*['\"]${current_version}['\"]/__version__ = \"${new_version}\"/" "$VERSION_FILE"
    fi

    log "Updated version from $current_version to $new_version in $VERSION_FILE"
}

commit_version_bump() {
    local version=$1

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would commit version bump to v$version"
        return 0
    fi

    git add "$VERSION_FILE"
    git commit -m "v$version"

    log "Created commit for version v$version"
}

create_git_tag() {
    local version=$1

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would create git tag v$version"
        return 0
    fi

    git tag "v$version"

    log "Created tag v$version"
}

detect_git_remote() {
    if [[ -n "$REMOTE" ]]; then
        if ! git remote | grep -q "^$REMOTE$"; then
            echo "Error: Specified remote '$REMOTE' not found" >&2
            exit 1
        fi
        log "Using user-specified git remote: $REMOTE"
        return
    fi

    if git remote | grep -q "^origin$"; then
        REMOTE="origin"
    elif git remote | grep -q "^github$"; then
        REMOTE="github"
    else
        echo "Error: Neither 'origin' nor 'github' remote found" >&2
        exit 1
    fi

    log "Using git remote: $REMOTE"
}

push_changes() {
    local remote=$1
    local version=$2

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would push changes to $remote/$MAIN_BRANCH with tags"
        echo "[DRY RUN] Would check for dev branch and update if present"
        return 0
    fi

    git push "$remote" HEAD:$MAIN_BRANCH --tags

    log "Pushed changes to $remote/$MAIN_BRANCH with tags"

    if git ls-remote --heads "$remote" dev | grep -q "refs/heads/dev"; then
        log "Dev branch found in remote, merging changes to dev"

        local current_branch=$(git rev-parse --abbrev-ref HEAD)

        git fetch "$remote" dev

        git checkout -B dev "$remote/dev"

        if ! git merge --no-ff "v$version" -m "Merge release v$version into dev"; then
            echo "Warning: Merge conflict occurred. Aborting dev branch update."
            git merge --abort
            git checkout "$current_branch"
        else
            git push "$remote" dev
            git checkout "$current_branch"
            log "Merged and pushed release v$version to $remote/dev"
        fi
    else
        log "No dev branch found in remote, skipping push to dev"
    fi
}

check_requirements() {
    local missing_tools=()

    if ! command -v git >/dev/null 2>&1; then
        missing_tools+=("git")
    fi

    if [[ -n "${GITHUB_TOKEN:-}" ]] && ! command -v curl >/dev/null 2>&1; then
        missing_tools+=("curl")
    fi

    if [[ -n "${GITHUB_TOKEN:-}" ]] && ! command -v jq >/dev/null 2>&1; then
        missing_tools+=("jq")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        echo "Error: Required tools not found: ${missing_tools[*]}" >&2
        echo "Please install these tools and try again" >&2
        exit 1
    fi
}

create_github_release() {
    local version=$1
    local token=$2
    local repo_url

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would create GitHub release for v$version"
        return 0
    fi

    log "Creating GitHub release for v$version"

    repo_url=$(git remote get-url "$REMOTE")

    if [[ $repo_url =~ github\.com[:/]([^/]+)/([^/.]+) ]]; then
        local owner="${BASH_REMATCH[1]}"
        local repo="${BASH_REMATCH[2]}"
        if [[ "$repo" == *.git ]]; then
            repo="${repo%.git}"
        fi
    else
        echo "Error: Could not parse GitHub repository information from URL: $repo_url" >&2
        return 1
    fi

    log "Repository: $owner/$repo"

    local prev_tag
    prev_tag=$(git describe --tags --abbrev=0 "v$version^" 2>/dev/null || echo "")

    if [[ -z "$prev_tag" ]]; then
        log "No previous tag found, this is the first release"
        generate_first_release_notes "$version" "$owner" "$repo" "$token"
    else
        log "Previous tag: $prev_tag"
        generate_release_notes "$prev_tag" "v$version" "$owner" "$repo" "$token"
    fi
}

generate_first_release_notes() {
    local version=$1
    local owner=$2
    local repo=$3
    local token=$4

    log "Creating release v$version with initial release notes"

    local release_data
    release_data=$(cat << EOF
{
  "tag_name": "v$version",
  "name": "v$version",
  "body": "Automated release of v$version.\n\n",
  "draft": false,
  "prerelease": false,
  "generate_release_notes": true
}
EOF
)

    local response
    response=$(curl -s -X POST \
        -H "Authorization: token $token" \
        -H "Accept: application/vnd.github.v3+json" \
        -d "$release_data" \
        "$GITHUB_API/repos/$owner/$repo/releases")

    if echo "$response" | jq -e '.errors' >/dev/null 2>&1; then
        echo "Error creating GitHub release:" >&2
        echo "$response" | jq '.errors' >&2
        return 1
    fi

    echo "Created release v$version on GitHub"
}

generate_release_notes() {
    local prev_tag=$1
    local current_tag=$2
    local owner=$3
    local repo=$4
    local token=$5

    log "Generating release notes from $prev_tag to $current_tag"

    local notes_data
    notes_data=$(cat << EOF
{
  "tag_name": "$current_tag",
  "previous_tag_name": "$prev_tag",
  "generate_release_notes": true
}
EOF
)

    log "Requesting release notes generation from GitHub API"

    local generated_notes
    generated_notes=$(curl -s -X POST \
        -H "Authorization: token $token" \
        -H "Accept: application/vnd.github.v3+json" \
        -d "$notes_data" \
        "$GITHUB_API/repos/$owner/$repo/releases/generate-notes")

    if echo "$generated_notes" | jq -e '.errors' >/dev/null 2>&1; then
        echo "Error generating release notes:" >&2
        echo "$generated_notes" | jq '.errors' >&2
        return 1
    fi

    local body
    body=$(echo "$generated_notes" | jq -r '.body // "Auto-generated release notes not available"')

    log "Creating release with generated notes"

    local release_data
    release_data=$(cat << EOF
{
  "tag_name": "$current_tag",
  "name": "$current_tag",
  "body": $(echo "$body" | jq -R -s .),
  "draft": false,
  "prerelease": false,
  "generate_release_notes": true
}
EOF
)

    local response
    response=$(curl -s -X POST \
        -H "Authorization: token $token" \
        -H "Accept: application/vnd.github.v3+json" \
        -d "$release_data" \
        "$GITHUB_API/repos/$owner/$repo/releases")

    if echo "$response" | jq -e '.errors' >/dev/null 2>&1; then
        echo "Error creating GitHub release:" >&2
        echo "$response" | jq '.errors' >&2
        return 1
    fi

    echo "Created release $current_tag on GitHub with AI-generated notes"
}

confirm_action() {
    local message=$1
    local default=${2:-n}

    local prompt
    if [[ "$default" == "y" ]]; then
        prompt="$message [Y/n] "
    else
        prompt="$message [y/N] "
    fi

    local response
    read -r -p "$prompt" response

    case "${response,,}" in
        y|yes)
            return 0
            ;;
        n|no)
            return 1
            ;;
        *)
            if [[ "$default" == "y" ]]; then
                return 0
            else
                return 1
            fi
            ;;
    esac
}

save_current_branch() {
    ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    log "Current branch is $ORIGINAL_BRANCH"
}

switch_to_main_branch() {
    if [[ "$ORIGINAL_BRANCH" != "$MAIN_BRANCH" ]]; then
        log "Switching from $ORIGINAL_BRANCH to $MAIN_BRANCH"

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY RUN] Would stash changes and switch to $MAIN_BRANCH"
            return 0
        fi

        if ! git diff-index --quiet HEAD --; then
            log "Stashing current changes"
            git stash push -m "Auto-stash for release script"
            STASHED=true
        fi

        git checkout "$MAIN_BRANCH"

        git pull "$REMOTE" "$MAIN_BRANCH"

        log "Now on branch $MAIN_BRANCH with latest changes"
    else
        log "Already on $MAIN_BRANCH branch"

        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY RUN] Would pull latest changes from $REMOTE/$MAIN_BRANCH"
            return 0
        fi

        git pull "$REMOTE" "$MAIN_BRANCH"
    fi
}

restore_original_branch() {
    if [[ "$ORIGINAL_BRANCH" != "$MAIN_BRANCH" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY RUN] Would switch back to $ORIGINAL_BRANCH and apply any stashed changes"
            return 0
        fi

        log "Switching back to original branch $ORIGINAL_BRANCH"
        git checkout "$ORIGINAL_BRANCH"

        if [[ "$STASHED" == true ]]; then
            log "Applying stashed changes"
            git stash pop
        fi

        log "Restored to original state on $ORIGINAL_BRANCH"
    else
        log "No need to switch branches, already on $MAIN_BRANCH"
    fi
}

validate_version() {
    local version=$1

    if ! [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Invalid version format '$version'. Must be in format X.Y.Z (e.g., 1.2.3)" >&2
        exit 1
    fi
}

main() {
    print_header

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --major)
                BUMP_TYPE="major"
                shift
                ;;
            --minor)
                BUMP_TYPE="minor"
                shift
                ;;
            --patch)
                BUMP_TYPE="patch"
                shift
                ;;
            --version)
                shift
                if [[ $# -eq 0 ]]; then
                    echo "Error: --version requires an argument" >&2
                    show_help
                    exit 1
                fi
                CUSTOM_VERSION="$1"
                validate_version "$CUSTOM_VERSION"
                shift
                ;;
            --remote)
                shift
                if [[ $# -eq 0 ]]; then
                    echo "Error: --remote requires an argument" >&2
                    show_help
                    exit 1
                fi
                REMOTE="$1"
                shift
                ;;
            --branch)
                shift
                if [[ $# -eq 0 ]]; then
                    echo "Error: --branch requires an argument" >&2
                    show_help
                    exit 1
                fi
                MAIN_BRANCH="$1"
                shift
                ;;
            --dry)
                DRY_RUN=true
                shift
                ;;
            -y|--yes)
                SKIP_CONFIRM=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Error: Unknown option: $1" >&2
                show_help
                exit 1
                ;;
        esac
    done

    check_requirements

    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Error: Not in a git repository" >&2
        exit 1
    fi

    save_current_branch

    if [[ "$DRY_RUN" == false ]]; then
        detect_git_remote
        switch_to_main_branch

        if ! git diff-index --quiet HEAD --; then
            echo "Error: Working directory is not clean after switching to $MAIN_BRANCH." >&2
            echo "This shouldn't happen as we stash changes. Please resolve manually." >&2
            exit 1
        fi
    else
        detect_git_remote
    fi

    if [[ ! -f "$VERSION_FILE" ]]; then
        echo "Error: Version file not found: $VERSION_FILE" >&2
        if [[ "$DRY_RUN" == false ]]; then
            restore_original_branch
        fi
        exit 1
    fi

    local current_version
    current_version=$(get_current_version)
    local new_version

    if [[ -n "$CUSTOM_VERSION" ]]; then
        new_version="$CUSTOM_VERSION"
        echo "Using custom version: $new_version"
    else
        new_version=$(calculate_new_version "$current_version" "$BUMP_TYPE")
        echo "Current version: $current_version"
        echo "New version: $new_version ($BUMP_TYPE bump)"
    fi

    if [[ "$SKIP_CONFIRM" == true ]] || confirm_action "Proceed with releasing version v$new_version?"; then
        update_version_file "$current_version" "$new_version"
        commit_version_bump "$new_version"
        create_git_tag "$new_version"
        push_changes "$REMOTE" "$new_version"

        if [[ -n "${GITHUB_TOKEN:-}" ]]; then
            create_github_release "$new_version" "$GITHUB_TOKEN"
        else
            echo "GITHUB_TOKEN not set, skipping GitHub release creation"
            echo "To create a release, set the GITHUB_TOKEN environment variable and run again"
        fi

        echo "Release v$new_version completed successfully!"
    else
        echo "Release cancelled by user"
        if [[ "$DRY_RUN" == false ]]; then
            restore_original_branch
        fi
        exit 0
    fi

    if [[ "$DRY_RUN" == false ]]; then
        restore_original_branch
    fi
}

main "$@"
