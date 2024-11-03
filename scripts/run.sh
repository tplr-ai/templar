#!/usr/bin/env bash

# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

set -euo pipefail

# Initialize default values
DEBUG=false
PROJECT="templar"
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
BUCKET=""
NETWORK=""
NEURON_TYPE=""

# Function to display help message
display_help() {
    cat << EOF
Usage: $0 [options]

Options:
    --debug                     Enable debug mode
    --project <project_name>    Set the project name (default: templar)
    --aws-access-key-id <key>   Set AWS Access Key ID
    --aws-secret-access-key <key> Set AWS Secret Access Key
    --bucket <bucket_name>      Set the S3 bucket name
    --network <network_name>    Set the network (options: finney, test, local)
    --neuron <neuron_type>     Set the neuron type (options: miner, validator)
    -h, --help                  Display this help message

Description:
    Installs and runs a τemplar neuron on your GPU. If the --network option is not provided, you will be prompted to select a network.
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --debug)
            DEBUG=true
            shift
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --aws-access-key-id)
            AWS_ACCESS_KEY_ID="$2"
            shift 2
            ;;
        --aws-secret-access-key)
            AWS_SECRET_ACCESS_KEY="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --network)
            NETWORK="$2"
            shift 2
            ;;
        --neuron)
            NEURON_TYPE="$2"
            shift 2
            ;;
        -h|--help|-help|--h)
            display_help
            exit 0
            ;;
        *)
            # Only error if not a network argument
            if [[ "$1" != "--network" ]]; then
                echo "Unknown option: $1"
                display_help
                exit 1
            fi
            shift
            ;;
    esac
done



# Set up colors and styles for terminal output
if [[ -t 1 ]]; then
    tty_escape() { printf "\033[%sm" "$1"; }
else
    tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_green="$(tty_mkbold 32)"
tty_yellow="$(tty_mkbold 33)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

# Logging functions for standardized output
ohai() {
    printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$*"
}

pdone() {
    printf "  ${tty_green}[✔]${tty_bold} %s${tty_reset}\n" "$*"
}

info() {
    printf "${tty_green}%s${tty_reset}\n" "$*"
}

warn() {
    printf "${tty_yellow}Warning${tty_reset}: %s\n" "$*" >&2
}

error() {
    printf "${tty_red}Error${tty_reset}: %s\n" "$*" >&2
}

abort() {
    error "$@"
    exit 1
}

trap 'abort "An unexpected error occurred."' ERR

# Function to get a single character input from the user
getc() {
    local save_state
    save_state="$(/bin/stty -g)"
    /bin/stty raw -echo
    IFS='' read -r -n 1 -d '' "$@"
    /bin/stty "${save_state}"
}

# Function to pause execution and wait for user confirmation
wait_for_user() {
    local c
    echo
    echo -e "${tty_bold}Press ${tty_green}RETURN/ENTER${tty_reset} ${tty_bold}to continue or any other key to abort:${tty_reset}"
    getc c
    if ! [[ "${c}" == $'\r' || "${c}" == $'\n' ]]
    then
        exit 1
    fi
}

# Function to execute a command with logging
execute() {
    ohai "Running: $*"
    if ! "$@"; then
        abort "Failed during: $*"
    fi
}

# Function to check for sudo access
have_sudo_access() {
    if ! command -v sudo &> /dev/null; then
        warn "sudo command not found. Please install sudo or run as root."
        return 1
    fi
    if [ "$EUID" -ne 0 ]; then 
        if ! sudo -n true 2>/dev/null; then
            warn "This script requires sudo access to install packages. Please run as root or ensure your user has sudo privileges."
            return 1
        fi
    fi
    return 0
}

# Function to execute commands with sudo if necessary
execute_sudo() {
    if have_sudo_access; then
        ohai "sudo $*"
        if ! sudo "$@"; then
            abort "Failed to execute: sudo $*"
        fi
    else
        warn "Sudo access is required, attempting to run without sudo"
        ohai "$*"
        if ! "$@"; then
            abort "Failed to execute: $*"
        fi
    fi
}

# Function to set or replace environment variables in bash_profile
set_or_replace_env_var() {
    local var_name="$1"
    local var_value="$2"
    local profile_file="$3"

    # Escape special characters for sed
    local escaped_var_value=$(printf '%s\n' "$var_value" | sed -e 's/[\/&]/\\&/g')

    if grep -q "^export $var_name=" "$profile_file"; then
        # Variable exists, replace it
        sed -i.bak "s/^export $var_name=.*/export $var_name=\"$escaped_var_value\"/" "$profile_file"
    else
        # Variable does not exist, append it
        echo "export $var_name=\"$var_value\"" >> "$profile_file"
    fi
}

# Define color codes
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
BOLD='\033[1m'
RED='\033[0;31m'

# Clear screen
clear

# Display the original logo with cyan color and bold
echo -e "${RED}${BOLD}"
printf '%s\n' "___  _  _ _  _ | _  _  "
printf '%s\n' "  | (/_| | ||_)|(_||   "
printf '%s\n' "  |         |          "
echo -e "${NC}"

echo -e "\n${BOLD}${BLUE}Welcome to the τemplar Installation Script${NC}\n"

echo -e "${YELLOW}This script will:${NC}\n"
echo -e "${CYAN}1.${NC} Install required software ${GREEN}(Git, npm, pm2, rust, uv, Python 3.12)${NC}"
echo -e "${CYAN}2.${NC} Set up ${GREEN}AWS credentials${NC}"
echo -e "${CYAN}3.${NC} Clone and set up the ${GREEN}τemplar repository${NC}"
echo -e "${CYAN}4.${NC} Create and register ${GREEN}Bittensor wallets${NC}"
echo -e "${CYAN}5.${NC} Configure ${GREEN}wandb for logging${NC}"
echo -e "${CYAN}6.${NC} Clean the specified ${GREEN}S3 bucket${NC}"
echo -e "${CYAN}7.${NC} Start ${GREEN}τemplar neurons${NC} on available GPUs on your chosen network\n"

echo -e "${YELLOW}⚠️  Please ensure you have:${NC}"
echo -e "   ${GREEN}✓${NC} A stable internet connection"
echo -e "   ${GREEN}✓${NC} Sufficient permissions to install software\n"



wait_for_user

# If network not provided, prompt user to select one
if [[ -z "$NETWORK" ]]; then
    echo "Please select a network:"
    echo "1) Finney"
    echo "2) Testnet"
    echo "3) Local"
    read -p "Enter selection [1-3]: " network_choice
    
    case $network_choice in
        1) NETWORK="Finney" ;;
        2) NETWORK="Testnet" ;;
        3) NETWORK="Local" ;;
        *) 
            echo "Invalid selection"
            exit 1
            ;;
    esac
fi

# Set network-specific variables based on the selected network
case "$NETWORK" in
    finney|FINNEY|Finney)
        SUBTENSOR_NETWORK="main"
        NETUID=3
        SUBTENSOR_CHAIN_ENDPOINT=""
        PM2_NETWORK_OPTIONS=""
        ;;
    test|testnet|TEST|TESTNET|Testnet)
        SUBTENSOR_NETWORK="test"
        NETUID=223
        SUBTENSOR_CHAIN_ENDPOINT="wss://test.finney.opentensor.ai:443/"
        PM2_NETWORK_OPTIONS="--test"
        ;;
    local|LOCAL|Local)
        SUBTENSOR_NETWORK="local"
        NETUID=1
        SUBTENSOR_CHAIN_ENDPOINT="wss://localhost:9944"
        PM2_NETWORK_OPTIONS=""
        ;;
    *)
        echo "Unknown network: $NETWORK"
        display_help
        exit 1
        ;;
esac


if [[ -z "$NEURON_TYPE" ]]; then
    echo "Please select a neuron type:"
    echo "1) Miner"
    echo "2) Validator"
    read -p "Enter selection [1-2]: " neuron_choice
    
    case $neuron_choice in
        1) NEURON_TYPE="miner" ;;
        2) NEURON_TYPE="validator" ;;
        *) 
            echo "Invalid selection"
            exit 1
            ;;
    esac
fi

# Validate neuron type
case "$NEURON_TYPE" in
    miner|validator)
        ;;
    *)
        echo "Invalid neuron type: $NEURON_TYPE"
        display_help
        exit 1
        ;;
esac

# Ensure ~/.bash_profile exists
touch ~/.bash_profile
source ~/.bash_profile

# Backup the bash_profile
cp ~/.bash_profile ~/.bash_profile.bak

# Prompt the user for AWS credentials if not supplied via command-line
ohai "Getting AWS credentials ..."
if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]] || [[ -z "$BUCKET" ]]; then
    warn "This script will store your AWS credentials in your ~/.bash_profile file."
    warn "This is not secure and is not recommended."
    read -p "Do you want to proceed? [y/N]: " proceed
    if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
        abort "Aborted by user."
    fi

    if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
        read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY_ID
    fi
    if [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
        read -p "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    fi
    if [[ -z "$BUCKET" ]]; then
        read -p "Enter your S3 Bucket Name: " BUCKET
    fi
fi

# Overwrite or add the AWS credentials in the bash_profile
set_or_replace_env_var "AWS_ACCESS_KEY_ID" "$AWS_ACCESS_KEY_ID" ~/.bash_profile
set_or_replace_env_var "AWS_SECRET_ACCESS_KEY" "$AWS_SECRET_ACCESS_KEY" ~/.bash_profile
set_or_replace_env_var "BUCKET" "$BUCKET" ~/.bash_profile

# Source the bash_profile to apply the changes
source ~/.bash_profile
pdone "AWS credentials set in ~/.bash_profile"

ohai "Installing requirements ..."
# Install Git if not present
if ! command -v git &> /dev/null; then
    ohai "Git not found. Installing git ..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Git..."
                if [[ "$DEBUG" == "true" ]]; then
                    execute_sudo apt-get update -y
                    execute_sudo apt-get install git -y
                else
                    execute_sudo apt-get update -y > /dev/null 2>&1
                    execute_sudo apt-get install git -y > /dev/null 2>&1
                fi
            else
                warn "Unsupported Linux distribution: $ID"
                abort "Cannot install Git automatically"
            fi
        else
            warn "Cannot detect Linux distribution"
            abort "Cannot install Git automatically"
        fi
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
else
    pdone "Git is already installed"
fi

# Check for Rust installation
if ! command -v rustc &> /dev/null; then
    ohai "Installing Rust ..."
    if [[ "$DEBUG" == "true" ]]; then
        execute curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    else
        execute curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
    fi
    # Add Rust to the PATH for the current session
    source $HOME/.cargo/env
fi
pdone "Rust is installed"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    ohai "Installing uv ..."
    if [[ "$DEBUG" == "true" ]]; then
        execute curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        execute curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    fi
    # Add uv to the PATH for the current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi
pdone "uv is installed"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    ohai "Installing npm ..."
    if ! command -v node &> /dev/null; then
        ohai "Node.js could not be found, installing..."
        if ! curl -fsSL https://deb.nodesource.com/setup_18.x | bash; then
            abort "Failed to download Node.js setup script"
        fi
        if ! execute_sudo apt-get install -y nodejs; then
            abort "Failed to install Node.js"
        fi
    fi
    if ! curl -L https://www.npmjs.com/install.sh | sh; then
        abort "Failed to install npm"
    fi
fi
pdone "npm is installed"

# Install pm2
if ! command -v pm2 &> /dev/null; then
    ohai "Installing pm2 ..."
    if [[ "$DEBUG" == "true" ]]; then
        execute npm install pm2 -g
    else
        execute npm install pm2 -g > /dev/null 2>&1
    fi
fi
pdone "pm2 is installed"

ohai "Installing τemplar ..."
# Check if we are inside the τemplar repository
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    REPO_PATH="."
else
    if [ ! -d "τemplar" ]; then
        ohai "Cloning τemplar ..."
        execute git clone https://github.com/RaoFoundation/templar
        REPO_PATH="templar"
    else
        REPO_PATH="templar"
    fi
fi
pdone "τemplar repository is ready at $REPO_PATH"

# Install Python 3.12 if not installed
if ! command -v python3.12 &> /dev/null; then
    ohai "Installing python3.12 ..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Python 3.12..."
                if [[ "$DEBUG" == "true" ]]; then
                    if have_sudo_access; then
                        execute_sudo add-apt-repository ppa:deadsnakes/ppa -y
                    else
                        warn "Skipping add-apt-repository due to lack of sudo access"
                    fi
                    execute_sudo apt-get update -y
                else
                    if have_sudo_access; then
                        execute_sudo add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
                    else
                        warn "Skipping add-apt-repository due to lack of sudo access"
                    fi
                    execute_sudo apt-get update -y > /dev/null 2>&1
                    execute_sudo apt-get install --reinstall python3-apt > /dev/null 2>&1
                    execute_sudo apt-get install python3.12 -y > /dev/null 2>&1
                    execute_sudo apt-get install python3.12-venv > /dev/null 2>&1
                fi
            else
                warn "Unsupported Linux distribution: $ID"
                abort "Cannot install Python 3.12 automatically"
            fi
        else
            warn "Cannot detect Linux distribution"
            abort "Cannot install Python 3.12 automatically"
        fi
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
fi
pdone "Python 3.12 is installed"

# Create a virtual environment if it does not exist
if [ ! -d "$REPO_PATH/venv" ]; then
    ohai "Creating virtual environment at $REPO_PATH..."
    if [[ "$DEBUG" == "true" ]]; then
        execute uv venv "$REPO_PATH/.venv"
    else
        execute uv venv "$REPO_PATH/.venv" > /dev/null 2>&1
    fi
fi
pdone "Virtual environment is set up at $REPO_PATH"


# Activate the virtual environment
ohai "Activating virtual environment ..."
source $REPO_PATH/.venv/bin/activate
pdone "Virtual environment activated"

ohai "Installing Python requirements ..."
cd "$REPO_PATH"

# First, ensure uv is properly set up
if [[ "$DEBUG" == "true" ]]; then
    execute uv pip install --upgrade pip
else
    execute uv pip install --upgrade pip > /dev/null 2>&1
fi

# Install PyTorch first
if [[ "$DEBUG" == "true" ]]; then
    execute uv pip install torch --index-url https://download.pytorch.org/whl/cu118
else
    execute uv pip install torch --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
fi

# Now run uv sync
if [[ "$DEBUG" == "true" ]]; then
    # remove prerelease once bt decode is released
    execute uv sync --extra all --prerelease=allow
else
    execute uv sync --extra all --prerelease=allow  > /dev/null 2>&1
fi

# Install flash-attn separately due to its special requirements
if [[ "$DEBUG" == "true" ]]; then
    execute uv pip install flash-attn --no-build-isolation
else
    execute uv pip install flash-attn --no-build-isolation > /dev/null 2>&1
fi

pdone "Python requirements installed"

# Check for GPUs
ohai "Checking for GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
    NUM_GPUS=0
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    if [ "$NUM_GPUS" -gt 0 ]; then
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | while read -r memory; do
            pdone "Found GPU with $((memory / 1024)) GB of memory"
        done
    else
        warn "No GPUs found on this machine."
    fi
fi

# Check system RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    pdone "System RAM: ${TOTAL_RAM} GB"
else
    warn "Cannot determine system RAM. 'free' command not found."
fi

# Check for GPUs
ohai "Checking for GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
    NUM_GPUS=0
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    if [ "$NUM_GPUS" -gt 0 ]; then
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | while read -r memory; do
            pdone "Found GPU with $((memory / 1024)) GB of memory"
        done
    else
        warn "No GPUs found on this machine."
    fi
fi
# Create wallets section
ohai "Creating wallets ..."

# Create coldkey if it doesn't exist
exists_on_device=$(python3 -c "import bittensor as bt; w = bt.wallet(); print(w.coldkey_file.exists_on_device())" 2>/dev/null)
if [ "$exists_on_device" != "True" ]; then
    echo "n" | btcli wallet new_coldkey --wallet.name default --n-words 12 > /dev/null 2>&1
fi
pdone "Wallet 'default' is ready"

# Create hotkeys based on neuron type
if [ "$NEURON_TYPE" = "validator" ]; then
    # Create single hotkey for validator
    HOTKEY_NAME="validator"
    
    # Check if hotkey exists
    exists_on_device=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='$HOTKEY_NAME'); print(w.hotkey_file.exists_on_device())" 2>/dev/null)
    
    if [ "$exists_on_device" != "True" ]; then
        echo "n" | btcli wallet new_hotkey --wallet.name default --wallet.hotkey "$HOTKEY_NAME" --n-words 12 > /dev/null 2>&1
        pdone "Created Validator Hotkey '$HOTKEY_NAME'"
    fi

    # Check registration status
    is_registered=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='$HOTKEY_NAME'); sub = bt.subtensor('$SUBTENSOR_NETWORK'); print(sub.is_hotkey_registered_on_subnet(hotkey_ss58=w.hotkey.ss58_address, netuid=$NETUID))")
    
    if [[ "$is_registered" != *"True"* ]]; then
        ohai "Registering validator hotkey on netuid $NETUID"
        btcli subnet pow_register --wallet.name default --wallet.hotkey "$HOTKEY_NAME" --netuid "$NETUID" --subtensor.network "$SUBTENSOR_NETWORK" --no_prompt > /dev/null 2>&1
        pdone "Registered Validator Hotkey on netuid $NETUID"
    else
        pdone "Validator Hotkey already registered on netuid $NETUID"
    fi
else
    # Create miner hotkeys
    if [ "$NUM_GPUS" -gt 0 ]; then
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            HOTKEY_NAME="C$i"
            
            # Check if hotkey exists
            exists_on_device=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='$HOTKEY_NAME'); print(w.hotkey_file.exists_on_device())" 2>/dev/null)
            
            if [ "$exists_on_device" != "True" ]; then
                echo "n" | btcli wallet new_hotkey --wallet.name default --wallet.hotkey "$HOTKEY_NAME" --n-words 12 > /dev/null 2>&1
                pdone "Created Miner Hotkey '$HOTKEY_NAME'"
            fi

            # Check registration status
            is_registered=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='$HOTKEY_NAME'); sub = bt.subtensor('$SUBTENSOR_NETWORK'); print(sub.is_hotkey_registered_on_subnet(hotkey_ss58=w.hotkey.ss58_address, netuid=$NETUID))")
            
            if [[ "$is_registered" != *"True"* ]]; then
                ohai "Registering miner hotkey $HOTKEY_NAME on netuid $NETUID"
                btcli subnet pow_register --wallet.name default --wallet.hotkey "$HOTKEY_NAME" --netuid "$NETUID" --subtensor.network "$SUBTENSOR_NETWORK" --no_prompt > /dev/null 2>&1
                pdone "Registered Miner Hotkey $HOTKEY_NAME on netuid $NETUID"
            else
                pdone "Miner Hotkey $HOTKEY_NAME already registered on netuid $NETUID"
            fi
        done
    fi
fi
else
    warn "No GPUs found. Skipping hotkey creation."
    exit 1
fi
pdone "All hotkeys registered"

# Initialize PM2
ohai "Stopping old pm2 processes..."
if pm2 list | grep -q 'online'; then
    pm2 delete all
    pdone "Old processes stopped"
fi

# Start neurons based on type
if [ "$NEURON_TYPE" = "validator" ]
then
    ohai "Starting validator on network '$NETWORK' ..."
    VALIDATOR_ARGS="--actual_batch_size 6 --wallet.name default --wallet.hotkey validator --bucket \"$BUCKET\" --use_wandb --project \"$PROJECT\" --netuid $NETUID"
    
    # Add network options
    [ -n "$PM2_NETWORK_OPTIONS" ] && VALIDATOR_ARGS="$VALIDATOR_ARGS $PM2_NETWORK_OPTIONS"
    [ -n "$SUBTENSOR_NETWORK" ] && VALIDATOR_ARGS="$VALIDATOR_ARGS --subtensor.network $SUBTENSOR_NETWORK"
    [ -n "$SUBTENSOR_CHAIN_ENDPOINT" ] && VALIDATOR_ARGS="$VALIDATOR_ARGS --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT"

    # Start validator
    if [ "$DEBUG" = "true" ]
    then
        execute pm2 start neurons/validator.py --interpreter python3 --name ${NETWORK}_validator -- $VALIDATOR_ARGS
    else
        execute pm2 start neurons/validator.py --interpreter python3 --name ${NETWORK}_validator -- $VALIDATOR_ARGS > /dev/null 2>&1
    fi
    pdone "Validator started"
    LOGGING_TARGET="${NETWORK}_validator"
fi

if [ "$NEURON_TYPE" = "miner" ]
then
    ohai "Starting miners on network '$NETWORK' ..."
    if [ "$NUM_GPUS" -gt 0 ]
    then
        for i in $(seq 0 $((NUM_GPUS - 1)))
        do
            GPU_INDEX=$i
            HOTKEY_NAME="C$i"
            GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "$((i + 1))p")
            
            if [ -z "$GPU_MEMORY" ]
            then
                warn "Could not get GPU memory for GPU $i"
                continue
            fi
            
            # Set batch size
            if [ "$GPU_MEMORY" -ge 80000 ]
            then
                BATCH_SIZE=6
            elif [ "$GPU_MEMORY" -ge 40000 ]
            then
                BATCH_SIZE=3
            else
                BATCH_SIZE=1
            fi
            
            ohai "Starting miner on GPU $GPU_INDEX with batch size $BATCH_SIZE..."
            MINER_ARGS="--actual_batch_size $BATCH_SIZE --wallet.name default --wallet.hotkey $HOTKEY_NAME --bucket \"$BUCKET\" --device cuda:$GPU_INDEX --use_wandb --project \"$PROJECT\" --netuid $NETUID"
            
            # Add network options
            [ -n "$PM2_NETWORK_OPTIONS" ] && MINER_ARGS="$MINER_ARGS $PM2_NETWORK_OPTIONS"
            [ -n "$SUBTENSOR_NETWORK" ] && MINER_ARGS="$MINER_ARGS --subtensor.network $SUBTENSOR_NETWORK"
            [ -n "$SUBTENSOR_CHAIN_ENDPOINT" ] && MINER_ARGS="$MINER_ARGS --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT"
            
            # Start miner
            if [ "$DEBUG" = "true" ]
            then
                execute pm2 start neurons/miner.py --interpreter python3 --name ${NETWORK}_$HOTKEY_NAME -- $MINER_ARGS
            else
                execute pm2 start neurons/miner.py --interpreter python3 --name ${NETWORK}_$HOTKEY_NAME -- $MINER_ARGS > /dev/null 2>&1
            fi
        done
        LOGGING_TARGET="${NETWORK}_C0"
    else
        warn "No GPUs found. Skipping miner startup."
    fi
    pdone "All miners started"
fi

# Display status
pm2 list
echo ""
pdone "SUCCESS"
echo ""

# Start logs
pm2 logs $LOGGING_TARGET