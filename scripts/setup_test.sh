apt update && apt install --assume-yes sudo tmux
sudo apt install --assume-yes make build-essential git clang curl libssl-dev llvm libudev-dev protobuf-compiler pkg-config libssl-dev tmux nodejs npm neovim iputils-ping

git clone https://github.com/opentensor/subtensor.git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup target add wasm32-unknown-unknown --toolchain stable-x86_64-unknown-linux-gnu
rustup component add rust-src --toolchain stable-x86_64-unknown-linux-gnu

cd subtensor && git checkout v2.0.11 && cd ..
subtensor/scripts/init.sh 
tmux new-session -d -s localnet 'cd subtensor && ./scripts/localnet.sh False --no-purge'

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv sync --all-extras
source .venv/bin/activate
uv pip install -e .[dev]
cargo install just
npm install -g pm2
npm install dotenv

git config --global user.email "test@tplr.ai"
git config --global user.name "remote testing"

python -c "import time; time.sleep(60*3)"

uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_coldkey --wallet.name owner -p ~/.bittensor/wallets --n-words 24 --no-use-password
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_coldkey --wallet.name templar_test -p ~/.bittensor/wallets --n-words 24 --no-use-password

uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name owner --wallet.hotkey default --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey V1 --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M1 --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M2 --n-words 24

uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet faucet --wallet.name owner --max-successes 1 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet faucet --wallet.name owner --max-successes 1 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet faucet --wallet.name templar_test --max-successes 1 --network local -y

uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet create --wallet.name owner --wallet.hotkey default --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M1 --netuid 2 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M2 --netuid 2 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey V1 --netuid 2 --network local -y

uvx --from bittensor-cli==9.7.1 --with torch  btcli stake add --wallet.name templar_test --wallet.hotkey V1 --netuid 2 --network local --unsafe -y

uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M3 --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M4 --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M5 --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M6 --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M7 --n-words 24
uvx --from bittensor-cli==9.7.1 --with torch  btcli wallet new_hotkey --wallet.name templar_test --wallet.hotkey M8 --n-words 24

uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M3 --netuid 2 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M4 --netuid 2 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M5 --netuid 2 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M6 --netuid 2 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M7 --netuid 2 --network local -y
uvx --from bittensor-cli==9.7.1 --with torch  btcli subnet register --wallet.name templar_test --wallet.hotkey M8 --netuid 2 --network local -y

