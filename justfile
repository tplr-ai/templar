# Start all miners
start:
    pm2 start neurons/miner.py --interpreter python3 --name M1 --  --actual_batch_size 6 --wallet.name noa-d --wallet.hotkey C0  --device cuda:5 --use_wandb --autoupdate --process_name M1 --project gonso --sync
    pm2 start neurons/miner.py --interpreter python3 --name M2 --  --actual_batch_size 6 --wallet.name noa-d --wallet.hotkey C1  --device cuda:4 --use_wandb --autoupdate --process_name M2 --project gonso --sync
    pm2 start neurons/miner.py --interpreter python3 --name M3 --  --actual_batch_size 6 --wallet.name noa-d --wallet.hotkey C2  --device cuda:3 --use_wandb --autoupdate --process_name M3 --project gonso --sync
    pm2 start neurons/miner.py --interpreter python3 --name M4 --  --actual_batch_size 6 --wallet.name noa-d --wallet.hotkey C3  --device cuda:2 --use_wandb --autoupdate --process_name M4 --project gonso --sync
    pm2 start neurons/miner.py --interpreter python3 --name M5 --  --actual_batch_size 6 --wallet.name noa-d --wallet.hotkey C4  --device cuda:1 --use_wandb --autoupdate --process_name M5 --project gonso --sync
    pm2 start neurons/miner.py --interpreter python3 --name M6 --  --actual_batch_size 6 --wallet.name noa-d --wallet.hotkey C5  --device cuda:0 --use_wandb --autoupdate --process_name M6 --project gonso --sync

# Stop all miners
stop:
    pm2 delete M1 M2 M3 M4 M5 M6 

# Stop all miners
restart:
    pm2 restart M1 M2 M3 M4 M5 M6 Runner

# Restart all miners with clean state
restart-clean: stop restart

 
# Start testnet miners and validator
start-testnet project='gonso-2':
    pm2 start neurons/miner.py --interpreter python3 --name TM1 -- --actual_batch_size 6 --wallet.name Bistro --wallet.hotkey M111 --device cuda:1 --use_wandb --autoupdate --process_name TM1 --project {{project}} --netuid 223 --test 
    pm2 start neurons/miner.py --interpreter python3 --name TM2 -- --actual_batch_size 6 --wallet.name Bistro --wallet.hotkey M222 --device cuda:2 --use_wandb --autoupdate --process_name TM2 --project {{project}} --netuid 223 --test 
    pm2 start neurons/validator.py --interpreter python3 --name TV1 -- --actual_batch_size 6 --wallet.name Bistro --wallet.hotkey V11 --device cuda:3 --use_wandb --autoupdate --process_name TV1 --project {{project}} --netuid 223 --test 

# Stop testnet miners and validator 
stop-testnet:
    pm2 delete TM1 TM2 TV1

# Restart testnet miners and validator
restart-testnet:
    pm2 restart TM1 TM2 TV1

# Remove checkpoint folder
clean-checkpoints:
    rm -rf checkpoints/

# Clean testnet data (R2 bucket and local folders)
clean-testnet:
    python3 scripts/clean_testnet.py

# Restart testnet with clean state
restart-testnet-clean project='gonso-2': stop-testnet clean-testnet
    just start-testnet {{project}}

# Run tests with uv
test:
    export NEST_ASYNCIO=1 && uv run pytest

# Run ruff linting
lint:
    ruff check --fix

# Run code formatting
format:
    ruff format 

# Run all code quality checks and tests
check: format lint test

# Stop evaluator
stop-eval:
    pm2 delete Eval

# Start evaluator
start-eval:
    pm2 start scripts/eval.py --interpreter python3 --name Eval -- --actual_batch_size 6 --device cuda:0 --use_wandb --process_name Eval

# Restart evaluator
restart-eval:
    pm2 restart Eval


