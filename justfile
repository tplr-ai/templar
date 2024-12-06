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


