

# pm2 delete all
pm2 start neurons/miner.py --interpreter python3 --name TM0 -- --wallet.name Bistro --wallet.hotkey M111 --device cuda:0 --subtensor.network test --use_wandb --debug
pm2 start neurons/miner.py  --interpreter python3 --name TM1 -- --wallet.name Bistro --wallet.hotkey M222 --device cuda:1 --subtensor.network test --use_wandb --debug
pm2 start neurons/validator.py  --interpreter python3 --name TV1 -- --wallet.name Bistro --wallet.hotkey V11 --device cuda:3 --subtensor.network test --use_wandb --debug
