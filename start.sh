

pm2 delete all
pm2 start run.py --interpreter python3 --name M0 -- --wallet.hotkey 0 --device cuda:0 --subtensor.network test --use_wandb --peers 2 3 4 5 6 7
pm2 start run.py --interpreter python3 --name M1 -- --wallet.hotkey 1 --device cuda:1 --subtensor.network test --use_wandb --peers 2 3 4 5 6 7
pm2 start run.py --interpreter python3 --name M2 -- --wallet.hotkey 2 --device cuda:2 --subtensor.network test --use_wandb --peers 2 3 4 5 6 7
pm2 start run.py --interpreter python3 --name V3 -- --wallet.hotkey 3 --device cuda:3 --subtensor.network test --use_wandb --peers 2 3 4 5 6 7 --is_validator
pm2 start run.py --interpreter python3 --name M4 -- --wallet.hotkey 4 --device cuda:4 --subtensor.network test --use_wandb --peers 2 3 4 5 6 7 --random
pm2 start run.py --interpreter python3 --name M5 -- --wallet.hotkey 5 --device cuda:5 --subtensor.network test --use_wandb --peers 2 3 4 5 6 7 --random
