require('dotenv').config({ path: '.env' });
const PROJECT_NAME = `test_${RANDOM_SUFFIX}`;

module.exports = {
    apps: [
        {
            name: "TM1",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M1 --device cuda:0 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM2",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M2 --device cuda:1 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
        },
	{
	    name: "TM3",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M2 --device cuda:2 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
	},
	{
	    name: "TM4",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M2 --device cuda:3 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
	},
	{
	    name: "TM5",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M2 --device cuda:4 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
	},
	{
	    name: "TM6",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M2 --device cuda:5 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
	},
	{
	    name: "TM7",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M2 --device cuda:6 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
	},
	{
	    name: "TM8",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name Bistro --wallet.hotkey M2 --device cuda:7 --subtensor.network ws://127.0.0.1:9945 --debug --netuid 1 --use_wandb --project "${PROJECT_NAME}"`
	},
    ]
} 
