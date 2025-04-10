require('dotenv').config(); // Loads environment variables from .env

const NUM_MINERS = process.env.NUM_MINERS || 3;
const PROJECT_NAME = `test-scoring`;
const LOCAL_FLAG = process.env.LOCAL === 'true' ? ' --local' : '';

const minerConfigs = [...Array(Number(NUM_MINERS))].map((_, index) => ({
    name: `TM${index}`,
    script: "neurons/miner.py",
    interpreter: "python3",
    env: {
        ...process.env,
        PROJECT_NAME: PROJECT_NAME
    },
        args: `--wallet.name miner${index + 1} --wallet.hotkey default --device cuda:${index} --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
}));

module.exports = {
    apps: [
        ...minerConfigs,
        {
            name: "TV1",
            script: "neurons/validator.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name validator --wallet.hotkey default --device cuda:3 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        }
    ]
}
