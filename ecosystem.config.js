const NUM_MINERS = process.env.NUM_MINERS || 5;
const PROJECT_NAME = `templar-test`;
const LOCAL_FLAG = process.env.LOCAL === 'true' ? ' --local' : '';

// Create array of miner configurations
const minerConfigs = [...Array(Number(NUM_MINERS))].map((_, index) => ({
    name: `TM${index}`,
    script: "neurons/miner.py",
    interpreter: "python3",
    env: {
        ...process.env,
        PROJECT_NAME: PROJECT_NAME
    },
    args: `--wallet.name miner --wallet.hotkey M${index} --device cpu --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"${LOCAL_FLAG}`
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
            args: `--wallet.name validator --wallet.hotkey default --device cpu --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"${LOCAL_FLAG}`
        }
    ]
}
