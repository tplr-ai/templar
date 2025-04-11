require('dotenv').config();
const child_process = require('child_process'); // Correctly import child_process
const NUM_MINERS = process.env.NUM_MINERS || 3;
const PROJECT_NAME = `test-scoring`;

// Special miner configuration (the one that will use more pages)
const SPECIAL_MINER_INDEX = process.env.SPECIAL_MINER_INDEX || 1;
const SPECIAL_MINER_PAGES = process.env.SPECIAL_MINER_PAGES || 1;
const SPECIAL_MINER_PREFIX = process.env.SPECIAL_MINER_PREFIX || '1page';

// Get the latest commit hash (first 7 characters)
let commitHash = 'unknown';
try {
    commitHash = child_process.execSync('git rev-parse --short HEAD').toString().trim();
    console.log(`Using commit hash: ${commitHash}`);
} catch (error) {
    console.error('Failed to get git commit hash:', error.message);
}

const minerConfigs = [...Array(Number(NUM_MINERS))].map((_, index) => {
    const isSpecialMiner = (index + 1) === Number(SPECIAL_MINER_INDEX);
    
    // Set special flags only for the special miner
    const pagesArg = isSpecialMiner ? ` --pages ${SPECIAL_MINER_PAGES}` : '';
    
    // Include commit hash in the name prefix
    const normalPrefix = `commit-${commitHash}-baseline`;
    const specialPrefix = `commit-${commitHash}-${SPECIAL_MINER_PREFIX}`;
    const prefixArg = isSpecialMiner ? ` --name_prefix "${specialPrefix}"` : ` --name_prefix "${normalPrefix}"`;
    
    return {
        name: `TM${index}`,
        script: "neurons/miner.py",
        interpreter: "python3",
        env: {
            ...process.env,
            PROJECT_NAME: PROJECT_NAME,
            COMMIT_HASH: commitHash
        },
        args: `--wallet.name miner${index + 1} --wallet.hotkey default --device cuda:${index} --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"${pagesArg}${prefixArg}`
    };
});

module.exports = {
    apps: [
        ...minerConfigs,
        {
            name: "TV1",
            script: "neurons/validator.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME,
                COMMIT_HASH: commitHash
            },
            args: `--wallet.name validator --wallet.hotkey default --device cuda:3 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}" --name_prefix "${commitHash}-baseline"`
        }
    ]
}
