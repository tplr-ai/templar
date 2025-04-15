require('dotenv').config();
const child_process = require('child_process');
const NUM_MINERS = process.env.NUM_MINERS || 3;
const PROJECT_NAME = `test-trueskill`;

// Special miner configuration (the one that will use more pages)
const SPECIAL_MINER_INDEX = process.env.SPECIAL_MINER_INDEX || 1;
const SPECIAL_MINER_PAGES = process.env.SPECIAL_MINER_PAGES || 12;
const SPECIAL_MINER_PREFIX = process.env.SPECIAL_MINER_PREFIX || '12page';

// Desync miner configuration
const DESYNC_MINER_INDEX = process.env.DESYNC_MINER_INDEX || 61;
const DESYNC_WINDOWS = process.env.DESYNC_WINDOWS || 3;
const WARMUP_WINDOWS = process.env.WARMUP_WINDOWS || 10;
const DESYNC_MINER_PREFIX = process.env.DESYNC_MINER_PREFIX || 'desync';

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
    const isDesyncMiner = (index + 1) === Number(DESYNC_MINER_INDEX);
    
    // Set special flags for the special miner
    const pagesArg = isSpecialMiner ? ` --pages ${SPECIAL_MINER_PAGES}` : '';
    
    // Set desync flags for the desync miner
    const desyncArg = isDesyncMiner ? ` --desync ${DESYNC_WINDOWS} --warmup ${WARMUP_WINDOWS}` : '';
    
    // Include commit hash in the name prefix
    const normalPrefix = `commit-${commitHash}-baseline`;
    const specialPrefix = `commit-${commitHash}-${SPECIAL_MINER_PREFIX}`;
    const desyncPrefix = `commit-${commitHash}-${DESYNC_MINER_PREFIX}`;
    
    let prefixArg = ` --name_prefix "${normalPrefix}"`;
    if (isSpecialMiner) {
        prefixArg = ` --name_prefix "${specialPrefix}"`;
    } else if (isDesyncMiner) {
        prefixArg = ` --name_prefix "${desyncPrefix}"`;
    }
    
    return {
        name: `TM${index}`,
        script: "neurons/miner.py",
        interpreter: "python3",
        env: {
            ...process.env,
            PROJECT_NAME: PROJECT_NAME,
            COMMIT_HASH: commitHash
        },
        args: `--wallet.name miner${index + 1} --wallet.hotkey default --device cuda:${index} --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"${pagesArg}${desyncArg}${prefixArg}`
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
