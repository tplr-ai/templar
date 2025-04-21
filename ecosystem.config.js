require('dotenv').config();
const child_process = require('child_process');
const NUM_MINERS = process.env.NUM_MINERS || 7; // 7 miners with different configurations
const PROJECT_NAME = `test-openskill`;

// Get the latest commit hash (first 7 characters)
let commitHash = 'unknown';
try {
    commitHash = child_process.execSync('git rev-parse --short HEAD').toString().trim();
    console.log(`Using commit hash: ${commitHash}`);
} catch (error) {
    console.error('Failed to get git commit hash:', error.message);
}

// Common warmup windows for all desynced miners
const WARMUP_WINDOWS = process.env.WARMUP_WINDOWS || 10;

// Define miner configurations - desync windows and pages
const minerConfigList = [
    { desync: 0, pages: 6, prefix: 'baseline' },     // UID 2: baseline
    { desync: 0, pages: 6, prefix: 'baseline' },     // UID 3: baseline
    { desync: 0, pages: 6, prefix: 'baseline' },     // UID 4: baseline
    { desync: 0, pages: 12, prefix: '12page-sync' }, // UID 5: 12 pages sync
    { desync: 1, pages: 6, prefix: 'desync-1' },     // UID 6: 1 windows desync
    { desync: 2, pages: 6, prefix: 'desync-2' },     // UID 7: 2 windows desync
    { desync: 3, pages: 6, prefix: 'desync-3' }      // UID 8: 3 windows desync
];

const minerConfigs = [...Array(Math.min(NUM_MINERS, minerConfigList.length))].map((_, index) => {
    const config = minerConfigList[index];
    
    // Set desync args based on configuration
    const desyncArg = config.desync > 0 ? 
        ` --desync ${config.desync} --warmup ${WARMUP_WINDOWS}` : '';
    
    // Set pages arg if specified
    const pagesArg = config.pages ? ` --pages ${config.pages}` : '';
    
    return {
        name: `TM${index}`,
        script: "neurons/miner.py",
        interpreter: "python3",
        env: {
            ...process.env,
            PROJECT_NAME: PROJECT_NAME,
            COMMIT_HASH: commitHash
        },
        args: `--wallet.name miner${index + 1} --wallet.hotkey default --device cuda:${index + 1} --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"${pagesArg}${desyncArg} --name_prefix "${commitHash}-${config.prefix}"`
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
            args: `--wallet.name validator --wallet.hotkey default --device cuda:0 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}" --name_prefix "${commitHash}"`
        }
    ]
}
