require('dotenv').config({ path: '.env' });
const RANDOM_SUFFIX = require('child_process').execSync("cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1").toString().trim();
const PROJECT_NAME = `test_${RANDOM_SUFFIX}`;

module.exports = {
    apps: [{
      name: "TM1",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name default --wallet.hotkey miner_1 --device cuda:0 --subtensor.network local --netuid 2 --enable-influxdb --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TV1",
            script: "neurons/validator.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name default --wallet.hotkey default --device cuda:1 --subtensor.network local --netuid 2 --enable-influxdb --use_wandb --project "${PROJECT_NAME}"`
        }
    ]
}
