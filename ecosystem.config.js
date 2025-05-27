require('dotenv').config({ path: '.env' });
const RANDOM_SUFFIX = require('child_process').execSync("cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1").toString().trim();
const PROJECT_NAME = `test_${RANDOM_SUFFIX}`;

module.exports = {
    apps: [
        {
            name: "TM1",
            script: "/bin/bash",
            args: `-c "CUDA_VISIBLE_DEVICES=0,1 /home/templar/templar/.venv/bin/torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29501 --rdzv_id=TM1_${PROJECT_NAME} neurons/miner.py --gpus 2 --project templar --subtensor.network local --netuid 2 --wallet.name templar_test --wallet.hotkey M1 --project '${PROJECT_NAME}'"`,
            interpreter: null,
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
        },
        {
            name: "TM2",
            script: "/bin/bash",
            args: `-c "CUDA_VISIBLE_DEVICES=2,3 /home/templar/templar/.venv/bin/torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29502 --rdzv_id=TM2_${PROJECT_NAME} neurons/miner.py --gpus 2 --project templar --subtensor.network local --netuid 2 --wallet.name templar_test --wallet.hotkey M2 --project '${PROJECT_NAME}'"`,
            interpreter: null,
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
        },
        {
            name: "TV1",
            script: "neurons/validator.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name templar_test --wallet.hotkey V1 --device cuda:6 --subtensor.network local --netuid 2 --project "${PROJECT_NAME}"`
        },
        {
            name: "Aggregator",
            script: "neurons/aggregator.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--netuid 2 --device cuda:4 --project "${PROJECT_NAME}" --subtensor.network local`
        }
    ]
}
