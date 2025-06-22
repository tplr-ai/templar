// ecosystem.config.js
require('dotenv').config({ path: '.env' });

const { execSync } = require('child_process');
const RANDOM_SUFFIX = execSync(
  "cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1"
)
  .toString()
  .trim();

const PROJECT_NAME = `localDeMo-1B-tests`;

module.exports = {
  apps: [
    /*───────────────────────── Miner ─────────────────────────*/
    {
      name            : "TM1",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "neurons/miner.py",
        "--wallet.name", "miner1",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "1"
      }
    },
    {
      name            : "TM2",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "neurons/miner.py",
        "--wallet.name", "miner2",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "2"
      }
    },
    {
      name            : "TM3",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "neurons/miner.py",
        "--wallet.name", "miner3",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "3"
      }
    },
    {
      name            : "TM4",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "neurons/miner.py",
        "--wallet.name", "miner4",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--desync-offset", "1",
        "--desync-after", "3",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "4"
      }
    },
    {
      name            : "TM5",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "neurons/miner.py",
        "--wallet.name", "miner5",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--desync-offset", "2",
        "--desync-after", "3",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "5"
      }
    },
    {
      name            : "TM6",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "neurons/miner.py",
        "--wallet.name", "miner6",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--max-inner-steps", "10",
        "--batches-before-local-optimization", "384",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "6"
      }
    },
    {
      name            : "TM7",
      exec_mode       : "fork",
      exec_interpreter: "none",
      script          : "torchrun",
      args: [
        "--standalone",
        "--nnodes", "1",
        "--nproc_per_node", "1",
        "neurons/miner.py",
        "--wallet.name", "miner7",
        "--wallet.hotkey", "default",
        "--device", "cuda",
        "--batches-before-local-optimization", "320",
        "--subtensor.network", "local",
        "--netuid", "2",
        "--use_wandb",
        "--project", PROJECT_NAME
      ],
      env: {
        ...process.env,
        PROJECT_NAME,
        CUDA_VISIBLE_DEVICES: "7"
      }
    },

    /*──────────────────────── Validator ──────────────────────*/
    {
      name: 'TV1',
      script: 'neurons/validator.py',
      interpreter: 'python3',
      env: {
        ...process.env,
        PROJECT_NAME
      },
      args: [
        '--wallet.name', 'validator',
        '--wallet.hotkey', 'default',
        '--device', 'cuda:0',
        '--subtensor.network', 'local',
        '--netuid', '2',
        '--use_wandb',
        `--project "${PROJECT_NAME}"`
      ].join(' ')
    }

    ///*──────────────────────── Aggregator ─────────────────────*/
    //{
    //  name: 'Aggregator',
    //  script: 'neurons/aggregator.py',
    //  interpreter: 'python3',
    //  env: {
    //    ...process.env,
    //    PROJECT_NAME
    //  },
    //  args: [
    //    '--netuid', '3',
    //    '--device', 'cuda:7',
    //    '--project', 'templar'
    //  ].join(' ')
    //}
  ]
};
