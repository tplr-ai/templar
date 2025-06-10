// ecosystem.config.js
require('dotenv').config({ path: '.env' });

const { execSync } = require('child_process');
const RANDOM_SUFFIX = execSync(
  "cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1"
)
  .toString()
  .trim();

const PROJECT_NAME = `test_${RANDOM_SUFFIX}`;

module.exports = {
  apps: [
    /*───────────────────────── Miner ─────────────────────────*/
    {
      /*
       * One PM2 process launches `torchrun`, which then forks the
       * required number of Python ranks (two in this example).
       */
      name: 'TM-DDP-2',            // miner, 2 distributed ranks
      script: 'bash',              // run a one-liner shell command
      interpreter: 'bash',
      args: [
        '-c',
        [
          // torchrun launch — change 2 → N to scale ranks
          'torchrun',
          '--standalone',
          '--nnodes', '1',
          '--nproc_per_node', '2',          // two ranks → two GPUs
          'neurons/miner.py',

          // miner-specific CLI flags
          '--wallet.name', 'templar_test',
          '--wallet.hotkey', 'M1',          // single hotkey is enough
          '--device', 'cuda',               // DDP selects proper GPU
          '--subtensor.network', 'local',
          '--netuid', '2',
          '--use_wandb',
          `--project "${PROJECT_NAME}"`,
        ].join(' ')
      ],
      env: {
        ...process.env,
        PROJECT_NAME
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
        '--wallet.name', 'templar_test',
        '--wallet.hotkey', 'V1',
        '--device', 'cuda:0',
        '--subtensor.network', 'local',
        '--netuid', '2',
        '--use_wandb',
        `--project "${PROJECT_NAME}"`
      ].join(' ')
    },

    /*──────────────────────── Aggregator ─────────────────────*/
    {
      name: 'Aggregator',
      script: 'neurons/aggregator.py',
      interpreter: 'python3',
      env: {
        ...process.env,
        PROJECT_NAME
      },
      args: [
        '--netuid', '3',
        '--device', 'cuda:4',
        '--project', 'templar'
      ].join(' ')
    }
  ]
};
