# Grafana-Checks Role

This role performs comprehensive verification of a Grafana stack deployment to ensure everything is working as expected.

## Features

- Validates all Grafana stack components (Grafana, Nginx, plugins, datasources, dashboards)
- Generates detailed verification reports
- Can be run independently with the `checks` tag
- Configurable to fail on errors or just report them

## Usage

The role is included in the main playbook and will run by default as part of the deployment.
You can also run the checks independently using tags:

```bash
# Run only the verification checks
ansible-playbook -i inventory playbook.yml --tags checks

# Run the full deployment but skip verification
ansible-playbook -i inventory playbook.yml --skip-tags checks,verification
```

The role has two tags:
- `checks`: For running only the verification checks
- `verification`: Included in the default run

## Configuration

You can customize the role's behavior with these variables:

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `fail_on_verification_error` | Whether to fail the playbook if verification checks fail | `false` |
| `save_verification_results` | Whether to save verification results to files | `true` |
| `verification_output_dir` | Directory to save verification results (on the control machine) | `playbook_dir/verification_results` |

## Reports

The role generates a detailed verification report with the following information:

- Overall status (SUCCESS/FAILED)
- Detailed checks with pass/fail indicators
- System information
- Timestamp

Reports are saved in the `verification_output_dir` with a timestamp in the filename.
