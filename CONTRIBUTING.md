# Contributing Guide

Thank you for your interest in contributing to templar! This guide outlines our
expectations and best practices for contributions.

## Table of Contents

- [Git Workflow](#git-workflow)
  - [Branch Naming Convention](#branch-naming-convention)
  - [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
  - [Merge Strategy](#merge-strategy)
  - [Applying Review Comments](#applying-review-comments)
  - [Reviewing](#reviewing)
- [Code Style](#code-style)
  - [Python](#python)
    - [Docstrings](#docstrings)
    - [Logging and Error Messages](#logging-and-error-messages)
- [Testing](#testing)
- [Tools](#tools)
  - [Python Package Manager](#python-package-manager)
  - [Linting and Formatting](#linting-and-formatting)

## Git Workflow

Using `git` effectively is essential for collaboration on this project. The
following guidelines will help maintain consistency and clarity.

### Branch Naming Convention

Follow this pattern for branch names:

```
<kind>/<description>
```

Where:
- `kind` is one word denoting the type of work
- `description` consists of multiple words separated by hyphens (`-`)

Common kinds include:

Kind         | Explanation
-------------|----------------------------------------------------------------
`feature`    | Used when working on or adding a new feature
`docs`       | Used when working on project documentation
`fix`        | Used when fixing issues or bugs
`maintenance`| Used when updating third-party packages
`refactor`   | Used when refactoring code
`tests`      | Used when adding or changing tests

Examples:
```
✅ Good: docs/add-branching-naming-convention
❌ Bad:  add-branching-naming-convention
❌ Bad:  docs/add_branching_naming_convention
```

### Commit Messages

Write clear, concise commit messages following this model:

```
Capitalized, short (50 chars or less) summary

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of an email and the rest of the text as the body. The blank
line separating the summary from the body is critical (unless you omit
the body entirely); tools like rebase can get confused if you run the
two together.

Write your commit message in the imperative: "Fix bug" and not "Fixed bug"
or "Fixes bug." This convention matches up with commit messages generated
by commands like git merge and git revert.

Further paragraphs come after blank lines.

- Bullet points are okay, too

- A hyphen is used for the bullet, followed by a single space, with blank
  lines in between

- Use a hanging indent
```

Before submitting a pull request, ensure that your commits form an
easy-to-follow narrative to assist reviewers.

For more insights on writing effective commits, see:
- [Write Better Commits, Build Better Projects](https://github.blog/2022-06-30-write-better-commits-build-better-projects/)
- [A Branch in Time (a story about revision histories)](https://tekin.co.uk/2019/02/a-talk-about-revision-histories)

## Pull Requests

Pull requests are central to collaboration and knowledge sharing in this project.

### Merge Strategy

To maintain a clean commit history on the `main` branch:

1. Make small, atomic commits that serve a single purpose
2. Format commit messages properly as described in the [Commit Messages](#commit-messages) section
3. Avoid introducing merge commits except when merging into `main`

If you need to incorporate the latest changes from `main` into your branch,
   rebase instead of merging:

```bash
git rebase main
git push --force-with-lease
```

Performing this operation before merging is recommended to keep your branch
based on the latest `main`.

### Applying Review Comments

When addressing review comments:

1. Use interactive rebase to organize your commits:
   ```bash
   git rebase -i main
   ```

2. Use the `--fixup` flag when committing changes:
   ```bash
   git commit --fixup=<commit-sha>
   ```

3. After approval, squash the modifications:
   ```bash
   git rebase -i main --autosquash
   ```

4. Push the changes:
   ```bash
   git push --force-with-lease
   ```

**Important:** Don't push any `fixup` commits to the `main` branch. Always
squash them before merging.

### Reviewing

When reviewing a pull request, try to consolidate your comments into a single
review when possible to avoid unnecessary notifications for contributors.

## Code Style

### Python

Python code should follow the [Google Python Style
Guide](https://google.github.io/styleguide/pyguide.html) with the exceptions
noted below.

#### Docstrings

Summary-only docstrings are permitted when function arguments and return
annotations are self-explanatory.

#### Logging and Error Messages

Use the project's logger consistently throughout the codebase:

```python
import tplr

# Example of logging an info message
tplr.logger.info("Processing data file")
```

Logging and error messages should:
- Be clear and concise
- Use proper capitalization and grammar
- **Not** end with a period

**Correct:**
```
"Failed to load data"
```

**Incorrect:**
```
"Failed to load data."
```



## Testing

Use `pytest` for testing. Structure your tests to mirror the project structure:

```
└── tests
    └── unit
        └── sub_package
            ├── conftest.py  # if applicable
            ├── module_A
            │   ├── test_module_A_1.py
            │   └── test_module_A_2.py
            └── module_B
                └── test_module_B.py
```

Keep tests clear, focused, and maintainable. Each test should verify a specific aspect of functionality and have an obvious purpose.

## Tools

### Python Package Manager

You may use any Python package manager, but `uv` is recommended for consistent environments:

Installation (macOS and Linux):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Common commands:
- Update `uv`: `uv self update`
- Install default dependencies: `uv sync --frozen`
- Install all development dependencies: `uv sync --all-extras --frozen`
- Upgrade dependencies: `uv sync --upgrade`

For more information, see the [uv documentation](https://docs.astral.sh/uv/).

### Linting and Formatting

`Ruff` handles linting and formatting. Configuration for this tool is defined
in `pyproject.toml`.
