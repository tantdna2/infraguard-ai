# Contributing to InfraGuard AI

InfraGuard AI targets Python 3.11.

## Development workflow

1. Create a dedicated branch for the GitHub Issue or task. Do not implement features directly on `main`.
2. Keep each change focused on the linked Issue.
3. Before opening a Pull Request, run:

   ```bash
   pytest
   ruff check .
   ```

4. Document any checks that could not be run.

Do not commit raw datasets, model weights, checkpoints, training runs, prediction artifacts, or credentials. Dataset and model artifacts must remain outside normal Git history.
