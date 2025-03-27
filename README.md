# Research project
This is a template for the repository of a reseach project containing research code. It provides the folder structure and the pre-commit hooks, and it assumes you are using uv as your packaging manager.

1. After creating a new repo off this one, initialize a uv environment by running `uv init --python 3.12`. This creates the environment and adds all uv-related files to the repo.
2. To set up the pre-commit hooks, activate your uv environment with `source .venv/bin/activate`, and then run `make install_hooks`.
3. Run `make install_jupyter` to get jupyter working.
4. Run `make install_python_basics` to install some python basic files.
5. For the installable package, a name has to be choosen, the `src/` folder renamed, and the `mypyproject.toml` file and notebook imports edited accordingly. Afterwards, in the uv environment, run `uv pip install -e .`.