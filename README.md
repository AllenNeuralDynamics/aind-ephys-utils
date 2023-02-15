# aind-ephys-utils

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)

Helpful methods for exploring *in vivo* electrophysiology data.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o doc_template/source/ src/aind_ephys_utils
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).


## Developing in Code Ocean

Members of the Allen Institute for Neural Dynamics can follow these steps to create a Code Ocean capsule from this repository:

1. Click the **⨁ New Capsule** button and select "Clone from AllenNeuralDynamics"
2. Type in `aind-ephys-utils` and click "Clone" (this step requires that your GitHub credentials are configured properly)
3. Select a Python base image, and optionally change the compute resources
4. Attach data to the capsule and any dependencies needed to load it (e.g. `pynwb`, `hdmf-zarr`)
5. Add plotting dependencies (e.g. `ipympl`, `plotly`)
6. Launch a Visual Studio Code cloud workstation

Inside Visual Studio Code, select "New Terminal" from the "Terminal" menu and run the following commands:

```bash
$ pip install -e .[dev]
$ git checkout -b <name of feature branch>
```

Now, you can create Jupyter notebooks in the "code" directory that can be used to test out new functions before updating the library. When prompted, install the "Python" extensions to be able to execute notebook cells.

Once you've finished writing your code and tests, run the following commands:

```bash
$ coverage run -m unittest discover && coverage report
$ interrogate . 
$ black .
$ flake8 .
$ isort .
```

Assuming all of these pass, you're ready to push your changes:

```bash
$ git add <files to add>
$ git commit -m "Commit message"
$ git push -u origin <name of feature branch>
```

After doing this, you can open a pull request on GitHub.

Note that `git` will only track files inside the `aind-ephys-utils` directory, and will ignore everything else in the capsule. You will no longer be able to commit changes to the capsule itself, which is why this workflow should only be used for developing a library, and not for performing any type of data analysis.

When you're done working, it's recommended to put the workstation on hold rather than shutting it down, in order to keep Visual Studio Code in the same state.



