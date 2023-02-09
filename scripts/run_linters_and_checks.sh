#!/bin/sh

# From the root directory, execute `./scripts/run_linters_and_checks.sh` to
# run linters. Execute `./scripts/run_linters_and_checks.sh --checks`
# to run linters and checks.
main() {
  # As a default, run linters only. Add option to run checks
  case $1 in
      -c|--checks) checks=true;
  esac

  # Run linters
  black . && isort .

  # Optionally run style checks, docstring coverage, and test coverage.
  # The results of the test coverage will additionally be saved to htmlcov.
  if [ $checks ]
  then
    flake8 . && interrogate --verbose . && coverage run -m unittest discover
    coverage report && coverage html
  fi
}

main "$@"
