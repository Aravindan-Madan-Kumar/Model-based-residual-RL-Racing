# Student Guide

## Installing Pixi 
Pixi is a package management tool for Python that we will use throughout the bonus point assignments. It is used to install the required libraries and tools for the assignments. You can imagine it as a more powerful version of `pip` or `conda`, should you be familiar with those. To install Pixi, please follow the instructions on the official Pixi website:

1. Navigate to https://pixi.prefix.dev/latest/installation/
2. Ensure that you install the version for the operating system that you are using

## Creating a Pixi environment for the assignment
Once you have downloaded pixi, open a terminal in the root directory that contains all the assignment files - specifically, the `pixi.toml` and `pixi.lock` files - and run the following command:

```bash
pixi install --frozen
```

This will create a new Python environment with all the required dependencies. The `--frozen` flag ensures that the exact versions of the dependencies specified in `pixi.lock` are installed, which allows you to reproduce the same environment that will be used for grading later on.

## Activating the Pixi environment
To activate the Pixi environment, run the following command in the terminal:

```bash
pixi shell
```

You should see the name of the environment in your terminal prompt, indicating that you are now working within the Pixi environment.
```bash
(pixi-env) $
```

You can now run Python commands and JupyterLab within this environment, and it will have access to all the libraries and tools that were installed with `pixi install`:

```bash
(pixi-env) $ python --version
Python 3.x.x
```

## Getting started with car racing
Once you have installed and activated the Pixi environment, you can start working on the car racing assignment. To validate your setup and to get a feeling for the car dynamics, you can run the following command which will render the car racing environment and allow you to control the car using your keyboard:

```bash
(rllbc-bpa-3) $ python run_human.py
```

Happy racing!

## Troubleshooting
We do our best to accommodate different operating systems and setups, but if you encounter any issues during installation or while working on the assignments, please don't hesitate to reach out to the course staff through the bonus point assignment Moodle forum.