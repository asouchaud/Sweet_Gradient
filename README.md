# Sweet_Gradient

![Tests](https://github.com/asouchaud/Sweet_Gradient/actions/workflows/tests.yml/badge.svg?branch=main)

Sweet Gradient provides a framework for optimisation of multichannel using perceptual loss functions, such as models for localisation, uncertainty, and colouration.

## Installation

Make sure you have Python@3.12 installed (on Mac, run `brew install Python@3.12`).
Then create a virtual environment and install requirements:

```bash
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run 
The next step is to run tests to check everything is running smoothly and then start running a simulation (see contents of main.py for more information):

```bash
python run_tests.py
python main.py
```

## Project layout

-   `main.py` runs optimisation and saves results
-   `scripts/` contains propagation, perceptual models, optimisation,
    plotting, and saving code
-   `tests/` contains unit and integration tests
-   `data/` contains SOFA files, filter banks, and example stimuli