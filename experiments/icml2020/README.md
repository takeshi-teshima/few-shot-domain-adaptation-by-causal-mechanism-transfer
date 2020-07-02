# Experiments
This directory contains the code for reproducing the experimental results.

# Preparation
This experiment depends on MongoDB to store the intermediate results.

1. Install this repository by following the [README.md](https://github.com/takeshi-teshima/few-shot-domain-adaptation-by-causal-mechanism-transfer/blob/master/README.md).

1. Install the requirements: `pip install -r requirements.txt`.
   (In case some dependencies are missing, please see `full_requirements.txt`)

1. Install MongoDB via tarball
  ```sh
  $ wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-3.6.3.tgz
  $ tar xaf mongodb-linux-x86_64-3.6.3.tgz
  ```
  and set `path`.
  (Sidenote: the installation via homebrew / linuxbrew did not work for me).

1. Modify the contents of `scripts/config.sh` to match your local environment (to specify where the database files will be stored).

1. Run `$ scripts/mongo.sh` to start the MongoDB process (the script requires the `tmux` package. If you don't have it, install it or start mongo with your own script).

1. Create appropriate users and tables.
  ```sh
  $ mongo
  > use icml2020
  > db.createUser({user:'me',pwd:'pass',roles:[{role:'dbOwner',db:'icml2020'}]})
  > use sacred
  > db.createUser({user:'me',pwd:'pass',roles:[{role:'dbOwner',db:'sacred'}]})
  ```

### Note
- Don't install `bson` package. It has a conflicting name with the `PyMongo` package.

# Running the experiment

- Modify the experiment condition in `config/`. At least, `config/database.yml` needs to be modified for the script to run correctly.
- Run the experiment by

```bash
$ python run_experiment.py

# Or debug mode (exceptions are emitted to outside of the experiment loop)
$ python run_experiment.py debug=True method.ica_train.max_epochs=20
```

## Checking the results
1. To check the results run by Sacred, modify the contents in `scripts/omniboard.sh` and run
  ```bash
  $ scripts/omniboard.sh
  ```

2. I also used DBeaver to check the contents in the MongoDB where the results are stored.
   To setup Omniboard, run
   ```bash
   $ brew install npm
   $ npm install -g omniboard
   ```

## Formatting the results into a LaTeX table.
- Run `$ jupyter notebook` to check the script to generate the table in the paper. (The compiled table is output in `output/`).
  Originally, a MongoDB database was used, but the records are pickled under `pickle/` here.

# Re-downloading the data
- See the README of each `data/<dataname>_raw` directory.
