# Experiments
This directory contains the code for reproducing the experimental results.

# Preparation
This experiment depends on MongoDB to store the intermediate results.

1. Install MongoDB via tarball
  ```sh
  $ wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-3.6.3.tgz
  $ tar xaf mongodb-linux-x86_64-3.6.3.tgz
  ```
  and set `path`.
  (Sidenote: the installation via homebrew / linuxbrew did not work for me).

2. Create appropriate users  and tables etc.
```sh
$ mongo
> use icml2020
> db.createUser({user:'me',pwd:'pass',roles:[{role:'dbOwner',db:'icml2020'}]})
> use sacred
> db.createUser({user:'me',pwd:'pass',roles:[{role:'dbOwner',db:'sacred'}]})
```

2. Modify the contents of `scripgts/config.sh` and run `$ scripts/mongo.sh`.

### Note
- Don't install `bson` package. It has a conflicting name with the `PyMongo` package.

# Experiment

- Modify the experiment condition in `config/`. At least, `config/database.yml` needs to be modified for the script to run correctly.
- Run the experiment by

```bash
$ python run_experiment.py
```

## Check the results
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

## Format the results into a LaTeX table.
- Run `$ jupyter notebook` to check the script to generate the table in the paper. (The compiled table is output in `output/`).
  Originally, a MongoDB database was used, but the records are pickled under `pickle/` here.

# Data
The raw data can be downloaded from:
- [World Gasoline Demand Data](http://people.stern.nyu.edu/wgreene/Econometrics/PanelDataSets.htm)
