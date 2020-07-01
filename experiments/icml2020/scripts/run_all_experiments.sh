TARGETS="AUSTRIA BELGIUM CANADA DENMARK FRANCE GERMANY GREECE IRELAND ITALY JAPAN NETHERLA NORWAY SPAIN SWEDEN SWITZERL TURKEY U.K. U.S.A."
data="gasoline"

for target in $TARGETS
do
    d="$target"_1
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_2
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_3
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_4
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_5
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_6
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_7
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_8
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_9
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
    d="$target"_0
    python run_experiment.py parallelization.data_run_id=$d data.target_domain=$target
done
