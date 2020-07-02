source ./config.sh
mkdir -p $MONGODB_DBPATH
echo Starting mongodb at $MONGODB_DBPATH
tmux kill-session -t mongodb &
tmux new-session -d -s mongodb
tmux send-keys "mongod --dbpath="$MONGODB_DBPATH ENTER
