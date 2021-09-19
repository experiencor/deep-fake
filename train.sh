export WANDB_BASE_URL=http://localhost:7001/
export WANDB_API_KEY=local-8f837cae7ccc9fc0011083e44b8c794842ffb364

python3 train.py    \
    --data-version "$1" \
    --code-version "$2" \
    --pret-version "$3"