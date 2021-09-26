export WANDB_BASE_URL=http://localhost:7001/
export WANDB_API_KEY=local-d90ae648a77515f035a960bace7ef52db9cb7638

python3 train.py    \
    --data-version "$1" \
    --code-version "$2" \
    --pret-version "$3"