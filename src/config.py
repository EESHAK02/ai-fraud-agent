# src/config.py
DATA_PATH = "data/raw/creditcard.csv"

# Stream/compute caps
DEMO_MAX_ROWS = 80000          # hard cap for any run (keeps features/IF fast)
TRAIN_MAX_ROWS = 50000         # cap for IF training
TOPK_CHART_MAX = 5000          # max points to send to the chart
K = 200
ALPHA_DEFAULT = 0.6
WINDOW_SECONDS = 300
TOPK_TABLE_ROWS = 200
