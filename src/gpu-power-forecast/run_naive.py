from data import PowerDataset
from naive_model import NaiveLastValueModel
from metrics import compute_metrics
from plots import plot_predictions

def run_naive():
    # ---- Load dataset ----
    dataset = PowerDataset("../metrics_log/merged_log.json", seq_len=60)

    # ---- Train / Val split (time-ordered) ----
    split = int(0.8 * len(dataset))
    train_idx = list(range(0, split))
    val_idx   = list(range(split, len(dataset)))

    # ---- Naive model ----
    model = NaiveLastValueModel()

    # ---- Evaluate on validation ----
    preds, actual = model.predict(dataset, val_idx)

    huber = compute_metrics(preds, actual, delta=2)

    print("Naive Baseline Results")
    print("----------------------")
    print(f"HUBER: {huber:.3f}")

    # ---- Plot ----
    plot_predictions(actual, preds)

if __name__ == "__main__":
    run_naive()
