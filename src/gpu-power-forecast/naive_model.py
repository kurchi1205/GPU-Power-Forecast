import numpy as np

class NaiveLastValueModel:
    """
    Predicts next power value as the last observed power.
    """

    def predict(self, dataset, indices):
        preds = []
        actual = []

        # Get the last observed value from the first sequence
        if len(indices) > 0:
            first_idx = indices[0]
            last_observed_value = dataset.power[first_idx + dataset.seq_len - 1]
        
        for i in indices:
            # Use the last observed value for all predictions (flat line)
            y_pred = last_observed_value
            # The actual next value to predict
            y_true = dataset.power[i + dataset.seq_len]

            preds.append(y_pred)
            actual.append(y_true)

        return np.array(preds), np.array(actual)