"""Out-of-sample accuracy metrics for wastewater-based forecasts."""
import numpy as np


def metrics(pred_rows):
    y = pred_rows['y_true'].to_numpy()
    p = pred_rows['y_pred'].to_numpy()
    b = pred_rows['y_persist'].to_numpy()
    mae = float(np.mean(np.abs(y - p)))
    mae_base = float(np.mean(np.abs(y - b)))
    true_dir = np.sign(y - b)
    pred_dir = np.sign(p - b)
    covered = (y >= pred_rows['lo'].to_numpy()) & (y <= pred_rows['hi'].to_numpy())
    return {
        'n_predictions': int(len(y)),
        'mae_model': round(mae, 4),
        'mae_persistence': round(mae_base, 4),
        'skill_vs_persistence_pct': round((1 - mae / mae_base) * 100, 1),
        'rmse_model': round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
        'direction_accuracy_pct': round(float(np.mean(true_dir == pred_dir)) * 100, 1),
        'interval_90_coverage_pct': round(float(np.mean(covered)) * 100, 1),
    }


def metrics_by_site(pred_rows):
    return {sid: metrics(g) for sid, g in pred_rows.groupby('site_id')}
