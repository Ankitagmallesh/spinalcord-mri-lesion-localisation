from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import ML_MODEL, DATA_ROOT, SpinalCordMRIDataset, _compute_metrics_from_dataset

print('ML_MODEL loaded:', ML_MODEL is not None)
print('DATA_ROOT exists:', DATA_ROOT.exists())
print('DATA_ROOT path:', DATA_ROOT)
try:
    ds = SpinalCordMRIDataset(DATA_ROOT, image_size=256)
    print('dataset length:', len(ds))
except Exception as e:
    print('dataset error:', type(e).__name__, e)
try:
    metrics = _compute_metrics_from_dataset()
    print('compute_metrics_from_dataset returned:', type(metrics).__name__)
    if metrics is not None:
        print('metrics keys:', list(metrics.keys()))
        print('confusion_matrix:', metrics['confusion_matrix'])
        print('accuracy:', metrics['classification_report'].get('accuracy'))
except Exception as e:
    print('compute metrics error:', type(e).__name__, e)
