#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import sys, torch, pandas, yaml
print('python',sys.version)
print('torch',torch.__version__,'cuda_available',torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu',torch.cuda.get_device_name(0))
print('pandas',pandas.__version__,'pyyaml',yaml.__version__)
PY
