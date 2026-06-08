# GPU 對照表

PyTorch 預設排序 (FASTEST_FIRST) ≠ nvidia-smi 實體 index，所以 **`--device cuda:N` 不等於實體 GPU N**。
以下為 **UUID 比對驗證**（2026-06-08，換過 GPU 後重抓）。硬體有變動就重跑下方指令更新。

## 你要寫的 `--device`  →  實際的實體 GPU (nvidia-smi index)
| `--device` | 實體 GPU |
|---|---|
| cuda:0 | 實體 7 |
| cuda:1 | 實體 0 |
| cuda:2 | 實體 1 |
| cuda:3 | 實體 2 |
| cuda:4 | 實體 3 |
| cuda:5 | 實體 4 |
| cuda:6 | 實體 5 |
| cuda:7 | 實體 6 |
| cuda:8 | 實體 8 |
| cuda:9 | 實體 9 |

## 反查：實體 GPU (nvidia-smi) → 要寫的 `--device`
| 實體 (REAL) | CODE (`cuda:N`) |
|---|---|
| 0 | cuda:1 |
| 1 | cuda:2 |
| 2 | cuda:3 |
| 3 | cuda:4 |
| 4 | cuda:5 |
| 5 | cuda:6 |
| 6 | cuda:7 |
| 7 | cuda:0 |
| 8 | cuda:8 |
| 9 | cuda:9 |

## 重新產生此表（換硬體後執行）
```bash
python3 - <<'PY'
import torch, subprocess
smi={}
for ln in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"]).decode().splitlines():
    idx,uuid=[x.strip() for x in ln.split(",")]; smi[uuid]=int(idx)
for i in range(torch.cuda.device_count()):
    u="GPU-"+str(torch.cuda.get_device_properties(i).uuid)
    print(f"cuda:{i} -> 實體 {smi.get(u,'?')}")
PY
```
