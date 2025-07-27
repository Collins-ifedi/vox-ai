import os
import pandas_ta

sqz_file = os.path.join(os.path.dirname(pandas_ta.__file__), "momentum", "squeeze_pro.py")

print(f"[PATCH] Attempting to patch {sqz_file}...")

# Read and replace faulty import
with open(sqz_file, "r") as f:
    code = f.read()

patched_code = code.replace(
    "from numpy import NaN as npNaN",
    "import numpy as np\nnpNaN = np.nan"
)

# Only write if changes are needed
if "from numpy import NaN as npNaN" in code:
    with open(sqz_file, "w") as f:
        f.write(patched_code)
    print(f"[PATCH] Successfully patched {sqz_file}")
else:
    print(f"[PATCH] No changes needed (already patched).")