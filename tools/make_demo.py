import numpy as np
from ternary_cli import pack_pt5

# 15 Lanes: 5 positive, 5 zero, 5 negative
# Sum should be (5*1) + (5*0) + (5*-1) = 0
weights = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1])
inputs  = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1])

packed_w = pack_pt5(weights)
packed_i = pack_pt5(inputs)

print(f"Weights (PT-5): {packed_w.hex()}")
print(f"Inputs  (PT-5): {packed_i.hex()}")
print(f"Expected Sum per Lane: {weights[0]*inputs[0]} ...")