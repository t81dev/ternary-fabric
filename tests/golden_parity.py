import numpy as np
from tools.ternary_cli import pack_pt5

def generate_golden_test(length=15):
    # 1. Generate Random Data
    weights = np.random.choice([-1, 0, 1], length)
    inputs = np.random.choice([-1, 0, 1], length)
    
    # 2. Calculate Golden Result
    golden_result = np.sum(weights * inputs)
    
    # 3. Print Verilog Stimulus
    print(f"// Golden Result Expected: {golden_result}")
    print(f"force uut.weight_byte = 8'h{pack_pt5(weights[:5])[0]:02x};")
    print(f"force uut.input_byte  = 8'h{pack_pt5(inputs[:5])[0]:02x};")
    # ... logic to check uut.vector_results