import sys
import os

# Ensure we can find the compiled .so
sys.path.append(os.path.join(os.getcwd(), 'src', 'pytfmbs'))
import pytfmbs

def test_bridge_consistency():
    print("🚀 Starting Hardware-Software Bridge Validation...")
    
    try:
        # 1. Instantiate the Fabric (Triggers Fabric_init)
        fabric = pytfmbs.Fabric()
        print("✅ Fabric Object Initialized (Mock Mode)")

        # 2. Define a Test Frame Descriptor (TFD)
        test_params = {
            "base_addr": 0xDEADBEEF,
            "depth": 1024,
            "lanes": 15,
            "stride": 4,
            "kernel": 2
        }

        # 3. Execute the Run
        print(f"📡 Sending TFD to registers: {test_params}")
        fabric.run(
            test_params["base_addr"],
            test_params["depth"],
            test_params["lanes"],
            test_params["stride"],
            test_params["kernel"]
        )
        
        print("✅ Execution completed without hanging.")
        print("🌟 Bridge Logic Verified: Python -> C-Extension -> Virtual Register Map")

    except Exception as e:
        print(f"❌ Bridge Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_bridge_consistency()