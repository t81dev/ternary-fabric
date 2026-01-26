# gen_mem.py
def generate_ternary_mem():
    # 0x55 = 01010101 (All +1s)
    # 0xAA = 10101010 (All -1s)
    
    with open("weights.mem", "w") as f_w, open("inputs.mem", "w") as f_i:
        for addr in range(1024):
            # Weights are always +1
            f_w.write("555555\n")
            
            # Inputs toggle: First 20 cycles UP (+1), next 20 cycles DOWN (-1)
            if (addr // 20) % 2 == 0:
                f_i.write("555555\n")
            else:
                f_i.write("AAAAAA\n")

if __name__ == "__main__":
    generate_ternary_mem()
    print("✅ Memory generated: Toggling every 20 addresses.")