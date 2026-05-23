from ultralytics import YOLO
import os

# Set max performance before starting
os.system("sudo nvpmodel -m 0")
os.system("sudo jetson_clocks")

models = [
    'yolo11n.pt',
    'yolo11n-seg.pt', 
    'yolo11n-pose.pt',
    'yolo11s.pt',
    'yolo11s-seg.pt',
    'yolo11s-pose.pt',
]

weights_dir = 'weights'

for pt_file in models:
    pt_path = f'{weights_dir}/{pt_file}'
    name = pt_file.replace('.pt', '')
    
    if not os.path.exists(pt_path):
        print(f"⚠️  Skipping {pt_file} - file not found")
        continue
    
    print(f"\n{'='*50}")
    print(f"Processing {pt_file}")
    print(f"{'='*50}")
    
    # FP32
    fp32_path = f'{weights_dir}/{name}-fp32.engine'
    if os.path.exists(fp32_path):
        print(f"⏭️  Skipping {name} FP32 - already exists")
    else:
        try:
            print(f"\n→ Exporting {name} FP32...")
            model = YOLO(pt_path)
            model.export(format='engine', half=False, int8=False, device=0,
                        workspace=4, simplify=True)
            os.rename(f'{weights_dir}/{name}.engine', fp32_path)
            print(f"✅ {name} FP32 done")
        except Exception as e:
            print(f"❌ {name} FP32 failed: {e}")

    # FP16
    fp16_path = f'{weights_dir}/{name}-fp16.engine'
    if os.path.exists(fp16_path):
        print(f"⏭️  Skipping {name} FP16 - already exists")
    else:
        try:
            print(f"\n→ Exporting {name} FP16...")
            model = YOLO(pt_path)
            model.export(format='engine', half=True, int8=False, device=0,
                        workspace=4, simplify=True)
            os.rename(f'{weights_dir}/{name}.engine', fp16_path)
            print(f"✅ {name} FP16 done")
        except Exception as e:
            print(f"❌ {name} FP16 failed: {e}")

    # INT8 - skip for seg models
    int8_path = f'{weights_dir}/{name}-int8.engine'
    if '-seg' in name:
        print(f"⚠️  Skipping {name} INT8 - not supported for seg models on TRT 10.3")
    elif os.path.exists(int8_path):
        print(f"⏭️  Skipping {name} INT8 - already exists")
    else:
        try:
            print(f"\n→ Exporting {name} INT8...")
            model = YOLO(pt_path)
            model.export(format='engine', half=False, int8=True, device=0,
                        workspace=4, simplify=True)
            os.rename(f'{weights_dir}/{name}.engine', int8_path)
            print(f"✅ {name} INT8 done")
        except Exception as e:
            print(f"❌ {name} INT8 failed: {e}")

print(f"\n{'='*50}")
print("All exports complete!")
print(f"{'='*50}")
os.system(f"ls -lah {weights_dir}/*.engine 2>/dev/null")
