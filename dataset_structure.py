import os

dataset_path = r"D:\Milk_adulteration_images"

print("\n===== COMPLETE DIRECTORY TREE =====\n")

for root, dirs, files in os.walk(dataset_path):

    level = root.replace(dataset_path,'').count(os.sep)

    indent = "    "*level

    print(f"{indent}📁 {os.path.basename(root)}")

    for d in dirs:
        print(f"{indent}    ├── Folder: {d}")

    for f in files[:10]:
        print(f"{indent}    ├── File: {f}")

print("\n===== END =====")