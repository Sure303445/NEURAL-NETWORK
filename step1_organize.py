import os
import shutil
import random

BASE = r"C:\Users\Welcome\OneDrive\NNDL PROJECT.CAREAI"

RAW_TB        = os.path.join(BASE, "TUBERCULOSIS DATASET")
RAW_PNEUMONIA = os.path.join(BASE, "PNEUMONIA")
RAW_NORMAL    = os.path.join(BASE, "NORMAL")
DEST          = os.path.join(BASE, "dataset_split")

def get_all_images(source_dir):
    images = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, f))
    return images

def split_dataset(source_dir, dest_dir, class_name):
    if not os.path.exists(source_dir):
        print("NOT FOUND: " + source_dir)
        return

    images = get_all_images(source_dir)
    if len(images) == 0:
        print("No images found in: " + source_dir)
        return

    print("Found " + str(len(images)) + " images in " + class_name)
    random.seed(42)
    random.shuffle(images)

    n         = len(images)
    train_end = int(n * 0.70)
    val_end   = train_end + int(n * 0.15)

    splits = {
        'train': images[:train_end],
        'val':   images[train_end:val_end],
        'test':  images[val_end:]
    }

    for split_name, file_list in splits.items():
        save_dir = os.path.join(dest_dir, split_name, class_name)
        os.makedirs(save_dir, exist_ok=True)
        for fpath in file_list:
            fname = os.path.basename(fpath)
            dst = os.path.join(save_dir, fname)
            if os.path.exists(dst):
                name, ext = os.path.splitext(fname)
                fname = name + "_" + str(random.randint(1000, 9999)) + ext
                dst = os.path.join(save_dir, fname)
            shutil.copy2(fpath, dst)
        print(split_name + "/" + class_name + ": " + str(len(file_list)) + " images")

print("Starting...")
print("")

print("Processing TB...")
split_dataset(RAW_TB, DEST, "TB")

print("")
print("Processing PNEUMONIA...")
split_dataset(RAW_PNEUMONIA, DEST, "PNEUMONIA")

print("")
print("Processing NORMAL...")
split_dataset(RAW_NORMAL, DEST, "NORMAL")

print("")
print("DONE! Checking final counts...")
print("")

total = 0
for split in ['train', 'val', 'test']:
    for cls in ['TB', 'PNEUMONIA', 'NORMAL']:
        path = os.path.join(DEST, split, cls)
        if os.path.exists(path):
            count = len(os.listdir(path))
            total += count
            print(split + "/" + cls + ": " + str(count) + " images")

print("")
print("Total images: " + str(total))
print("Ready for training!")
