import os

# Specify the folder path
folder_path = "data/val"

# List subfolders in the folder
subfolders = sorted([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))])

# Subfolder names to search for
target_names = ["n04552348", "n04285008", "n01530575", "n02123597", "n02422699", 
                "n02107574", "n01641577", "n03417042", "n02389026", "n03095699"]

# Find indexes of the target subfolder names
indexes = {name: subfolders.index(name) for name in target_names if name in subfolders}

print("Subfolders:", subfolders[0:10])
print("Indexes of target names:", indexes)

items = []
for key, item in indexes.items():
    items.append(item)
print(items)