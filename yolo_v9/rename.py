import os

def rename_frames_in_subfolders(root_folder, extension=".jpg"):
    """
    Renames all image files in subfolders by prefixing them with the subfolder name and a sequential number.
    
    Args:
        root_folder (str): Path to the root folder containing subfolders with frames.
        extension (str): File extension for the frames (e.g., '.jpg', '.png'). Default is '.jpg'.
    """
    # Ensure the root folder exists
    if not os.path.exists(root_folder):
        print(f"Error: The folder '{root_folder}' does not exist.")
        return

    # Iterate through each subfolder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Ensure it's a directory
        if os.path.isdir(subfolder_path):
            print(f"Renaming frames in subfolder: {subfolder}")
            
            # Get all files in the subfolder with the specified extension
            files = [f for f in os.listdir(subfolder_path) if f.endswith(extension)]
            files.sort()  # Sort files alphabetically

            # Rename each file
            for i, file in enumerate(files, start=1):
                old_path = os.path.join(subfolder_path, file)
                new_name = f"{subfolder}_{i}{extension}"
                new_path = os.path.join(subfolder_path, new_name)

                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

    print("Renaming completed!")

# Usage example
rename_frames_in_subfolders("persons", extension=".jpg")
