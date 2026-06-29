import os
import shutil

# List of folder names (files inside will be deleted)
folders = [
    "output_data/data_raw/excel/",
    "output_data/data_raw/txt/",
    "output_data/emissions/",
    "output_data/trajectory",
]

# Folder where also subfolders should be deleted
scenario_folder = "input_data/scenarios/"

# File extensions to delete in current directory
extensions_to_delete = [".inp", ".out", ".plt"]


def delete_files_in_folders(folder_list):
    for folder in folder_list:
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass


def delete_folder_with_subfolders(folder):
    if os.path.exists(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception:
                pass


def delete_files_in_current_dir(extensions):
    for file in os.listdir("."):
        if any(file.endswith(ext) for ext in extensions):
            try:
                os.remove(file)
            except Exception:
                pass


# Prompt user for confirmation
def confirm_deletion():
    response = input(
        "Are you sure you want to delete the files and folders? (yes/no): "
    ).strip().lower()
    return response == "yes"


# Main function
def main():
    if confirm_deletion():
        delete_files_in_folders(folders)
        delete_folder_with_subfolders(scenario_folder)
        delete_files_in_current_dir(extensions_to_delete)
        print("Files and folders deleted successfully.")
    else:
        print("Operation canceled. No files were deleted.")


if __name__ == "__main__":
    main()
