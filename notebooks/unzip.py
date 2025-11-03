import os
import zipfile
import sys

def unzip_all_in_tree(root_folder: str):
    """
    Recursively searches for .zip files under root_folder and extracts each
    into its containing directory.
    """
    if not os.path.isdir(root_folder):
        print(f"Error: '{root_folder}' is not a valid directory.")
        return

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".zip"):
                zip_path = os.path.join(dirpath, filename)
                extract_dir = dirpath

                print(f"Unzipping: {zip_path} -> {extract_dir}")

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"✓ Extracted '{filename}' successfully.")
                except zipfile.BadZipFile:
                    print(f"✗ Skipping '{filename}' (corrupted or invalid zip).")
                except Exception as e:
                    print(f"✗ Error extracting '{filename}': {e}")

    print("\nAll zip files processed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python unzip_all.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    unzip_all_in_tree(folder_path)