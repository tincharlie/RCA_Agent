import os

def extract_python_files(root_folder, output_file):
    skip_folders = {'venv', '__pycache__', '.git'}

    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        for current_folder, dirs, files in os.walk(root_folder):
            # ✅ Modify dirs in-place to SKIP folders (not delete them!)
            dirs[:] = [d for d in dirs if d not in skip_folders]

            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(current_folder, file)

                    relative_path = os.path.relpath(full_path, root_folder)

                    try:
                        with open(full_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                    except Exception as e:
                        content = f"# Error reading file: {e}"

                    outfile.write(f"{relative_path}\n")
                    outfile.write(content)
                    outfile.write("\n\n")


# ✅ Usage
root_directory = r"C:\Users\H538532\Desktop\RCA_Agent"
output_file = "combined_output.py"

extract_python_files(root_directory, output_file)