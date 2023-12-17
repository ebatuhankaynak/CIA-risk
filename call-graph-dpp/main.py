import os

# Specify the input and output directory paths
input_directory = 'D:\_SELEN\_2022-2023\CS588\GitHub_Codes\code\CIA-risk\outputs\commons-fileupload-callGraphOutput'  # Replace with your input directory path
output_directory = 'D:\_SELEN\_2022-2023\CS588\GitHub_Codes\code\CIA-risk\outputs\commons-fileupload-callGraphOutput-tsv'  # Replace with your output directory path

# Ensure the output directory exists, create it if necessary
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Traverse through the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):  # Process only CSV files
        # Read the original file and modify the content
        input_file_path = os.path.join(input_directory, filename)
        with open(input_file_path, 'r', newline='', encoding='utf-8') as file:
            lines = file.readlines()

        # Modify lines by replacing "),"" with ")\t"
        modified_lines = [line.replace('),', ')\t') for line in lines]

        # Create a new file with the same name but .tsv extension in the output directory
        output_filename = os.path.splitext(filename)[0] + '.tsv'
        output_file_path = os.path.join(output_directory, output_filename)
        with open(output_file_path, 'w', newline='', encoding='utf-8') as modified_file:
            modified_file.writelines(modified_lines)