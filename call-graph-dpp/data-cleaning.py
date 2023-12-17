import os
import re

# Input and output directory paths
input_directory = 'D:\_SELEN\_2022-2023\CS588\GitHub_Codes\code\CIA-risk\outputs\commons-fileupload-callGraphOutput-tsv'  # Replace with your input directory path
output_directory = 'D:\_SELEN\_2022-2023\CS588\GitHub_Codes\code\CIA-risk\outputs\commons-fileupload-callGraph-final'  # Replace with your output directory path

# Ensure the output directory exists, create it if necessary
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each TSV file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.tsv'):  # Process only TSV files
        # Read the TSV file
        with open(os.path.join(input_directory, filename), 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Function to remove 'Anonymous' from a line
        def remove_anonymous_from_line(line):
            return re.sub(r'\.Anonymous-[\w-]+', '', line)

        # Modify lines by removing 'Anonymous' from each line
        modified_lines = [remove_anonymous_from_line(line) for line in lines]

        # Write the modified data to a new TSV file in the output directory
        with open(os.path.join(output_directory, filename), 'w', encoding='utf-8') as modified_file:
            modified_file.writelines(modified_lines)