#!/usr/sbin/anaconda

import os
import fnmatch

### 07-14-2023 ##
# Misc helper functions for scATAC-Express

# ============================================
def load_files_with_match(directory, string_match):
    matched_files = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, f'*{string_match}*'):
            file_path = os.path.join(directory, file)
            matched_files.append(file_path)
    return matched_files
