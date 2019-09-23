import os

CORRECT_EXTENSIONS = ['.npz', '.mid']


def get_valid_files_in_dir(path):
    def is_file_valid(f):
        return os.path.splitext(f)[1] in CORRECT_EXTENSIONS and f[0] != '_'

    all_files = os.listdir(path)
    return list(filter(is_file_valid, all_files))


def parse_file_paths(src_path, dst_path, to_numpy):
    default_extension = '.npz' if to_numpy else '.mid'
    input_paths = []
    output_paths = []
    if os.path.isfile(src_path):
        input_paths = [src_path]
        output_paths = [dst_path]
    elif os.path.isdir(src_path):
        files_in_dir = get_valid_files_in_dir(src_path)
        output_files = [os.path.splitext(
            f)[0] + default_extension for f in files_in_dir]
        input_file_paths = [os.path.join(src_path, f) for f in files_in_dir]
        output_file_paths = [os.path.join(dst_path, f) for f in output_files]
        input_paths = input_file_paths
        output_paths = output_file_paths

    return input_paths, output_paths
