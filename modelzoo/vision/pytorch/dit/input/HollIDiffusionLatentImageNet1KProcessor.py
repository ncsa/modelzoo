import tarfile
import os

from modelzoo.vision.pytorch.dit.input.DiffusionLatentImageNet1KProcessor import (
    DiffusionLatentImageNet1KProcessor,
)


def run_command(command):
    # Run the command and check that it succeeded.
    import subprocess
    result = subprocess.run(command, shell=True, check=True)
    # Check that the return code is a success code
    if result.returncode != 0:
        raise RuntimeError(f"Command {command} failed with return code {result.returncode}")


class HollIDiffusionLatentImageNet1KProcessor(DiffusionLatentImageNet1KProcessor):
    def __init__(self, params):

        self.unpack_dir = params.get("unpack_dir")

        data_package = params.get("data_package")
        if not isinstance(data_package, list):
            data_package = [data_package]

        # Check that each data_package tarball contains a single directory
        data_package_dirs = []
        for package in data_package:
            found_dirs = []
            with tarfile.open(package, "r") as tar:
                for member in tar.getmembers():
                    if member.isdir():
                        found_dirs.append(member.name)
            # Check there is only a single root directory
            root_track = {}
            for i in range(len(found_dirs)):
                dir = found_dirs[i]
                root_track[dir] = 0
                for j in range(len(found_dirs)):
                    if i != j:
                        dir2 = found_dirs[j]
                        if dir2.startswith(dir):
                            root_track[dir] += 1
            dir_sort = sorted(found_dirs, key=lambda d: root_track[d], reverse=True)
            max_root_dir = dir_sort[0]
            if root_track[max_root_dir] != len(found_dirs) - 1:
                raise RuntimeError(
                    f"Tarball {package} contains multiple root directories"
                )
            data_package_dirs.append(max_root_dir)

        # Unpack the tarballs into the unpack directory
        for package, r_dir in zip(data_package, data_package_dirs):
            run_command("tar -xf " + package + " -C " + self.unpack_dir)

        # Set the data_dir to the unpacked directories
        data_dir = [os.path.join(self.unpack_dir, d) for d in data_package_dirs]

        for dir in data_dir:
            first_few_files = []
            file_limit = 100
            num_files = 0
            # List all files recursively
            for root, dirs, files in os.walk(dir):
                for file in files:
                    full_filepath = os.path.join(root, file)
                    first_few_files.append(full_filepath)
                    num_files += 1
                    if num_files >= file_limit:
                        break
                if num_files >= file_limit:
                    break

        # Update the parameters and remove the unpack_dir and data_package
        params.update({"data_dir": data_dir})
        del params['unpack_dir']
        del params['data_package']

        # Instantiate the superclass as usual
        super().__init__(params)
