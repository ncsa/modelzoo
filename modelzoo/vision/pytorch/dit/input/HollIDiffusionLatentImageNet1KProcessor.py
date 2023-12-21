import tarfile
import os

from modelzoo.vision.pytorch.dit.input.DiffusionLatentImageNet1KProcessor import (
    DiffusionLatentImageNet1KProcessor,
)


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
        for package in data_package:
            with tarfile.open(package, "r") as tar:
                for member in tar.getmembers():
                    dest_file = os.path.join(self.unpack_dir, member.name)
                    if member.isdir():
                        if not os.path.exists(dest_file):
                            os.makedirs(dest_file)
                    else:
                        if not os.path.exists(os.path.dirname(dest_file)):
                            tar.extract(member, self.unpack_dir)

        # Set the data_dir to the unpacked directories
        data_dir = [os.path.join(self.unpack_dir, d) for d in data_package_dirs]

        # Update the parameters and remove the unpack_dir and data_package
        params.update({"data_dir": data_dir})
        del params['unpack_dir']
        del params['data_package']

        # Instantiate the superclass as usual
        super().__init__(params)
