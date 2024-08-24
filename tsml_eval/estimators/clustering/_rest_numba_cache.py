import os


def _kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception:
            print("failed on filepath: %s" % file_path)  # noqa


def _kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "../../../../../")

    for root, dirnames, _ in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    _kill_files(root + "/" + dirname)
                except Exception:
                    print("failed on %s", root)  # noqa


if __name__ == "__main__":
    print("Resetting numba cache")  # noqa
    _kill_numba_cache()
    print("Resetting numba cache complete")  # noqa
