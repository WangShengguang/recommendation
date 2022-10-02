import os
import shutil
import tarfile
import zipfile


def _extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Args:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = file_path
    path = path

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False
