import shutil
from pathlib import Path

class DatasetZipper:
    @staticmethod
    def zip_dataset(source_dir: Path, output_file: Path) -> Path:
        """Zip the dataset folder for download."""
        # Ensure the parent directory of output_file exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # shutil.make_archive adds the extension automatically if not present
        archive_base = str(output_file).replace(".zip", "")
        shutil.make_archive(archive_base, 'zip', source_dir)
        
        return output_file
