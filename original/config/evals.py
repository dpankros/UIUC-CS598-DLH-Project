import os

from dataclasses import dataclass

from config.train import ModelPrefix

@dataclass
class Evals:
    root_dir: str
    # for now, assume all files start with "Transformer_"
    file_prefix: ModelPrefix = ModelPrefix.Transformer

    def result_filenames(self) -> list[str]:
        """
        Returns files that look like results files that are in path
        :param path: a path
        :return:
        """
        files = os.listdir(self.root_dir)
        full_file_prefix = f"{self.file_prefix.to_string()}_"
        return [
            file for file in files 
            if (
                len(file.split('_')) >= 3 
                and file.startswith(full_file_prefix)
            )
        ]

