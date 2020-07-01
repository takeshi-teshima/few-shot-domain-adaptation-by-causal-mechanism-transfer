from pathlib import Path
import dill as pickle


class Pickler:
    def __init__(self, path, base_path=Path('pickle')):
        self.base_path = base_path
        self.cache_path = base_path / f'{path}.dill'
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        with self.cache_path.open('rb') as _f:
            res = pickle.load(_f)
        return res

    def save(self, content):
        with self.cache_path.open('wb') as _f:
            pickle.dump(content, _f)

    def find_or_create(self, func):
        if not self.cache_path.exists():
            self.save(func())

        res = self._load()
        return res

    def load_or_none(self):
        if self.cache_path.exists():
            return self._load()
        else:
            return None
