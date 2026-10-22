from pathlib import Path
import os.path as osp
mineru_path = "test"
path_dir = Path(mineru_path).resolve()

mineru_dir_list = [d for d in path_dir.iterdir() if d.is_dir()]
print(mineru_dir_list)
for mineru_dir in mineru_dir_list:
    print(mineru_path)
    _mineru_dir = Path(osp.join(mineru_dir.resolve(),mineru_dir.name,"hybrid_auto" ))

    file_list = [ f.name for f in _mineru_dir.iterdir() if f.is_file()]
    print(file_list)