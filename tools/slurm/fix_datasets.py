import soundfile as sf
from glob import glob
from tqdm import tqdm
import shutil
from pathlib import Path


def voxid(path):
    return '/'.join(path.split('/')[-3:])


def fix_vox1_dataset():
    files = glob(f'data/voxceleb1/*/*/*.wav')

    errors = 0

    for file in tqdm(files):
        try:
            _ = sf.read(file)
        except:
            errors += 1
            src = '/lustre/fsmisc/dataset/VoxCeleb1/dev/wav/' + voxid(file)
            if Path(src).exists():
                # print(src, file)
                shutil.copyfile(src, file)
            src = '/lustre/fsmisc/dataset/VoxCeleb1/test/wav/' + voxid(file)
            if Path(src).exists():
                # print(src, file)
                shutil.copyfile(src, file)

    print(f"Fixed {errors} files")


def fix_vox2_dataset():
    files = glob(f'data/voxceleb2/*/*/*.wav')

    errors = 0

    for file in tqdm(files):
        try:
            _ = sf.read(file)
        except:
            errors += 1
            src = '/lustre/fsmisc/dataset/VoxCeleb2/dev/aac/' + voxid(file)
            shutil.copyfile(src, file)
            # print(src, file)

    print(f"Fixed {errors} files")


def fix_dataset(dataset):
    files = glob(f'data/storage/{dataset}/*/*/*.wav')

    errors = 0

    for file in tqdm(files):
        try:
            _ = sf.read(file.replace('data/storage/', 'data/'))
        except:
            errors += 1
            # shutil.copyfile(file, file.replace('data/storage/', 'data/'))
            print(file, '->', file.replace('data/storage/', 'data/'))

    print(f"Fixed {errors} files")



# fix_dataset("simulated_rirs")
# fix_dataset("musan_split")
# fix_vox1_dataset()
fix_vox2_dataset()