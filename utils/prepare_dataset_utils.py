import subprocess
import hashlib
from glob import glob as _glob


def glob(path):
    paths = _glob(path)
    paths = [p.replace('\\', '/') for p in paths]
    return paths


def get_md5(path):
    hash_md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(entries):
    for url, md5 in entries:
        filename = url.split('/')[-1]
        status = subprocess.call('wget %s -O %s' % (url, filename), shell=True)
        if status != 0:
            raise Exception('Download of %s failed' % filename)

        if md5 != get_md5(filename):
            raise Warning('Checksum of %s failed' % filename)


def concatenate(entries):
    for src, dst, md5 in entries:
        subprocess.call('cat %s > %s' % (src, dst), shell=True)
        subprocess.call('rm %s' % (src), shell=True)

        if md5 != get_md5(dst):
            raise Warning('Checksum of %s failed' % dst)


def extract(entries):
    for filename in entries:
        if filename.endswith('.tar.gz'):
            subprocess.call('tar xf %s' % (filename), shell=True)
        elif filename.endswith('.zip'):
            subprocess.call('unzip %s' % (filename), shell=True)