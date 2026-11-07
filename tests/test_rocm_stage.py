"""Pinned runtime extraction and launch validation, without vendor binaries."""
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import struct

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pinned_payload_hash_soname_and_fresh_destination(tmp_path, monkeypatch):
    stage = module('stage_rocm')
    # A tiny actual RPM/newc/XZ envelope exercises the retained parser.
    def cpio(name, data):
        fields = [0, 0o100644, 0, 0, 1, 0, len(data), 0, 0, 0, 0, len(name)+1, 0]
        value = b'070701'+b''.join(f'{n:08x}'.encode() for n in fields)+name.encode()+b'\0'
        value += b'\0'*(-len(value)%4)
        value += data
        return value+b'\0'*(-len(value)%4)
    payload = b'\x7fELFfixture'
    header = b'\x8e\xad\xe8\x01'+b'\0'*4+struct.pack('>II', 0, 0)
    rpm = b'\xed\xab\xee\xdb'+b'\0'*92+header+header+lzma.compress(
        cpio('opt/rocm/lib/libfixture.so.1.2', payload)+cpio('TRAILER!!!', b''))
    downloads = tmp_path/'downloads'
    downloads.mkdir()
    package = downloads/'fixture.rpm'
    package.write_bytes(rpm)
    tools = tmp_path/'tools'
    metadata = tools/'rocm-7.2.1'
    metadata.mkdir(parents=True)
    monkeypatch.setattr(stage, '__file__', str(tools/'stage_rocm.py'))
    packages = {'packages': [{'url': 'https://example.invalid/fixture.rpm',
                             'bytes': len(rpm), 'sha256': hashlib.sha256(rpm).hexdigest()}]}
    row = {'path': 'opt/rocm/lib/libfixture.so.1.2', 'bytes': len(payload),
           'sha256': hashlib.sha256(payload).hexdigest(),
           'dynamic': [{'tag': 'DT_SONAME', 'value': 'libfixture.so.1'}]}
    (metadata/'packages.json').write_text(json.dumps(packages))
    (metadata/'elf-manifest.json').write_text(json.dumps([row]))
    destination = tmp_path/'runtime'
    package.write_bytes(rpm[:-1]+b'X')
    with pytest.raises(ValueError, match='RPM differs'):
        stage.stage(downloads, destination)
    assert not destination.exists()
    package.write_bytes(rpm)
    stage.stage(downloads, destination)
    assert (destination/'libfixture.so.1').read_bytes() == payload
    assert (destination/'libfixture.so.1').readlink() == Path('libfixture.so.1.2')
    with pytest.raises(FileExistsError):
        stage.stage(downloads, destination)
    bench = module('bench')
    monkeypatch.setattr(bench, 'ROOT', tmp_path)
    assert bench.rocm_runtime(destination) == destination
    (destination/'libfixture.so.1').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='library differs'):
        bench.rocm_runtime(destination)
