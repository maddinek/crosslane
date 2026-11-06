"""Portable/default guards and fail-closed site policy; no device access."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('portable_host_profile',HERE/'host_profile.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def site(tmp_path,monkeypatch,value):
    path=tmp_path/'site.json';path.write_text(json.dumps(value))
    monkeypatch.setenv('CROSSLANE_HOST_PROFILE',str(path))
    return path


def test_standalone_has_no_production_or_wrapper_defaults(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path)
    monkeypatch.delenv('CROSSLANE_HOST_PROFILE',raising=False)
    assert h.profile()=={} and h.idle_inputs()==[] and h.setting('b65_guard')==''
    h.require_pause()


def test_private_layout_and_explicit_missing_profile_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path)
    marker=tmp_path/'engine/strata/parity-g01/window.py';marker.parent.mkdir(parents=True);marker.touch()
    monkeypatch.delenv('CROSSLANE_HOST_PROFILE',raising=False)
    with pytest.raises(FileNotFoundError):h.profile()
    monkeypatch.setenv('CROSSLANE_HOST_PROFILE',str(tmp_path/'absent'))
    with pytest.raises(FileNotFoundError):h.profile()


def test_pause_check_only_when_profile_enables_it(tmp_path,monkeypatch):
    pause=tmp_path/'paused'
    path=site(tmp_path,monkeypatch,{'production_checks':True,'pause_file':str(pause)})
    with pytest.raises(ValueError,match='pause'):h.require_pause()
    pause.touch();h.require_pause()
    path.write_text(json.dumps({'production_checks':False,'pause_file':'/absent'}));h.require_pause()


def test_wrapper_lock_order_busy_gate_and_required_guard(tmp_path,monkeypatch):
    lock=tmp_path/'gpu.lock';b65=tmp_path/'b65.lock';marker=tmp_path/'entered'
    wrapper=tmp_path/'guard.sh'
    wrapper.write_text('#!/bin/bash\nset -eu\n[[ ! -e /proc/$$/fd/8 ]]\ntouch "'+str(marker)+'"\nexec 9>"'+str(b65)+'"\nflock -n 9\n"$@"\n[[ ! -e /proc/$$/fd/8 ]]\n');wrapper.chmod(0o755)
    site(tmp_path,monkeypatch,{'b65_guard':str(wrapper),'b65_lock':str(b65),'lock':str(lock),'require_b65_guard':True})
    child=tmp_path/'child.py';child.write_text('import os,fcntl\nfrom pathlib import Path\nassert Path("/proc/self/fd/8").exists() and Path("/proc/self/fd/9").exists()\n')
    cmd=[sys.executable,str(HERE/'host_profile.py'),sys.executable,str(child)]
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    marker.unlink()
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        assert subprocess.run(cmd,capture_output=True).returncode!=0
    assert not marker.exists()
    monkeypatch.setenv('CROSSLANE_B65_GUARD','')
    assert h.setting('b65_guard')==''
    r=subprocess.run(cmd,capture_output=True,text=True)
    assert r.returncode!=0 and 'requires a B65 guard' in r.stderr
    assert not marker.exists()


def test_optional_empty_guard_executes_under_lock(tmp_path,monkeypatch):
    site(tmp_path,monkeypatch,{'lock':str(tmp_path/'gpu.lock')})
    monkeypatch.setenv('CROSSLANE_B65_GUARD','')
    cmd=[sys.executable,str(HERE/'host_profile.py'),sys.executable,'-c',
         'from pathlib import Path; assert Path("/proc/self/fd/8").exists(); assert not Path("/proc/self/fd/9").exists()']
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
