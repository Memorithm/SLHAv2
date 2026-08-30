from pathlib import Path

p = Path("slha-c/src/elastic_ffi.rs")
s = p.read_text()
old = '''        lock_cache(&cache)\n            .write_at_dense_budget(slot, bytes, target_resident_bytes)\n            .map(|_| ())\n            .map_err(|error| cache_error("elastic dense-budget write failed", error))\n'''
new = '''        let result = lock_cache(&cache)\n            .write_at_dense_budget(slot, bytes, target_resident_bytes)\n            .map(|_| ())\n            .map_err(|error| cache_error("elastic dense-budget write failed", error));\n        result\n'''
if s.count(old) != 1:
    raise RuntimeError(f"expected one dense-budget FFI expression, got {s.count(old)}")
p.write_text(s.replace(old, new, 1))
