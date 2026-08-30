from pathlib import Path

p = Path("slha-c/src/elastic_ffi.rs")
s = p.read_text()

replacements = [
    (
        '''        lock_cache(&cache)\n            .write_at(slot, bytes)\n            .map_err(|error| cache_error("elastic fixed-slot write failed", error))\n''',
        '''        let result = lock_cache(&cache)\n            .write_at(slot, bytes)\n            .map_err(|error| cache_error("elastic fixed-slot write failed", error));\n        result\n''',
        "fixed-slot write tail guard",
    ),
    (
        '''        lock_cache(&cache)\n            .demote_to(target_resident_bytes)\n            .map(|_| ())\n            .map_err(|error| cache_error("elastic demotion failed", error))\n''',
        '''        let result = lock_cache(&cache)\n            .demote_to(target_resident_bytes)\n            .map(|_| ())\n            .map_err(|error| cache_error("elastic demotion failed", error));\n        result\n''',
        "demote tail guard",
    ),
    (
        '''        lock_cache(&cache)\n            .offload_to(target_resident_bytes)\n            .map(|_| ())\n            .map_err(|error| cache_error("elastic offload failed", error))\n''',
        '''        let result = lock_cache(&cache)\n            .offload_to(target_resident_bytes)\n            .map(|_| ())\n            .map_err(|error| cache_error("elastic offload failed", error));\n        result\n''',
        "offload tail guard",
    ),
    (
        '''        lock_cache(&cache).restore_slot(slot).map_err(|error| {\n            ffi_error(\n                SLHA_ERR_NOT_RESIDENT,\n                format!("elastic restore failed for slot {slot}: {error}"),\n            )\n        })\n''',
        '''        let result = lock_cache(&cache).restore_slot(slot).map_err(|error| {\n            ffi_error(\n                SLHA_ERR_NOT_RESIDENT,\n                format!("elastic restore failed for slot {slot}: {error}"),\n            )\n        });\n        result\n''',
        "restore tail guard",
    ),
    (
        '''        lock_cache(&cache).promote_slot(slot).map_err(|error| {\n            ffi_error(\n                SLHA_ERR_NOT_RESIDENT,\n                format!("elastic promotion failed for slot {slot}: {error}"),\n            )\n        })\n''',
        '''        let result = lock_cache(&cache).promote_slot(slot).map_err(|error| {\n            ffi_error(\n                SLHA_ERR_NOT_RESIDENT,\n                format!("elastic promotion failed for slot {slot}: {error}"),\n            )\n        });\n        result\n''',
        "promote tail guard",
    ),
    (
        '''    match lock_cache(&cache).tier(slot) {\n        Some(PhysicalTier::Hot) => SLHA_ELASTIC_TIER_HOT,\n        Some(PhysicalTier::Warm) => SLHA_ELASTIC_TIER_WARM,\n        Some(PhysicalTier::Cold) => SLHA_ELASTIC_TIER_COLD,\n        Some(PhysicalTier::Pinned) => SLHA_ELASTIC_TIER_PINNED,\n        None => SLHA_ELASTIC_TIER_ABSENT,\n    }\n''',
        '''    let tier = lock_cache(&cache).tier(slot);\n    match tier {\n        Some(PhysicalTier::Hot) => SLHA_ELASTIC_TIER_HOT,\n        Some(PhysicalTier::Warm) => SLHA_ELASTIC_TIER_WARM,\n        Some(PhysicalTier::Cold) => SLHA_ELASTIC_TIER_COLD,\n        Some(PhysicalTier::Pinned) => SLHA_ELASTIC_TIER_PINNED,\n        None => SLHA_ELASTIC_TIER_ABSENT,\n    }\n''',
        "tier tail guard",
    ),
]

for old, new, label in replacements:
    if old not in s:
        raise RuntimeError(f"missing replacement anchor: {label}")
    s = s.replace(old, new, 1)

p.write_text(s)
