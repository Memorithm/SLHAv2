//! Deterministic ElasticContext example: the full residency lifecycle.
//!
//! Demonstrates the mission's central story:
//! logical context grows → VRAM pressure approaches HIGH → ElasticContext
//! predicts exhaustion → low-value HOT tiles demote to WARM → additional
//! pressure offloads selected WARM to COLD → relevant history becomes
//! active → prefetch/restore → HOT promotion → pressure falls below LOW →
//! stable state.
//!
//! Run: `cargo run -p slhav2-vram --example elastic_context --release`

use slhav2_vram::elastic_context::{ContextObservation, ElasticContext, KvTopology};

fn main() {
    // A small model topology: 28 layers, 4 KV heads, 128-dim K, bf16, K+V.
    let topology = KvTopology {
        layers: 28,
        kv_heads: 4,
        head_dim_k: 128,
        k_bytes_per_elem: 2,
        has_v: true,
        v_bytes_per_elem: 2,
    };
    // A deliberately tight VRAM budget so the lifecycle is visible.
    // The ECA budget is the physical cache budget (tiles), while the raw
    // KV demand is computed from the topology (the two differ by the
    // compression factor — that difference is the whole point).
    let budget_bytes = 256 << 10; // 256 KiB of physical tiles

    let mut ctx = ElasticContext::new("demo", budget_bytes, topology);
    ctx.set_positional_limit(4096); // hard model limit

    println!(
        "ElasticContext demo — topology {} B/token raw, physical tile budget {} KiB, positional limit 4096",
        topology.raw_bytes_per_token(),
        budget_bytes >> 10
    );
    println!(
        "raw KV for 4096 tokens would be {:.1} MiB\n",
        (4096 * topology.raw_bytes_per_token()) as f64 / (1 << 20) as f64
    );

    println!(
        "{:>4} {:>6} {:>9} {:>9} {:>9} {:>6} {:>6} {:>6} {:>6} {:>7}",
        "step", "tokens", "raw_MiB", "resident", "vram_p", "hot", "warm", "cold", "pin", "action"
    );

    // Deterministic script: grow to 4096 tokens, then stay, then drain.
    let mut resident_hi_water = 0usize;
    for step in 1..=60u64 {
        let logical = if step <= 38 {
            step * 100 // grow 100 -> 4000
        } else {
            3800u64.saturating_sub((step - 39) * 100) // drain
        };
        let growth = if step <= 38 { 200 } else { 0 };
        let obs = ContextObservation {
            logical_tokens: logical,
            predicted_growth: growth,
            ..ContextObservation::new(
                step,
                logical,
                topology,
                budget_bytes,
                budget_bytes,
                1 << 30,
                1 << 30,
            )
        };

        // Insert a tile for every new logical token (the K-transform would
        // do this in production).
        while ctx.cache().counts().0 + ctx.cache().counts().1 + ctx.cache().counts().3
            < logical as usize
        {
            let mut tile = [0u8; 128];
            tile[112..116].copy_from_slice(&(logical as u32).to_le_bytes()); // position
            ctx.cache_mut().insert(tile);
        }

        let action = match ctx.step(&obs) {
            Ok(a) => a,
            Err(e) => {
                // The model positional limit is a hard constraint; the
                // controller reports it instead of silently truncating.
                println!("{step:>4} {logical:>6}   -- hard constraint: {e}");
                break;
            }
        };
        resident_hi_water = resident_hi_water.max(ctx.telemetry.resident_bytes as usize);
        let t = &ctx.telemetry;
        let raw_mib = t.raw_kv_bytes as f64 / (1 << 20) as f64;
        let res = t.resident_bytes as f64 / (1 << 20) as f64;
        println!(
            "{step:>4} {logical:>6} {raw_mib:>9.2} {res:>9.2} {:>9.3} {:>6} {:>6} {:>6} {:>6} {:>7}",
            t.vram_pressure,
            t.hot_tiles,
            t.warm_tiles,
            t.cold_tiles,
            t.pinned_tiles,
            action.name()
        );
    }

    let t = &ctx.telemetry;
    println!("\n--- telemetry ---");
    println!("logical tokens      : {}", t.logical_tokens);
    println!("raw KV bytes       : {}", t.raw_kv_bytes);
    println!("resident bytes     : {}", t.resident_bytes);
    println!(
        "resident high-water: {} ({:.2} MiB)",
        resident_hi_water,
        resident_hi_water as f64 / (1 << 20) as f64
    );
    println!(
        "compression ratio  : {:.2}x (raw / resident)",
        t.raw_kv_bytes as f64 / t.resident_bytes.max(1) as f64
    );
    println!(
        "hot/warm/cold/pin  : {}/{}/{}/{}",
        t.hot_tiles, t.warm_tiles, t.cold_tiles, t.pinned_tiles
    );
    println!(
        "demotions/promotions/evictions: {}/{}/{}",
        t.demotions, t.promotions, t.evictions
    );
    println!(
        "hard-constraint violations: {}",
        t.hard_constraint_violations
    );
    println!("\nElasticContext demo complete (deterministic).");
}
