//! CCOS COLD → EventLog : démonstration de la persistance des tuiles évincées.
//!
//! `cargo run --release --example ccos_eventlog`
//!
//! Remplit un cache élastique au-delà de son budget, force la pagination puis
//! l'éviction, montre que chaque tuile évincée est journalisée sur disque
//! (format déterministe `scirust::eventlog`), puis en réhydrate une.

use scirust::ccos::ElasticKvCache;
use scirust::eventlog::EventLog;
use scirust::scenario::{build_tile, generate, Projection};

fn main() {
    let proj = Projection::new(0xC0DE);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "scirust_ccos_eventlog_demo_{}.sllg",
        std::process::id()
    ));

    // Budget = 3 tuiles HOT (3 × 128 o) ; on en insère 12.
    let budget = 3 * 128;
    let mut cache = ElasticKvCache::with_budget(budget);
    cache.attach_event_log(EventLog::create(&path).expect("création du journal"));

    println!("== CCOS EventLog — persistance COLD ==\n");
    println!("  Budget      : {budget} octets (≈ 3 tuiles HOT)");
    println!("  Journal     : {}", path.display());

    let (_q, toks) = generate(1, 12, 0.35);
    for (i, t) in toks.iter().enumerate() {
        cache.insert(build_tile(&proj, t, i as u32, false));
    }
    cache.enforce_budget();

    let (hot, warm, cold) = cache.counts();
    println!("\n  Après enforce_budget sur 12 tuiles :");
    println!("    HOT {hot} · WARM {warm} · COLD {cold}");
    println!(
        "    empreinte élastique : {} octets (≤ budget)",
        cache.live_bytes()
    );
    println!("    erreurs de journalisation : {}", cache.log_errors());

    // Relire le journal indépendamment.
    let mut log = EventLog::open(&path).expect("réouverture du journal");
    let recs = log.read_all().expect("lecture du journal");
    println!("\n  Journal sur disque : {} enregistrement(s)", recs.len());
    for r in recs.iter().take(5) {
        let mode = if r.tile.is_warm() { "WARM" } else { "HOT " };
        println!(
            "    seq {:>2} · slot {:>2} · {mode} · position {}",
            r.seq, r.slot, r.tile.position
        );
    }
    if recs.len() > 5 {
        println!("    … ({} autres)", recs.len() - 5);
    }

    // Réhydrater la plus ancienne tuile évincée (seq 0), si présente.
    if let Some(first) = recs.first().map(|r| r.seq) {
        match cache.rehydrate(first).expect("réhydratation") {
            Some(slot) => println!(
                "\n  Réhydratation de seq {first} → slot {slot} (réadmis HOT, importance remise à zéro)."
            ),
            None => println!("\n  seq {first} introuvable au journal."),
        }
    }

    let _ = std::fs::remove_file(&path);
    println!(
        "\n  Format déterministe (aucun horodatage) : deux exécutions identiques\n  \
              produisent un journal octet-pour-octet identique."
    );
}
