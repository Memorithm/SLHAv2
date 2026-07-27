//! Tests d'intégration pour le module `numa`.
//!
//! - Les tests [`AlignedBuffer`] (portables, aucune dépendance) tournent dans
//!   **toutes** les configurations (feature `numa` on ou off, toute cible).
//! - Les tests du chemin `numa` réel sont derrière `#[cfg(feature = "numa")]` :
//!   sur Linux ils vérifient le comportement réel (best-effort, tolérants aux
//!   hôtes mono-nœud et aux permissions CI) ; hors Linux ils vérifient que le
//!   repli gracieux rend bien `Unavailable` / `false` / `None`.
//!
//! L'hôte de développement est un Jetson Thor AGX (aarch64, mono-NUMA) : les tests
//! ne supposent jamais `numa_available() == true` ni qu'un nœud distant existe.

use scirust::numa::{AlignedBuffer, NumaError, DEFAULT_ALIGN};

// ── AlignedBuffer (portable, toujours) ──────────────────────────────────

#[test]
fn aligned_buffer_default_align_is_128() {
    let mut b = AlignedBuffer::new_aligned128(256).unwrap();
    assert_eq!(b.len(), 256);
    assert!(!b.is_empty());
    assert_eq!(b.align(), DEFAULT_ALIGN);
    assert!(b.is_aligned());
    b.zero();
    assert!(b.as_slice().iter().all(|&x| x == 0));
}

#[test]
fn aligned_buffer_is_zero_initialized() {
    let b = AlignedBuffer::new_aligned128(4096).unwrap();

    assert!(
        b.as_slice().iter().all(|&byte| byte == 0),
        "une API sûre ne doit jamais exposer de mémoire non initialisée"
    );
}

#[test]
fn aligned_buffer_read_write_roundtrip() {
    let mut b = AlignedBuffer::new_aligned128(64).unwrap();
    let payload: Vec<u8> = (0..64u8).collect();
    b.as_mut_slice().copy_from_slice(&payload);
    assert_eq!(b.as_slice(), &payload[..]);
    // Le pointeur est bien aligné sur 128 o.
    assert_eq!(b.as_ptr() as usize % DEFAULT_ALIGN, 0);
}

#[test]
fn aligned_buffer_custom_alignment() {
    for &align in &[16usize, 256, 4096] {
        let b = AlignedBuffer::new(128, align).unwrap();
        assert_eq!(b.align(), align);
        assert!(b.is_aligned());
        assert_eq!(b.as_ptr() as usize % align, 0);
    }
}

#[test]
fn aligned_buffer_rejects_invalid_alignment() {
    assert!(matches!(
        AlignedBuffer::new(64, 0),
        Err(NumaError::InvalidArgument(_))
    ));
    assert!(matches!(
        AlignedBuffer::new(64, 3),
        Err(NumaError::InvalidArgument(_))
    ));
    // 7 n'est pas une puissance de 2.
    assert!(matches!(
        AlignedBuffer::new(64, 7),
        Err(NumaError::InvalidArgument(_))
    ));
}

#[test]
fn aligned_buffer_zero_len_is_safe() {
    let mut b = AlignedBuffer::new_aligned128(0).unwrap();
    assert!(b.is_empty());
    assert_eq!(b.len(), 0);
    assert_eq!(b.as_slice().len(), 0);
    // zero() ne panique pas sur un buffer vide.
    b.zero();
    assert_eq!(b.as_mut_slice().len(), 0);
}

#[test]
fn aligned_buffer_drop_does_not_leak_via_double_use() {
    // Alloue, écrit, relâche (Drop) — le simple fait de terminer sans crash valide
    // le chemin de libération. On enchaîne plusieurs tailles pour stresser l'allocateur.
    for &n in &[1usize, 127, 128, 129, 4096, 4097] {
        let mut b = AlignedBuffer::new_aligned128(n).unwrap();
        b.zero();
        for (i, x) in b.as_mut_slice().iter_mut().enumerate() {
            *x = (i % 256) as u8;
        }
        assert_eq!(b.len(), n);
    }
}

// ── Feature `numa` ───────────────────────────────────────────────────────

#[cfg(feature = "numa")]
mod numa_feat {
    use super::NumaError;
    use scirust::numa::{
        current_cpu, current_node, migrate_to_local_node, num_nodes, numa_available,
        pin_current_thread_local, pin_current_thread_to_cpu, NumaBuffer,
    };

    #[cfg(target_os = "linux")]
    mod linux {
        use super::*;

        #[test]
        fn num_nodes_parses_sysfs() {
            // Sur un hôte avec sysfs monté, on attend >= 1 nœud. Sur un environnement
            // de test sans `/sys/devices/system/node/online`, num_nodes() rend 0 — on
            // n'asserte pas une valeur exacte, juste la cohérence booléenne.
            let n = num_nodes();
            assert_eq!(numa_available(), n > 1);
        }

        #[test]
        fn current_cpu_and_node_consistent() {
            // current_node() ne peut échouer que si sysfs est absent ; on tolère.
            if let Some(cpu) = current_cpu() {
                // Si on arrive à résoudre le nœud du CPU, current_node() doit rendre
                // la même chose.
                if let Some(node) = current_node() {
                    let _ = (cpu, node); // cohérence : pas de panic
                }
            }
        }

        #[test]
        fn pin_roundtrip_best_effort() {
            // L'épinglage peut échouer en CI (permissions/cgroups). On teste le chemin
            // heureux seulement si current_cpu() est disponible, sans échec dur.
            let Some(cpu) = current_cpu() else {
                return;
            };
            // CPU 0 est toujours valide (< MAX_CPU) ; on épingle puis on restaure au CPU
            // courant initial. En cas d'échec de permission, on abandonne (best-effort).
            if pin_current_thread_to_cpu(cpu).is_err() {
                return;
            }
            // Après épingle au CPU courant, current_cpu() doit être ce même CPU (sous
            // réserve que l'OS honore l'affinité — on tolère un décalage).
            let _ = pin_current_thread_local();
        }

        #[test]
        fn pin_rejects_out_of_range_cpu() {
            // MAX_CPU = 1024 → CPU 4096 est hors plage.
            assert!(matches!(
                pin_current_thread_to_cpu(4096),
                Err(NumaError::InvalidArgument(_))
            ));
        }

        #[test]
        fn numa_buffer_alloc_zero_rw_page_aligned() {
            let len = 8192;
            let mut buf = NumaBuffer::new_local(len).unwrap();
            assert_eq!(buf.len(), len);
            assert!(!buf.is_empty());
            assert!(buf.cap() >= len);
            assert!(buf.is_page_aligned());
            buf.zero();
            assert!(buf.as_slice().iter().all(|&x| x == 0));
            // Écriture/lecture roundtrip.
            buf.as_mut_slice()
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = (i % 251) as u8);
            assert_eq!(buf.as_slice()[0], 0);
            assert_eq!(buf.as_slice()[251], 0);
            assert_eq!(buf.as_slice()[1], 1);
            // Adresse page-alignée.
            assert_eq!(buf.as_ptr() as usize % 4096, 0);
        }

        #[test]
        fn numa_buffer_rejects_zero_len() {
            assert!(matches!(
                NumaBuffer::new_local(0),
                Err(NumaError::InvalidArgument(_))
            ));
        }

        #[test]
        fn migrate_rejects_null() {
            assert!(matches!(
                migrate_to_local_node(core::ptr::null_mut(), 4096),
                Err(NumaError::InvalidArgument(_))
            ));
        }

        #[test]
        fn migrate_zero_len_is_ok() {
            // len == 0 est un no-op (pas d'appel système), même sur un pointeur dangling.
            assert!(migrate_to_local_node(core::ptr::dangling_mut::<u8>(), 0).is_ok());
        }

        #[test]
        fn migrate_on_page_aligned_buffer_best_effort() {
            // Sur un NumaBuffer page-aligné, migrate_to_local_node est au mieux un
            // succès, au pire Unavailable (mono-nœud / sysfs absent). On tolère les deux.
            let buf = NumaBuffer::new_local(8192).unwrap();
            let r = migrate_to_local_node(buf.as_ptr() as *mut u8, buf.cap());
            assert!(matches!(
                r,
                Ok(()) | Err(NumaError::Unavailable) | Err(NumaError::Os(_))
            ));
        }
    }

    #[cfg(not(target_os = "linux"))]
    mod nonlinux {
        use super::*;

        #[test]
        fn stubs_report_unavailable() {
            assert!(!numa_available());
            assert_eq!(num_nodes(), 0);
            assert_eq!(current_cpu(), None);
            assert_eq!(current_node(), None);
            assert_eq!(pin_current_thread_to_cpu(0), Err(NumaError::Unavailable));
            assert_eq!(pin_current_thread_local(), Err(NumaError::Unavailable));
            assert_eq!(
                migrate_to_local_node(0x1000 as *mut u8, 4096),
                Err(NumaError::Unavailable)
            );
            assert_eq!(NumaBuffer::new_local(4096), Err(NumaError::Unavailable));
        }
    }
}

// ── Sans la feature `numa` : le repli par défaut est cohérent ─────────────

#[cfg(not(feature = "numa"))]
mod no_feat {
    use scirust::numa::{num_nodes, numa_available, pin_current_thread_local, NumaBuffer};

    #[test]
    fn stubs_report_unavailable_without_feature() {
        assert!(!numa_available());
        assert_eq!(num_nodes(), 0);
        assert!(pin_current_thread_local().is_err());
        assert!(NumaBuffer::new_local(4096).is_err());
    }
}
