//! Allocation alignée + politique NUMA + épinglage de thread.
//!
//! Deux niveaux d'API :
//!
//! 1. **Toujours disponible (portable, aucune dépendance)** : [`AlignedBuffer`] —
//!    allocation heap alignée (128 o par défaut, ou alignement configurable) via
//!    `std::alloc`. Fonctionne sur toutes les cibles (x86_64, aarch64, …) sans
//!    dépendance externe. Utile pour aligner les buffers chauds sur des lignes de
//!    cache, indépendamment du NUMA.
//! 2. **Feature `numa` (Linux + `libc`)** : politique NUMA + épinglage de thread —
//!    [`NumaBuffer`] (région mmap + `mbind` sur le nœud local, page-alignée),
//!    [`migrate_to_local_node`] (best-effort, exige une adresse page-alignée),
//!    [`pin_current_thread_to_cpu`], [`pin_current_thread_local`],
//!    [`current_cpu`], [`current_node`], [`numa_available`], [`num_nodes`]. Repli
//!    gracieux hors Linux ou sans la feature : les fonctions renvoient
//!    [`NumaError::Unavailable`] et [`NumaBuffer`] n'est pas construisible.
//!
//! ## Stratégie d'intégration (first-touch)
//!
//! Sur Linux, la politique d'allocation par défaut place les pages sur le nœud du
//! thread qui les touche en premier (first-touch). Pour rapprocher l'arena KV-cache
//! d'un thread d'inférence : (1) épingler ce thread à un cœur du nœud local via
//! [`pin_current_thread_local`], puis (2) remplir l'arena — les pages atterrissent sur
//! le bon nœud sans `mbind`. C'est l'intégration sûre pour le `Vec` existant de ccos
//! (dont l'allocateur global ne garantit pas la page-alignance requise par `mbind`).
//!
//! ## Note Jetson / mémoire unifiée
//!
//! Sur une puce à mémoire unifiée (Jetson Thor AGX, Apple Silicon) le système est
//! mono-socket / mono-nœud NUMA : [`numa_available`] rend `false` et l'épinglage reste
//! utile (évite les migrations). Pour le zero-Copy CPU/GPU, voir Phase 3
//! (`zero_copy`). Ce module est purement CPU.

use core::ptr::NonNull;
use std::alloc::{self, Layout};

// ── Erreur ──────────────────────────────────────────────────────────────

/// Erreur d'allocation / politique NUMA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumaError {
    /// Feature `numa` désactivée, ou cible non-Linux.
    Unavailable,
    /// Appel système Linux échoué (errno brut).
    Os(i32),
    /// Argument invalide (CPU/nœud hors plage, longueur nulle, alignement non power-of-2…).
    InvalidArgument(&'static str),
    /// Échec d'allocation (`std::alloc::alloc_zeroed` a rendu null).
    Alloc,
}

impl core::fmt::Display for NumaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unavailable => {
                write!(
                    f,
                    "numa indisponible (feature `numa` off ou cible non-Linux)"
                )
            }
            Self::Os(errno) => write!(f, "appel système Linux échoué (errno {errno})"),
            Self::InvalidArgument(msg) => write!(f, "argument NUMA invalide : {msg}"),
            Self::Alloc => write!(f, "échec d'allocation (std::alloc a rendu null)"),
        }
    }
}

impl std::error::Error for NumaError {}

/// Récupère l'errno du dernier appel système via `std::io`.
#[cfg(all(feature = "numa", target_os = "linux"))]
fn last_os_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

// ── Alignement par défaut ──────────────────────────────────────────────

/// Alignement par défaut : 128 o (couvre la ligne L1d/L2 du Jetson Thor et la tuile
/// SLHAv2). Surchageable via [`AlignedBuffer::new`].
pub const DEFAULT_ALIGN: usize = 128;

fn check_align(align: usize) -> Result<(), NumaError> {
    if align == 0 || !align.is_power_of_two() {
        return Err(NumaError::InvalidArgument(
            "alignement doit être une puissance de 2 non nulle",
        ));
    }
    Ok(())
}

// ── AlignedBuffer (portable, toujours disponible) ───────────────────────

/// Allocation heap alignée, portable (aucune dépendance, toutes cibles).
///
/// La mémoire est allouée via l'allocateur global `std::alloc` avec un `Layout`
/// d'alignement `align` (128 o par défaut). Le buffer est initialisé à zéro
/// dès l'allocation, afin que les méthodes sûres ne puissent jamais exposer
/// des octets non initialisés.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

impl AlignedBuffer {
    /// Alloue `len` octets alignés sur `align`, initialisés à zéro.
    pub fn new(len: usize, align: usize) -> Result<Self, NumaError> {
        check_align(align)?;
        if len == 0 {
            // Layout 0 est interdit pour alloc ; on garde un pointeur dangling non alloué.
            return Ok(Self {
                ptr: NonNull::dangling(),
                layout: Layout::new::<u8>(),
                len: 0,
            });
        }
        let layout = Layout::from_size_align(len, align)
            .map_err(|_| NumaError::InvalidArgument("layout overflow"))?;
        // SAFETY: layout valide (taille > 0, align power-of-2, pas d'overflow).
        // L'initialisation à zéro est nécessaire avant d'exposer un slice Rust sûr.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(NumaError::Alloc)?;
        Ok(Self { ptr, layout, len })
    }

    /// Alloue `len` octets alignés sur 128 o (défaut).
    pub fn new_aligned128(len: usize) -> Result<Self, NumaError> {
        Self::new(len, DEFAULT_ALIGN)
    }

    /// Longueur demandée (octets).
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` si longueur nulle.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Alignement effectif de l'allocation.
    pub fn align(&self) -> usize {
        self.layout.align()
    }

    /// Pointeur brut (lecture seule).
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Pointeur brut mutable.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Slice en lecture.
    ///
    /// Le contenu est toujours initialisé : l'allocation commence remplie de zéros.
    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: ptr valide pour `len` octets, propriété exclusive via &self.
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Slice mutable.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr valide pour `len` octets, propriété exclusive via &mut self.
            unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Remplit le buffer de zéros.
    pub fn zero(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr valide pour `len` octets.
            unsafe { core::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len) };
        }
    }

    /// Vérifie que l'adresse est bien alignée sur l'alignement demandé (invariant
    /// garanti par `std::alloc`, utile en test).
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize).is_multiple_of(self.layout.align())
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr vient de `alloc::alloc_zeroed` avec ce `layout` ; libération unique.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }
}

// SAFETY: la mémoire est du contenu brut sans affinité de thread ; le
// déplacement du handle vers un autre thread est sûr (l'accès synchronisé
// relève de l'utilisateur).
unsafe impl Send for AlignedBuffer {}
// SAFETY: même invariant que Send ci-dessus — aucun état interne non
// synchronisé n'est muté à travers &AlignedBuffer.
unsafe impl Sync for AlignedBuffer {}

// ════════════════════════════════════════════════════════════════════════
//  Feature `numa` + Linux : politique NUMA + épinglage + NumaBuffer
// ════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "numa", target_os = "linux"))]
mod imp {
    //! Constantes de politique mémoire (ABI kernel uapi `linux/mempolicy.h`) :
    //!   MPOL_DEFAULT=0, MPOL_PREFERRED=1, MPOL_BIND=2, MPOL_INTERLEAVE=3, MPOL_LOCAL=4.
    //!   drapeaux mbind : MPOL_MF_STRICT=1, MPOL_MF_MOVE=2 (déplacer pages existantes).
    //!
    //! Épinglage : `sched_setaffinity` sur le thread appelant. Introspection :
    //! `sched_getcpu` + parsing sysfs (`/sys/devices/system/node/…`) pour CPU→nœud.

    use super::{last_os_errno, NumaError};

    use core::ptr::NonNull;

    const MPOL_BIND: core::ffi::c_int = 2;
    const MPOL_MF_MOVE: core::ffi::c_ulong = 2;

    /// `CPU_SETSIZE` (typ. 1024) — borne de validation des numéros de CPU.
    const MAX_CPU: usize = 1024;

    /// `true` si le système a plus d'un nœud NUMA (politique NUMA pertinente).
    pub fn numa_available() -> bool {
        num_nodes() > 1
    }

    /// Nombre de nœuds NUMA (parsing sysfs). 0 si sysfs indisponible.
    pub fn num_nodes() -> usize {
        match std::fs::read_to_string("/sys/devices/system/node/online") {
            Ok(s) => parse_int_ranges(&s).len(),
            Err(_) => 0,
        }
    }

    /// CPU du thread appelant (`sched_getcpu`), ou `None` si l'appel échoue.
    pub fn current_cpu() -> Option<usize> {
        // SAFETY: sched_getcpu est sûr (renvoie -1 sur échec).
        let cpu = unsafe { libc::sched_getcpu() };
        if cpu < 0 {
            None
        } else {
            Some(cpu as usize)
        }
    }

    /// Nœud NUMA du thread appelant (via sysfs CPU→nœud), ou `None` si indéterminé.
    pub fn current_node() -> Option<usize> {
        let cpu = current_cpu()?;
        node_of_cpu(cpu)
    }

    /// Épingle le thread appelant à un cœur unique `cpu` (masque d'affinité à 1 bit).
    pub fn pin_current_thread_to_cpu(cpu: usize) -> Result<(), NumaError> {
        if cpu >= MAX_CPU {
            return Err(NumaError::InvalidArgument(
                "cpu hors plage (>= CPU_SETSIZE)",
            ));
        }
        // SAFETY: CPU_ZERO/CPU_SET sur une `cpu_set_t` zeroed locale est sûr ; le masque
        // ne contient que le bit `cpu`. sched_setaffinity(0=appelant) est sûr.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
            let r = libc::sched_setaffinity(0, core::mem::size_of::<libc::cpu_set_t>(), &set);
            if r != 0 {
                return Err(NumaError::Os(last_os_errno()));
            }
        }
        Ok(())
    }

    /// Épingle le thread appelant à son CPU courant (first-touch : à appeler avant de
    /// remplir un buffer pour que ses pages atterrissent sur le nœud local). Renvoie
    /// le CPU sélectionné.
    pub fn pin_current_thread_local() -> Result<usize, NumaError> {
        let cpu = current_cpu().ok_or(NumaError::Unavailable)?;
        pin_current_thread_to_cpu(cpu)?;
        Ok(cpu)
    }

    /// Best-effort : déplace les pages couvrant `[ptr, ptr+len)` vers le nœud NUMA
    /// local.
    ///
    /// **Exige une adresse page-alignée et une longueur arrondie à la page** (invariant
    /// de `mbind`). Pour un `Vec` classique (aligné à 16 o par l'allocateur global),
    /// préférez la stratégie first-touch ([`super::pin_current_thread_local`]) plutôt
    /// que cette fonction. Utilisée de façon fiable par [`NumaBuffer`] (mmap →
    /// page-alignée).
    pub fn migrate_to_local_node(ptr: *mut u8, len: usize) -> Result<(), NumaError> {
        if len == 0 {
            return Ok(());
        }
        if ptr.is_null() {
            return Err(NumaError::InvalidArgument("pointeur null"));
        }
        let node = current_node().ok_or(NumaError::Unavailable)?;
        // nodemask : bit `node` (nœud unique). maxnode = 64 bits → couvre nœuds 0..63.
        if node >= 64 {
            return Err(NumaError::InvalidArgument(
                "nœud >= 64 non supporté (nodemask u64)",
            ));
        }
        let mask: u64 = 1u64 << node;
        // SAFETY: mbind sur une région mmap page-alignée valide ; best-effort
        // (MPOL_MF_MOVE sans STRICT). nodemask passé par pointeur, maxnode=64.
        let r = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                ptr as core::ffi::c_ulong,
                len as core::ffi::c_ulong,
                MPOL_BIND as core::ffi::c_ulong,
                &mask as *const u64,
                64u64,
                MPOL_MF_MOVE,
            )
        };
        if r < 0 {
            return Err(NumaError::Os(last_os_errno()));
        }
        Ok(())
    }

    /// Région mémoire page-alignée allouée via `mmap(MAP_ANONYMOUS|MAP_PRIVATE)` puis
    /// placée sur le nœud NUMA local via `mbind(MPOL_BIND, MPOL_MF_MOVE)` (best-effort).
    ///
    /// Contrairement à un `Vec` (aligné à 16 o par l'allocateur global), l'adresse est
    /// page-alignée → `mbind` réussit. La capacité est arrondie à la page ; `len` est la
    /// taille demandée. Le contenu est **non initialisé** (utiliser [`NumaBuffer::zero`]).
    pub struct NumaBuffer {
        ptr: NonNull<u8>,
        len: usize,
        cap: usize, // page-rounded
    }

    impl NumaBuffer {
        /// Alloue `len` octets sur le nœud local (capacité arrondie à la page).
        /// Best-effort NUMA : si le système est mono-nœud, `mbind` peut échouer
        /// silencieusement et la région reste simplement mmap (page-alignée, first-touch
        /// local).
        pub fn new_local(len: usize) -> Result<Self, NumaError> {
            if len == 0 {
                return Err(NumaError::InvalidArgument("len doit être > 0"));
            }
            let page = page_size();
            let cap = round_up(len, page).max(page);
            // SAFETY: mmap anonymique privé ; MAP_FAILED signalé par retour == -1.
            let raw = unsafe {
                libc::mmap(
                    core::ptr::null_mut(),
                    cap,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if raw == libc::MAP_FAILED {
                return Err(NumaError::Os(last_os_errno()));
            }
            let ptr = raw as *mut u8;
            // Best-effort NUMA : on ignore l'échec (mono-nœud, permission, etc.).
            let _ = migrate_to_local_node(ptr, cap);
            Ok(Self {
                // SAFETY: mmap a réussi (non MAP_FAILED) → ptr non-null valide pour `cap`.
                ptr: unsafe { NonNull::new_unchecked(ptr) },
                len,
                cap,
            })
        }

        /// Longueur demandée (octets).
        pub fn len(&self) -> usize {
            self.len
        }

        /// `true` si longueur nulle (toujours faux — `new_local` refuse len=0).
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Capacité allouée (arrondie à la page).
        pub fn cap(&self) -> usize {
            self.cap
        }

        /// Pointeur brut (lecture).
        pub fn as_ptr(&self) -> *const u8 {
            self.ptr.as_ptr()
        }

        /// Pointeur brut (écriture).
        pub fn as_mut_ptr(&mut self) -> *mut u8 {
            self.ptr.as_ptr()
        }

        /// Slice en lecture (contenu non initialisé tant que non écrit).
        pub fn as_slice(&self) -> &[u8] {
            // SAFETY: ptr page-aligné valide pour `len` octets (cap >= len), &self.
            unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }

        /// Slice mutable.
        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: idem, &mut self exclusif.
            unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }

        /// Remplit le buffer (taille demandée) de zéros.
        pub fn zero(&mut self) {
            // SAFETY: ptr valide pour `len` octets.
            unsafe { core::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len) };
        }

        /// Vérifie l'alignement page.
        pub fn is_page_aligned(&self) -> bool {
            (self.ptr.as_ptr() as usize).is_multiple_of(page_size())
        }
    }

    impl Drop for NumaBuffer {
        fn drop(&mut self) {
            // SAFETY: ptr vient de mmap(cap) ; munmap une fois.
            unsafe {
                libc::munmap(self.ptr.as_ptr() as *mut core::ffi::c_void, self.cap);
            }
        }
    }

    // SAFETY: le mapping mmap sous-jacent est de la mémoire brute possédée
    // exclusivement par ce handle (munmap au Drop) ; aucune affinité de thread.
    unsafe impl Send for NumaBuffer {}
    // SAFETY: &NumaBuffer n'expose que des lectures de champs immuables
    // (ptr/cap) ; la synchronisation des accès au contenu relève de l'appelant.
    unsafe impl Sync for NumaBuffer {}

    // ── Helpers internes ─────────────────────────────────────────────────

    fn page_size() -> usize {
        // SAFETY: sysconf(_SC_PAGESIZE) est sûr ; repli à 4096 si -1.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            p as usize
        } else {
            4096
        }
    }

    fn round_up(n: usize, m: usize) -> usize {
        n.div_ceil(m) * m
    }

    /// Nœud NUMA d'un CPU donné (scan sysfs `node{N}/cpulist`), ou `None`.
    fn node_of_cpu(cpu: usize) -> Option<usize> {
        let entries = std::fs::read_dir("/sys/devices/system/node").ok()?;
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("node") {
                if let Ok(node) = rest.parse::<usize>() {
                    if let Ok(cpulist) = std::fs::read_to_string(e.path().join("cpulist")) {
                        if parse_int_ranges(&cpulist).contains(&cpu) {
                            return Some(node);
                        }
                    }
                }
            }
        }
        None
    }

    /// Parse une liste d'entiers Linux ("0-3,5,7-9" → [0,1,2,3,5,7,8,9]).
    fn parse_int_ranges(s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for part in s.trim().split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    for c in a..=b {
                        out.push(c);
                    }
                }
            } else if let Ok(c) = part.parse::<usize>() {
                out.push(c);
            }
        }
        out
    }
}

#[cfg(all(feature = "numa", target_os = "linux"))]
pub use imp::{
    current_cpu, current_node, migrate_to_local_node, num_nodes, numa_available,
    pin_current_thread_local, pin_current_thread_to_cpu, NumaBuffer,
};

// ── Repli : feature off ou cible non-Linux ──────────────────────────────

#[cfg(not(all(feature = "numa", target_os = "linux")))]
mod stub {
    use super::NumaError;

    /// Toujours `false` sans la feature `numa` ou hors Linux.
    pub fn numa_available() -> bool {
        false
    }

    /// 0 sans Linux/feature.
    pub fn num_nodes() -> usize {
        0
    }

    pub fn current_cpu() -> Option<usize> {
        None
    }

    pub fn current_node() -> Option<usize> {
        None
    }

    pub fn pin_current_thread_to_cpu(_cpu: usize) -> Result<(), NumaError> {
        Err(NumaError::Unavailable)
    }

    pub fn pin_current_thread_local() -> Result<usize, NumaError> {
        Err(NumaError::Unavailable)
    }

    pub fn migrate_to_local_node(_ptr: *mut u8, _len: usize) -> Result<(), NumaError> {
        Err(NumaError::Unavailable)
    }

    /// Stand-in non construisible (aucun variant public) : occupe le nom `NumaBuffer`
    /// pour la doc et l'API quand la feature est off. Toutes les méthodes rendent
    /// `Unavailable`.
    pub enum NumaBuffer {
        __Phantom,
    }

    impl NumaBuffer {
        pub fn new_local(_len: usize) -> Result<Self, NumaError> {
            Err(NumaError::Unavailable)
        }
    }
}

#[cfg(not(all(feature = "numa", target_os = "linux")))]
pub use stub::{
    current_cpu, current_node, migrate_to_local_node, num_nodes, numa_available,
    pin_current_thread_local, pin_current_thread_to_cpu, NumaBuffer,
};
