//! In-progress markers for files a remote store is fetching into its cache.
//!
//! Processes sharing a remote store's cache directory tend to miss the same
//! file at the same moment (workers given consecutive records on one
//! chromosome all miss it within milliseconds). Without coordination each
//! downloads it; the atomic publish (see [`super::atomic`]) keeps that correct
//! but wastes the transfer. So before downloading, a process claims the file
//! by creating `.rgstore.fetch.<name>` beside it with `O_CREAT|O_EXCL`, which
//! exactly one claimant wins, on local disks and on NFS and Lustre alike (the
//! same primitive as [`super::lock`]). The others wait for the file to appear.
//!
//! # Liveness
//!
//! The claimant touches the marker every [`TOUCH_INTERVAL`] while bytes
//! arrive, so a fresh marker means a download that is progressing, not merely
//! a process that is alive. A waiter gives up on a marker whose mtime it has
//! seen unchanged for [`stale_after`] (default 30 s, about 30 missed touches)
//! by its own clock: it compares successive mtimes, never an mtime against its
//! clock, so clock skew between hosts sharing the cache cannot fake staleness.
//! A waiter that gives up downloads the file itself. The marker only saves
//! transfers; correctness never depends on it.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};

use super::atomic::FETCH_MARKER_PREFIX;

/// How often a claimant refreshes its marker while bytes arrive.
const TOUCH_INTERVAL: Duration = Duration::from_secs(1);

/// How often a waiter looks for the published file.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

const DEFAULT_STALE_AFTER: Duration = Duration::from_secs(30);

/// How long a waiter tolerates an unchanged marker before downloading the file
/// itself. `GTARS_FETCH_STALE_AFTER` (seconds) overrides the default.
fn stale_after() -> Duration {
    std::env::var("GTARS_FETCH_STALE_AFTER")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|secs| *secs >= 0.0)
        .map(Duration::from_secs_f64)
        .unwrap_or(DEFAULT_STALE_AFTER)
}

/// The marker for fetching `dest`, in the same directory.
pub(crate) fn marker_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    dest.with_file_name(format!("{}{}", FETCH_MARKER_PREFIX, name))
}

/// A held claim on fetching one file. Removes its marker on drop.
pub(crate) struct FetchClaim {
    path: PathBuf,
    file: File,
    touched: Instant,
}

impl FetchClaim {
    /// Claim fetching `dest`. `Ok(None)` if another process holds the claim.
    pub(crate) fn try_claim(dest: &Path) -> Result<Option<Self>> {
        let path = marker_path(dest);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)
                .with_context(|| format!("creating cache directory {}", dir.display()))?;
        }
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => Ok(Some(FetchClaim {
                path,
                file,
                touched: Instant::now(),
            })),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
            Err(e) => Err(e).with_context(|| format!("creating fetch marker {}", path.display())),
        }
    }

    /// Refresh the marker if [`TOUCH_INTERVAL`] has passed since the last time.
    /// Call as bytes arrive.
    pub(crate) fn touch(&mut self) {
        if self.touched.elapsed() >= TOUCH_INTERVAL {
            // Best-effort: a missed touch costs at worst a duplicate download.
            let _ = self.file.set_modified(SystemTime::now());
            self.touched = Instant::now();
        }
    }
}

impl Drop for FetchClaim {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// How waiting on another process's fetch of a file ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Waited {
    /// The file is in the cache.
    Published,
    /// The marker went away without the file appearing: the claimant failed.
    /// Try claiming it again.
    Released,
    /// The marker stopped changing: the claimant died or stalled. Its last
    /// mtime is returned so the caller can remove that marker, and only that
    /// one, after fetching the file itself.
    Abandoned(Option<SystemTime>),
}

/// Wait while another process fetches `dest`.
pub(crate) fn wait_for_fetch(dest: &Path) -> Waited {
    let marker = marker_path(dest);
    let stale_after = stale_after();
    let mut last_mtime: Option<SystemTime> = None;
    let mut unchanged_since = Instant::now();
    loop {
        if dest.exists() {
            return Waited::Published;
        }
        let mtime = match fs::metadata(&marker) {
            Ok(meta) => meta.modified().ok(),
            Err(_) => {
                // Gone: published (checked once more) or abandoned by its owner.
                return if dest.exists() { Waited::Published } else { Waited::Released };
            }
        };
        if mtime != last_mtime {
            last_mtime = mtime;
            unchanged_since = Instant::now();
        } else if unchanged_since.elapsed() >= stale_after {
            return Waited::Abandoned(mtime);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Remove `dest`'s marker if it still has the mtime a waiter judged stale.
/// A marker a new claimant has made since has another mtime and is kept.
pub(crate) fn remove_abandoned_marker(dest: &Path, judged: Option<SystemTime>) {
    let marker = marker_path(dest);
    if let Ok(meta) = fs::metadata(&marker) {
        if meta.modified().ok() == judged {
            let _ = fs::remove_file(&marker);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_claimant_and_marker_removed_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("sub").join("x.seq");
        let claim = FetchClaim::try_claim(&dest).unwrap().expect("first claim wins");
        assert!(FetchClaim::try_claim(&dest).unwrap().is_none());
        assert!(marker_path(&dest).exists());
        drop(claim);
        assert!(!marker_path(&dest).exists());
        assert!(FetchClaim::try_claim(&dest).unwrap().is_some());
    }

    #[test]
    fn waiter_sees_publication_and_release() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.seq");
        let claim = FetchClaim::try_claim(&dest).unwrap().unwrap();
        let publisher = {
            let dest = dest.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(250));
                fs::write(&dest, b"bytes").unwrap();
                drop(claim);
            })
        };
        assert_eq!(wait_for_fetch(&dest), Waited::Published);
        publisher.join().unwrap();

        let other = dir.path().join("y.seq");
        let claim = FetchClaim::try_claim(&other).unwrap().unwrap();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(250));
            drop(claim);
        });
        assert_eq!(wait_for_fetch(&other), Waited::Released);
        releaser.join().unwrap();
    }

    #[test]
    fn only_the_marker_judged_stale_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.seq");
        let claim = FetchClaim::try_claim(&dest).unwrap().unwrap();
        std::mem::forget(claim); // its owner died
        let judged = fs::metadata(marker_path(&dest)).unwrap().modified().ok();
        // A different mtime means a new claimant: kept.
        remove_abandoned_marker(&dest, Some(SystemTime::UNIX_EPOCH));
        assert!(marker_path(&dest).exists());
        remove_abandoned_marker(&dest, judged);
        assert!(!marker_path(&dest).exists());
    }
}
