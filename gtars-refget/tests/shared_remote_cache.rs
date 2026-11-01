//! Multi-PROCESS tests for remote stores sharing one cache directory.
//!
//! Several processes (e.g. the workers of one Cloud Batch task) open the same
//! remote store with the same cache directory and fetch sequences on demand.
//! Every process must read exactly the remote bytes, whatever the others are
//! fetching, publishing or invalidating at the same moment.
//!
//! As in `concurrent_store_writers.rs`, children are re-executions of this test
//! binary (`harness = false`), so the coordination under test is between real
//! processes. The parent serves the remote store over HTTP from a thread.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use gtars_refget::digest::digest_sequence;
use gtars_refget::store::{FastaImportOptions, RefgetStore};

/// Env var turning a re-exec of this binary into a reader child: the remote URL.
const CHILD_REMOTE: &str = "GTARS_TEST_CHILD_REMOTE";
/// The shared cache directory the child opens the remote store with.
const CHILD_CACHE: &str = "GTARS_TEST_CHILD_CACHE";
/// How the child reads: "whole" loads each sequence, "ranges" reads substrings.
const CHILD_MODE: &str = "GTARS_TEST_CHILD_MODE";

const CHILDREN: usize = 8;
/// Delay between child starts, per round.
const STAGGER_MS: &[u64] = &[0, 5, 15, 40, 100];
const ROUNDS: usize = 10;
const ALIAS_NAMESPACE: &str = "test";

/// The remote store's sequences: (name, length, seed). The first is large
/// enough that publishing it takes a measurable time.
const SEQUENCES: &[(&str, usize, u64)] = &[
    ("chr_big", 24_000_000, 1),
    ("chr_mid", 3_000_000, 2),
    ("chr_small", 500_000, 3),
    ("chr_tiny", 1_000, 4),
];

/// Deterministic ACGT bases, so parent and children agree without sharing data.
fn bases(length: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            b"ACGT"[(state >> 62) as usize]
        })
        .collect()
}

fn expected() -> Vec<(String, String, Vec<u8>)> {
    SEQUENCES
        .iter()
        .map(|&(name, length, seed)| {
            let data = bases(length, seed);
            let digest = digest_sequence(name, &data).metadata().sha512t24u.clone();
            (name.to_string(), digest, data)
        })
        .collect()
}

fn fail(message: String) -> ! {
    eprintln!("child {}: {}", std::process::id(), message);
    std::process::exit(1);
}

/// Runs in the CHILD: open the remote store on the shared cache and check every
/// sequence, and every alias, against the bytes the parent built it from.
fn run_as_reader(remote: &str, cache: &str, mode: &str) -> ! {
    let mut store = RefgetStore::open_remote(cache, remote)
        .unwrap_or_else(|e| fail(format!("open_remote: {e:#}")));
    store.set_quiet(true);
    store.set_cache_whole_sequences(mode == "ranges");
    store
        .load_sequence_index()
        .unwrap_or_else(|e| fail(format!("load_sequence_index: {e:#}")));
    for (name, digest, data) in expected() {
        match store.get_sequence_metadata_by_alias(ALIAS_NAMESPACE, &name) {
            Some(meta) if meta.sha512t24u == digest => {}
            other => fail(format!("alias {name}: {:?}", other.map(|m| &m.sha512t24u))),
        }
        let ranges: Vec<(usize, usize)> = match mode {
            "whole" => {
                store
                    .load_sequence(&digest)
                    .unwrap_or_else(|e| fail(format!("load_sequence {name}: {e:#}")));
                vec![(0, data.len())]
            }
            _ => {
                let n = data.len();
                vec![(0, n.min(10)), (n / 3, n / 3 + n.min(97)), (n - n.min(25), n)]
            }
        };
        for (start, end) in ranges {
            let got = store
                .get_substring(&digest, start, end)
                .unwrap_or_else(|e| fail(format!("{name}[{start}..{end}]: {e:#}")));
            if got.as_bytes() != &data[start..end] {
                fail(format!("{name}[{start}..{end}]: wrong bases ({} read)", got.len()));
            }
        }
    }
    std::process::exit(0);
}

fn main() {
    // Child role first: a re-exec must never fall through into the tests.
    if let (Ok(remote), Ok(cache), Ok(mode)) = (
        std::env::var(CHILD_REMOTE),
        std::env::var(CHILD_CACHE),
        std::env::var(CHILD_MODE),
    ) {
        run_as_reader(&remote, &cache, &mode);
    }

    let tests: &[(&str, fn())] = &[
        (
            "processes_loading_whole_sequences_share_a_cold_cache",
            processes_loading_whole_sequences_share_a_cold_cache,
        ),
        (
            "range_reads_cache_whole_sequences_once_per_cache",
            range_reads_cache_whole_sequences_once_per_cache,
        ),
        (
            "range_reads_fetch_byte_ranges_by_default",
            range_reads_fetch_byte_ranges_by_default,
        ),
    ];

    println!("\nrunning {} tests", tests.len());
    for (name, f) in tests {
        f();
        println!("test {} ... ok", name);
    }
    println!("\ntest result: ok. {} passed; 0 failed\n", tests.len());
}

/// Build the remote store: every sequence, each with an alias.
fn build_remote_store(store_dir: &Path, work: &Path) {
    let fasta = work.join("remote.fa");
    let mut text = String::new();
    for &(name, length, seed) in SEQUENCES {
        text.push_str(&format!(">{name}\n"));
        text.push_str(std::str::from_utf8(&bases(length, seed)).unwrap());
        text.push('\n');
    }
    std::fs::write(&fasta, text).unwrap();
    let mut store = RefgetStore::on_disk(store_dir).unwrap();
    store.set_quiet(true);
    store
        .add_sequence_collection_from_fasta(&fasta, FastaImportOptions::new())
        .unwrap();
    for (name, digest, _) in expected() {
        store.add_sequence_alias(ALIAS_NAMESPACE, &name, &digest).unwrap();
    }
    store.write().unwrap();
}

/// Requests served, by path, with whether each asked for a byte range.
type RequestLog = Arc<Mutex<HashMap<String, (usize, usize)>>>;

/// Serve `root` over HTTP, honoring single `Range: bytes=a-b` requests. Bodies
/// are written in chunks with a pause between them, as a real network would
/// deliver them, so concurrent children overlap in their fetches.
fn serve(root: PathBuf) -> (String, RequestLog) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log: RequestLog = Arc::default();
    let thread_log = Arc::clone(&log);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let root = root.clone();
            let log = Arc::clone(&thread_log);
            std::thread::spawn(move || respond(stream, &root, &log));
        }
    });
    (url, log)
}

fn respond(stream: TcpStream, root: &Path, log: &RequestLog) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("/").to_string();
    let mut range = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
            let mut ends = value.trim().split('-').map(|v| v.parse::<usize>());
            if let (Some(Ok(start)), Some(Ok(end))) = (ends.next(), ends.next()) {
                range = Some((start, end));
            }
        }
    }
    {
        let mut log = log.lock().unwrap();
        let entry = log.entry(path.clone()).or_default();
        entry.0 += 1;
        entry.1 += range.is_some() as usize;
    }
    let mut stream = stream;
    let body = match std::fs::read(root.join(path.trim_start_matches('/'))) {
        Ok(body) => body,
        Err(_) => {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return;
        }
    };
    let (status, body) = match range {
        Some((start, end)) => {
            let end = (end + 1).min(body.len());
            let start = start.min(end);
            let header = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\n",
                start,
                end.saturating_sub(1),
                body.len()
            );
            (header, body[start..end].to_vec())
        }
        None => ("HTTP/1.1 200 OK\r\n".to_string(), body),
    };
    let header = format!("{status}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }
    for chunk in body.chunks(256 * 1024) {
        if stream.write_all(chunk).is_err() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

fn spawn_reader(remote: &str, cache: &Path, mode: &str) -> std::process::Child {
    Command::new(std::env::current_exe().unwrap())
        .env(CHILD_REMOTE, remote)
        .env(CHILD_CACHE, cache)
        .env(CHILD_MODE, mode)
        .spawn()
        .expect("failed to spawn reader child")
}

/// Start `CHILDREN` readers on one fresh cache, `stagger` apart, and require
/// every one to succeed. Returns the cache directory.
///
/// Readers started together all miss the cache and fetch for themselves; the
/// races are met by a reader that arrives while another is mid-publish or
/// mid-open, so each round staggers the starts differently.
fn run_round(remote: &str, work: &Path, round: usize, mode: &str) -> PathBuf {
    let cache = work.join(format!("cache-{mode}-{round}"));
    let stagger = std::time::Duration::from_millis(STAGGER_MS[round % STAGGER_MS.len()]);
    let children: Vec<_> = (0..CHILDREN)
        .map(|_| {
            let child = spawn_reader(remote, &cache, mode);
            std::thread::sleep(stagger);
            child
        })
        .collect();
    let failed = children
        .into_iter()
        .map(|mut child| child.wait().unwrap())
        .filter(|status| !status.success())
        .count();
    assert_eq!(failed, 0, "{failed} of {CHILDREN} {mode} readers failed in round {round}");
    cache
}

/// Leftovers a finished open or fetch must not leave in the cache.
fn transient_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            found.extend(transient_files(&path));
        } else if name.starts_with(".rgstore.tmp.") || name.starts_with(".rgstore.lock") {
            found.push(path);
        }
    }
    found
}

/// Readers that load whole sequences, all starting on an empty cache: none may
/// read a partly written file, or lose a file another reader's open deleted.
fn processes_loading_whole_sequences_share_a_cold_cache() {
    let work = tempfile::tempdir().unwrap();
    let remote_dir = work.path().join("remote");
    build_remote_store(&remote_dir, work.path());
    let (remote, _log) = serve(remote_dir);
    for round in 0..ROUNDS {
        let cache = run_round(&remote, work.path(), round, "whole");
        assert_eq!(transient_files(&cache), Vec::<PathBuf>::new());
    }
}

/// Requests for `.seq` files, as (whole-file requests, byte-range requests).
fn seq_requests(log: &RequestLog) -> (usize, usize) {
    let log = log.lock().unwrap();
    log.iter()
        .filter(|(path, _)| path.ends_with(".seq"))
        .fold((0, 0), |(whole, ranged), (_, &(all, range))| (whole + all - range, ranged + range))
}

/// The cached `.seq` files, by path relative to the cache directory.
fn cached_seq_files(dir: &Path, root: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            cached_seq_files(&path, root, found);
        } else if path.extension().is_some_and(|e| e == "seq") {
            found.push(path.strip_prefix(root).unwrap().to_path_buf());
        }
    }
}

/// Readers that only read substrings, with whole-sequence caching on: every
/// sequence lands in the shared cache, identical to the remote's file, after
/// whole-file downloads only, and readers that miss together are the only ones
/// to download it twice.
fn range_reads_cache_whole_sequences_once_per_cache() {
    let work = tempfile::tempdir().unwrap();
    let remote_dir = work.path().join("remote");
    build_remote_store(&remote_dir, work.path());
    let (remote, log) = serve(remote_dir.clone());
    for round in 0..STAGGER_MS.len() {
        let before = seq_requests(&log);
        let cache = run_round(&remote, work.path(), round, "ranges");
        assert_eq!(transient_files(&cache), Vec::<PathBuf>::new());
        let (whole, ranged) = seq_requests(&log);
        assert_eq!(ranged, before.1, "a byte-range request was made in round {round}");
        let downloads = whole - before.0;
        assert!(
            (SEQUENCES.len()..=SEQUENCES.len() * CHILDREN).contains(&downloads),
            "{downloads} .seq downloads in round {round}"
        );
        let mut cached = Vec::new();
        cached_seq_files(&cache, &cache, &mut cached);
        assert_eq!(cached.len(), SEQUENCES.len());
        for path in cached {
            assert_eq!(
                std::fs::read(cache.join(&path)).unwrap(),
                std::fs::read(remote_dir.join(&path)).unwrap(),
                "cached {} differs from the remote",
                path.display()
            );
        }
        println!("  round {round}: {downloads} .seq downloads for {} sequences", SEQUENCES.len());
    }
}

/// Without whole-sequence caching, substring reads stay byte-range requests and
/// cache no sequence file.
fn range_reads_fetch_byte_ranges_by_default() {
    let work = tempfile::tempdir().unwrap();
    let remote_dir = work.path().join("remote");
    build_remote_store(&remote_dir, work.path());
    let (remote, log) = serve(remote_dir);
    let cache = run_round(&remote, work.path(), 0, "ranges-remote");
    let (whole, ranged) = seq_requests(&log);
    assert_eq!(whole, 0);
    assert!(ranged > 0);
    let mut cached = Vec::new();
    cached_seq_files(&cache, &cache, &mut cached);
    assert_eq!(cached, Vec::<PathBuf>::new());
}
