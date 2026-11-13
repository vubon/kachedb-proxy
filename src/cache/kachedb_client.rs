//! Asynchronous KacheDB TCP Client with connection pooling for exact and vector caching.

use crate::error::ProxyError;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tracing::{debug, warn};

const IO_TIMEOUT: Duration = Duration::from_secs(2);

pub struct KacheDbClient {
    host: String,
    port: u16,
    pool: Mutex<Vec<TcpStream>>,
}

#[derive(Debug, Clone)]
pub struct VectorMatch {
    pub id: String,
    pub score: f32,
    pub payload: String,
}

impl KacheDbClient {
    pub fn new(host: String, port: u16) -> Self {
        Self {
            host,
            port,
            pool: Mutex::new(Vec::new()),
        }
    }

    async fn acquire_connection(&self) -> Result<TcpStream, ProxyError> {
        let pooled = {
            let mut pool = self.pool.lock().await;
            pool.pop()
        };
        if let Some(stream) = pooled {
            return Ok(stream);
        }

        let addr = format!("{}:{}", self.host, self.port);
        let stream = tokio::time::timeout(IO_TIMEOUT, TcpStream::connect(&addr))
            .await
            .map_err(|_| ProxyError::KacheDb(format!("Connect timeout to KacheDB at {addr}")))?
            .map_err(|e| {
                ProxyError::KacheDb(format!("Failed to connect to KacheDB at {addr}: {e}"))
            })?;
        let _ = stream.set_nodelay(true);
        Ok(stream)
    }

    async fn release_connection(&self, stream: TcpStream) {
        let mut pool = self.pool.lock().await;
        if pool.len() < 16 {
            pool.push(stream);
        }
    }

    async fn execute_cmd(&self, cmd: &[u8]) -> Result<Option<String>, ProxyError> {
        for attempt in 0..2 {
            let mut stream = match self.acquire_connection().await {
                Ok(s) => s,
                Err(e) => {
                    warn!("KacheDB connect error: {e}");
                    return Ok(None);
                }
            };

            let write_res = tokio::time::timeout(IO_TIMEOUT, stream.write_all(cmd)).await;
            if let Err(e) = write_res {
                if attempt == 0 {
                    continue;
                }
                warn!("Timed out writing to KacheDB: {e}");
                return Ok(None);
            } else if let Ok(Err(e)) = write_res {
                if attempt == 0 {
                    continue;
                }
                warn!("Failed to write to KacheDB: {e}");
                return Ok(None);
            }

            let read_res = tokio::time::timeout(IO_TIMEOUT, read_resp_frame(&mut stream)).await;
            match read_res {
                Ok(Ok(Some(response))) => {
                    self.release_connection(stream).await;
                    return Ok(Some(response));
                }
                _ => {
                    if attempt == 0 {
                        continue;
                    }
                    return Ok(None);
                }
            }
        }
        Ok(None)
    }

    /// Retrieve exact key from SwissTable memory store via Redis GET command.
    pub async fn get(&self, key: &str) -> Result<Option<String>, ProxyError> {
        let mut cmd = Vec::new();
        cmd.extend_from_slice(b"*2\r\n$3\r\nGET\r\n$");
        let klen = key.len().to_string();
        cmd.extend_from_slice(klen.as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(key.as_bytes());
        cmd.extend_from_slice(b"\r\n");

        let response = match self.execute_cmd(&cmd).await? {
            Some(r) => r,
            None => return Ok(None),
        };

        parse_get_response(&response)
    }

    /// Store exact key and payload in SwissTable memory store via Redis SET command.
    pub async fn set(
        &self,
        key: &str,
        value: &str,
        ttl_seconds: Option<u64>,
    ) -> Result<bool, ProxyError> {
        let mut cmd = Vec::new();
        if let Some(ttl) = ttl_seconds {
            let ttl_str = ttl.to_string();
            cmd.extend_from_slice(b"*5\r\n$3\r\nSET\r\n$");
            let klen = key.len().to_string();
            cmd.extend_from_slice(klen.as_bytes());
            cmd.extend_from_slice(b"\r\n");
            cmd.extend_from_slice(key.as_bytes());
            cmd.extend_from_slice(b"\r\n$");
            let vlen = value.len().to_string();
            cmd.extend_from_slice(vlen.as_bytes());
            cmd.extend_from_slice(b"\r\n");
            cmd.extend_from_slice(value.as_bytes());
            cmd.extend_from_slice(b"\r\n$2\r\nEX\r\n$");
            cmd.extend_from_slice(ttl_str.len().to_string().as_bytes());
            cmd.extend_from_slice(b"\r\n");
            cmd.extend_from_slice(ttl_str.as_bytes());
            cmd.extend_from_slice(b"\r\n");
        } else {
            cmd.extend_from_slice(b"*3\r\n$3\r\nSET\r\n$");
            let klen = key.len().to_string();
            cmd.extend_from_slice(klen.as_bytes());
            cmd.extend_from_slice(b"\r\n");
            cmd.extend_from_slice(key.as_bytes());
            cmd.extend_from_slice(b"\r\n$");
            let vlen = value.len().to_string();
            cmd.extend_from_slice(vlen.as_bytes());
            cmd.extend_from_slice(b"\r\n");
            cmd.extend_from_slice(value.as_bytes());
            cmd.extend_from_slice(b"\r\n");
        }

        let response = match self.execute_cmd(&cmd).await? {
            Some(r) => r,
            None => return Ok(false),
        };

        debug!("SET response: {}", response.trim());
        Ok(response.starts_with("+OK") || response.contains("OK"))
    }

    /// Perform SIMD vector search in KacheDB index.
    pub async fn vsearch(
        &self,
        index: &str,
        query_vector: &[f32],
        top_k: usize,
        threshold: f32,
    ) -> Result<Option<VectorMatch>, ProxyError> {
        let mut raw_bytes = Vec::with_capacity(query_vector.len() * 4);
        for f in query_vector {
            raw_bytes.extend_from_slice(&f.to_ne_bytes());
        }

        let mut cmd = Vec::new();
        cmd.extend_from_slice(b"*7\r\n$7\r\nVSEARCH\r\n$");
        cmd.extend_from_slice(index.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(index.as_bytes());
        cmd.extend_from_slice(b"\r\n$");
        cmd.extend_from_slice(raw_bytes.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(&raw_bytes);
        cmd.extend_from_slice(b"\r\n$4\r\nTOPK\r\n$");
        let topk_str = top_k.to_string();
        cmd.extend_from_slice(topk_str.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(topk_str.as_bytes());
        cmd.extend_from_slice(b"\r\n$9\r\nTHRESHOLD\r\n$");
        let thresh_str = format!("{threshold:.3}");
        cmd.extend_from_slice(thresh_str.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(thresh_str.as_bytes());
        cmd.extend_from_slice(b"\r\n");

        let response = match self.execute_cmd(&cmd).await? {
            Some(r) => r,
            None => return Ok(None),
        };

        parse_vsearch_response(&response)
    }

    /// Store a vector and its associated payload in KacheDB index.
    pub async fn vadd(
        &self,
        index: &str,
        id: &str,
        vector: &[f32],
        payload: &str,
        ttl_seconds: u64,
    ) -> Result<bool, ProxyError> {
        let dim = vector.len();
        let dim_str = dim.to_string();
        let mut raw_bytes = Vec::with_capacity(dim * 4);
        for f in vector {
            raw_bytes.extend_from_slice(&f.to_ne_bytes());
        }

        let ttl_str = ttl_seconds.to_string();
        let mut cmd = Vec::new();
        cmd.extend_from_slice(b"*9\r\n$4\r\nVADD\r\n$");
        cmd.extend_from_slice(index.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(index.as_bytes());
        cmd.extend_from_slice(b"\r\n$");
        cmd.extend_from_slice(id.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(id.as_bytes());
        cmd.extend_from_slice(b"\r\n$");
        cmd.extend_from_slice(dim_str.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(dim_str.as_bytes());
        cmd.extend_from_slice(b"\r\n$");
        cmd.extend_from_slice(raw_bytes.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(&raw_bytes);
        cmd.extend_from_slice(b"\r\n$7\r\nPAYLOAD\r\n$");
        cmd.extend_from_slice(payload.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(payload.as_bytes());
        cmd.extend_from_slice(b"\r\n$2\r\nEX\r\n$");
        cmd.extend_from_slice(ttl_str.len().to_string().as_bytes());
        cmd.extend_from_slice(b"\r\n");
        cmd.extend_from_slice(ttl_str.as_bytes());
        cmd.extend_from_slice(b"\r\n");

        let response = match self.execute_cmd(&cmd).await? {
            Some(r) => r,
            None => return Ok(false),
        };

        debug!("VADD response: {}", response.trim());
        Ok(response.starts_with("+OK") || response.starts_with(":1") || response.contains("OK"))
    }
}

pub type DynKacheDbClient = Arc<KacheDbClient>;

/// Read a complete RESP frame from the TCP socket across potentially fragmented TCP packets.
async fn read_resp_frame(stream: &mut TcpStream) -> std::io::Result<Option<String>> {
    let mut acc = Vec::with_capacity(4096);
    let mut chunk = [0u8; 16384];

    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            // Peer closed connection before a complete RESP frame was received.
            // Return None so the closed/truncated socket is discarded rather than pooled.
            return Ok(None);
        }
        acc.extend_from_slice(&chunk[..n]);
        if let Some(frame_len) = check_resp_complete(&acc) {
            return Ok(Some(String::from_utf8_lossy(&acc[..frame_len]).to_string()));
        }
    }
}

/// Check if `buf` holds a complete RESP2/RESP3 frame and return its total byte length.
fn check_resp_complete(buf: &[u8]) -> Option<usize> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    let crlf_pos = buf.windows(2).position(|w| w == b"\r\n")?;
    let header_end = crlf_pos + 2;

    match first {
        b'+' | b'-' | b':' | b',' => Some(header_end),
        b'$' => {
            let len_str = std::str::from_utf8(&buf[1..crlf_pos]).ok()?;
            let len: isize = len_str.parse().ok()?;
            if len < 0 {
                return Some(header_end);
            }
            let total = header_end + (len as usize) + 2;
            if buf.len() >= total {
                Some(total)
            } else {
                None
            }
        }
        b'*' => {
            let count_str = std::str::from_utf8(&buf[1..crlf_pos]).ok()?;
            let count: isize = count_str.parse().ok()?;
            if count <= 0 {
                return Some(header_end);
            }
            let mut offset = header_end;
            for _ in 0..count {
                let elem_len = check_resp_complete(&buf[offset..])?;
                offset += elem_len;
            }
            Some(offset)
        }
        _ => Some(buf.len()),
    }
}

/// Parse the next scalar or bulk string token from a RESP buffer, skipping array headers.
fn parse_next_resp_token(mut s: &str) -> Option<(&str, &str)> {
    loop {
        s = s.strip_prefix("\r\n").unwrap_or(s);
        if s.starts_with('*') {
            let pos = s.find("\r\n")?;
            s = &s[pos + 2..];
        } else {
            break;
        }
    }

    if s.is_empty() {
        return None;
    }

    if let Some(rest_hdr) = s.strip_prefix('$') {
        let pos = rest_hdr.find("\r\n")?;
        let len_str = &rest_hdr[..pos];
        let len: isize = len_str.parse().ok()?;
        let after_len = &rest_hdr[pos + 2..];
        if len < 0 {
            return None;
        }
        let ulen = len as usize;
        if after_len.len() >= ulen && after_len.is_char_boundary(ulen) {
            let val = &after_len[..ulen];
            let rem = after_len[ulen..]
                .strip_prefix("\r\n")
                .unwrap_or(&after_len[ulen..]);
            return Some((val, rem));
        } else {
            let val = after_len.strip_suffix("\r\n").unwrap_or(after_len);
            return Some((val, ""));
        }
    } else if s.starts_with(',') || s.starts_with(':') || s.starts_with('+') {
        let pos = s.find("\r\n").unwrap_or(s.len());
        let val = &s[1..pos];
        let rem = if pos + 2 <= s.len() {
            &s[pos + 2..]
        } else {
            ""
        };
        return Some((val, rem));
    }

    None
}

/// Parse RESP bulk string response from GET command using declared byte length.
fn parse_get_response(resp: &str) -> Result<Option<String>, ProxyError> {
    if resp.starts_with("$-1") || resp.starts_with("-ERR") || resp.is_empty() {
        return Ok(None);
    }

    if resp.starts_with('$')
        && let Some((val, _)) = parse_next_resp_token(resp)
    {
        return Ok(Some(val.to_string()));
    }

    Ok(None)
}

/// Parse RESP array response from VSEARCH command using declared bulk-string byte lengths.
fn parse_vsearch_response(resp: &str) -> Result<Option<VectorMatch>, ProxyError> {
    if resp.starts_with("*-1")
        || resp.starts_with("*0")
        || resp.starts_with("-ERR")
        || resp.is_empty()
    {
        return Ok(None);
    }

    let Some((item_id, rest)) = parse_next_resp_token(resp) else {
        return Ok(None);
    };
    let Some((score_str, rest)) = parse_next_resp_token(rest) else {
        return Ok(None);
    };
    let Some((payload, _)) = parse_next_resp_token(rest) else {
        return Ok(None);
    };

    if payload.is_empty() {
        return Ok(None);
    }

    let score = score_str
        .trim_start_matches(',')
        .parse::<f32>()
        .unwrap_or(0.0);

    Ok(Some(VectorMatch {
        id: item_id.to_string(),
        score,
        payload: payload.to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    async fn spawn_mock_kachedb_server() -> (String, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    while let Ok(n) = socket.read(&mut buf).await {
                        if n == 0 {
                            break;
                        }
                        let req = String::from_utf8_lossy(&buf[..n]);
                        if req.contains("GET") {
                            let _ = socket.write_all(b"$12\r\nhello cached\r\n").await;
                        } else if req.contains("SET") {
                            let _ = socket.write_all(b"+OK\r\n").await;
                        } else if req.contains("VSEARCH") {
                            let resp = "*1\r\n*3\r\n$7\r\nmatch_1\r\n$5\r\n0.952\r\n$11\r\nv_cached_ok\r\n";
                            let _ = socket.write_all(resp.as_bytes()).await;
                        } else if req.contains("VADD") {
                            let _ = socket.write_all(b"+OK\r\n").await;
                        } else {
                            let _ = socket.write_all(b"-ERR unknown command\r\n").await;
                        }
                    }
                });
            }
        });

        ("127.0.0.1".to_string(), port)
    }

    #[tokio::test]
    async fn test_client_operations_against_mock_server() {
        let (host, port) = spawn_mock_kachedb_server().await;
        let client = KacheDbClient::new(host, port);

        // Test GET
        let val = client.get("foo").await.unwrap();
        assert_eq!(val, Some("hello cached".to_string()));

        // Test SET with TTL
        let ok_ttl = client.set("foo", "bar", Some(3600)).await.unwrap();
        assert!(ok_ttl);

        // Test SET without TTL
        let ok_no_ttl = client.set("foo", "baz", None).await.unwrap();
        assert!(ok_no_ttl);

        // Test VSEARCH
        let vec = vec![0.1; 384];
        let vmatch = client.vsearch("test_idx", &vec, 1, 0.85).await.unwrap();
        assert!(vmatch.is_some());
        let m = vmatch.unwrap();
        assert_eq!(m.id, "match_1");
        assert!((m.score - 0.952).abs() < 1e-4);
        assert_eq!(m.payload, "v_cached_ok");

        // Test VADD
        let vadd_ok = client
            .vadd("test_idx", "id_1", &vec, "payload_val", 3600)
            .await
            .unwrap();
        assert!(vadd_ok);
    }

    #[tokio::test]
    async fn test_client_connect_failure() {
        let client = KacheDbClient::new("127.0.0.1".to_string(), 1); // Port 1 won't be open
        let val = client.get("foo").await.unwrap();
        assert_eq!(val, None);

        let set_res = client.set("foo", "bar", None).await.unwrap();
        assert!(!set_res);

        let vec = vec![0.1; 384];
        let vsearch_res = client.vsearch("idx", &vec, 1, 0.8).await.unwrap();
        assert!(vsearch_res.is_none());

        let vadd_res = client.vadd("idx", "id", &vec, "p", 3600).await.unwrap();
        assert!(!vadd_res);
    }

    #[test]
    fn test_parse_vsearch_response() {
        let resp = "*1\r\n*3\r\n$6\r\nquery1\r\n$5\r\n0.925\r\n$12\r\nhello world!\r\n";
        let m = parse_vsearch_response(resp).unwrap().unwrap();
        assert_eq!(m.id, "query1");
        assert!((m.score - 0.925).abs() < 1e-3);
        assert_eq!(m.payload, "hello world!");

        // Empty payload
        let resp_empty_payload = "*1\r\n*3\r\n$6\r\nquery1\r\n$5\r\n0.925\r\n";
        assert!(
            parse_vsearch_response(resp_empty_payload)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_parse_empty_response() {
        assert!(parse_vsearch_response("*0\r\n").unwrap().is_none());
        assert!(parse_vsearch_response("*-1\r\n").unwrap().is_none());
    }

    #[test]
    fn test_parse_get_response() {
        assert!(parse_get_response("$-1\r\n").unwrap().is_none());
        assert!(
            parse_get_response("-ERR unknown key\r\n")
                .unwrap()
                .is_none()
        );
        assert!(parse_get_response("+OK\r\n").unwrap().is_none());

        let res = parse_get_response("$12\r\nHello World!\r\n").unwrap();
        assert_eq!(res.as_deref(), Some("Hello World!"));

        let res_no_trailing_crlf = parse_get_response("$5\r\nHello").unwrap();
        assert_eq!(res_no_trailing_crlf.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_multiline_and_special_char_payloads() {
        let multiline_code = "Here is the code:\r\n* bullet point\r\n$PATH = /usr/bin\r\n\r\nfn main() {\n    println!(\"ok\");\n}\r\n";
        let vsearch_resp = format!(
            "*1\r\n*3\r\n$7\r\nmatch_9\r\n,0.981\r\n${}\r\n{}\r\n",
            multiline_code.len(),
            multiline_code
        );
        let m = parse_vsearch_response(&vsearch_resp).unwrap().unwrap();
        assert_eq!(m.id, "match_9");
        assert!((m.score - 0.981).abs() < 1e-3);
        assert_eq!(m.payload, multiline_code);

        let get_resp = format!("${}\r\n{}\r\n", multiline_code.len(), multiline_code);
        let g = parse_get_response(&get_resp).unwrap().unwrap();
        assert_eq!(g, multiline_code);
    }

    #[tokio::test]
    async fn test_large_fragmented_resp_over_tcp() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let large_payload = "fn compute() -> usize { 42 }\r\n* item\r\n$VAR\r\n".repeat(2500); // > 100 KB

        let expected = large_payload.clone();
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut buf = [0u8; 1024];
                // First command: large GET response sent in small chunks
                let _ = socket.read(&mut buf).await;
                let frame = format!("${}\r\n{}\r\n", expected.len(), expected);
                for chunk in frame.as_bytes().chunks(8192) {
                    let _ = socket.write_all(chunk).await;
                    let _ = socket.flush().await;
                }
                // Second command on the SAME pooled connection: verify no leftover bytes corrupt next reply
                let _ = socket.read(&mut buf).await;
                let _ = socket.write_all(b"$5\r\nclean\r\n").await;
            }
        });

        let client = KacheDbClient::new("127.0.0.1".to_string(), port);
        let first = client.get("large_key").await.unwrap().unwrap();
        assert_eq!(first.len(), large_payload.len());
        assert_eq!(first, large_payload);

        let second = client.get("next_key").await.unwrap().unwrap();
        assert_eq!(second, "clean");
    }
}
