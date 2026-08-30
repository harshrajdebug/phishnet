"""Read a remote ZIP's central directory over HTTP range requests.

The Phishpedia benchmark zip is 19.3 GB and stored uncompressed, while this
machine has ~14 GB free. Rather than download the whole archive we parse the
central directory and range-fetch only the members we actually want.
"""
from __future__ import annotations

import io
import re
import struct
import sys
import threading
import time
from dataclasses import dataclass

import requests

FILE_ID = "12ypEMPRQ43zGRqHGut0Esq2z5en0DH4g"


@dataclass
class Entry:
    name: str
    offset: int          # offset of the local file header
    comp_size: int
    uncomp_size: int
    method: int


class RemoteZip:
    def __init__(self, file_id: str = FILE_ID):
        self.file_id = file_id
        self._local = threading.local()
        self._lock = threading.Lock()
        self.url = self._resolve(file_id)
        self.size = self._length()

    @property
    def sess(self) -> requests.Session:
        """One Session per thread; sharing one across threads is not safe."""
        s = getattr(self._local, "sess", None)
        if s is None:
            s = requests.Session()
            s.mount("https://", requests.adapters.HTTPAdapter(
                pool_connections=4, pool_maxsize=4))
            self._local.sess = s
        return s

    # -- Drive plumbing ----------------------------------------------------
    def _resolve(self, file_id: str) -> str:
        r = self.sess.get(
            "https://drive.google.com/uc",
            params={"id": file_id, "export": "download"},
            timeout=60,
        )
        r.raise_for_status()
        if "Too many users" in r.text:
            raise RuntimeError("Drive quota block still active")
        m = re.search(r'name="uuid"\s+value="([^"]+)"', r.text)
        if not m:
            raise RuntimeError("no uuid in Drive confirm page")
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm=t&uuid={m.group(1)}"
        )

    def _length(self) -> int:
        r = self.sess.get(self.url, headers={"Range": "bytes=0-1"}, timeout=60)
        r.raise_for_status()
        return int(r.headers["content-range"].split("/")[1])

    def get(self, start: int, end: int, retries: int = 4) -> bytes:
        """Inclusive byte range."""
        for attempt in range(retries):
            try:
                r = self.sess.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=180,
                )
                if r.status_code in (200, 206):
                    return r.content
                # a stale uuid shows up as an HTML error page; only one thread
                # should re-resolve, the rest pick up the refreshed url
                if r.status_code in (401, 403, 404, 500, 503):
                    stale = self.url
                    with self._lock:
                        if self.url == stale:
                            self.url = self._resolve(self.file_id)
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"range {start}-{end} failed after {retries} tries")

    # -- zip parsing -------------------------------------------------------
    def central_directory(self) -> list[Entry]:
        tail = self.get(max(0, self.size - 65_600), self.size - 1)

        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise RuntimeError("no EOCD found")
        cd_size = struct.unpack("<I", tail[i + 12:i + 16])[0]
        cd_off = struct.unpack("<I", tail[i + 16:i + 20])[0]
        count = struct.unpack("<H", tail[i + 10:i + 12])[0]

        # ZIP64: the 32-bit fields saturate, real values live in the ZIP64 EOCD
        j = tail.rfind(b"PK\x06\x07")          # ZIP64 EOCD locator
        if j >= 0:
            z64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
            rec = self.get(z64_off, z64_off + 55)
            if rec[:4] == b"PK\x06\x06":
                count = struct.unpack("<Q", rec[32:40])[0]
                cd_size = struct.unpack("<Q", rec[40:48])[0]
                cd_off = struct.unpack("<Q", rec[48:56])[0]

        print(f"central directory: {count} entries, "
              f"{cd_size/1e6:.1f} MB at offset {cd_off}", flush=True)

        buf = io.BytesIO(self.get(cd_off, cd_off + cd_size - 1))
        entries: list[Entry] = []
        while True:
            sig = buf.read(4)
            if sig != b"PK\x01\x02":
                break
            hdr = buf.read(42)
            method = struct.unpack("<H", hdr[6:8])[0]
            comp = struct.unpack("<I", hdr[16:20])[0]
            uncomp = struct.unpack("<I", hdr[20:24])[0]
            n_len, x_len, c_len = struct.unpack("<HHH", hdr[24:30])
            off = struct.unpack("<I", hdr[38:42])[0]
            name = buf.read(n_len).decode("utf-8", "replace")
            extra = buf.read(x_len)
            buf.read(c_len)

            # ZIP64 extra field patches any 0xFFFFFFFF placeholder, in order
            if 0xFFFFFFFF in (uncomp, comp, off):
                k = 0
                while k + 4 <= len(extra):
                    tag, sz = struct.unpack("<HH", extra[k:k + 4])
                    body = extra[k + 4:k + 4 + sz]
                    if tag == 0x0001:
                        p = 0
                        if uncomp == 0xFFFFFFFF and p + 8 <= len(body):
                            uncomp = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                        if comp == 0xFFFFFFFF and p + 8 <= len(body):
                            comp = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                        if off == 0xFFFFFFFF and p + 8 <= len(body):
                            off = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                        break
                    k += 4 + sz
            entries.append(Entry(name, off, comp, uncomp, method))
        return entries

    def read_member(self, e: Entry) -> bytes:
        """Fetch one member, decompressing if it is deflated."""
        if e.comp_size == 0:
            # 14% of samples ship an empty yolo_coords.txt (no logo detected).
            # Asking for bytes=N-(N-1) is an inverted range and 400s.
            return b""
        head = self.get(e.offset, e.offset + 29)
        if head[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {e.name}")
        n_len, x_len = struct.unpack("<HH", head[26:30])
        start = e.offset + 30 + n_len + x_len
        data = self.get(start, start + e.comp_size - 1)
        if e.method == 0:
            return data
        import zlib
        return zlib.decompress(data, -15)


if __name__ == "__main__":
    rz = RemoteZip()
    print(f"archive: {rz.size/1e9:.2f} GB", flush=True)
    ents = rz.central_directory()
    print(f"parsed {len(ents)} entries", flush=True)

    # what kinds of files are in here?
    from collections import Counter
    exts = Counter(n.name.rsplit(".", 1)[-1].lower() if "." in n.name.rsplit("/")[-1]
                   else "<dir/none>" for n in ents)
    print("\nextensions:", exts.most_common(20), flush=True)

    print("\nfirst 25 names:", flush=True)
    for e in ents[:25]:
        print(f"  {e.name}  ({e.uncomp_size} B, method={e.method})", flush=True)

    depth = Counter(n.name.count("/") for n in ents)
    print("\npath depths:", sorted(depth.items()), flush=True)

    import json, pathlib
    pathlib.Path("data/phishpedia").mkdir(parents=True, exist_ok=True)
    with open("data/phishpedia/central_directory.json", "w") as f:
        json.dump([e.__dict__ for e in ents], f)
    print("\nsaved -> data/phishpedia/central_directory.json", flush=True)
