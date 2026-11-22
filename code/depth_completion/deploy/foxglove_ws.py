"""foxglove_ws.py - minimal Foxglove WebSocket (v1) server, no third-party deps.

JSON control messages + binary message-data frames.  Channels are advertised on
connect; a dedicated writer thread per client keeps the capture loop non-blocking
and drops frames when a client cannot keep up.
"""
import base64
import hashlib
import json
import queue
import socket
import struct
import threading
import time

import numpy as np

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x1, 0x2, 0x8, 0x9, 0xA

COMPRESSED_IMAGE_PROTO = "Ch5mb3hnbG92ZS9Db21wcmVzc2VkSW1hZ2UucHJvdG8SCGZveGdsb3ZlIiEKBFRpbWUSCwoDc2VjGAEgASgFEgwKBG5zZWMYAiABKA0iZAoPQ29tcHJlc3NlZEltYWdlEiEKCXRpbWVzdGFtcBgBIAEoCzIOLmZveGdsb3ZlLlRpbWUSEAoIZnJhbWVfaWQYAiABKAkSDAoEZGF0YRgDIAEoDBIOCgZmb3JtYXQYBCABKAliBnByb3RvMw=="

POINTCLOUD_PROTO = "Chlmb3hnbG92ZS9Qb2ludENsb3VkLnByb3RvEghmb3hnbG92ZSIhCgRUaW1lEgsKA3NlYxgBIAEoBRIMCgRuc2VjGAIgASgNIicKBFZlYzMSCQoBeBgBIAEoARIJCgF5GAIgASgBEgkKAXoYAyABKAEiOAoKUXVhdGVybmlvbhIJCgF4GAEgASgBEgkKAXkYAiABKAESCQoBehgDIAEoARIJCgF3GAQgASgBIlMKBFBvc2USIAoIcG9zaXRpb24YASABKAsyDi5mb3hnbG92ZS5WZWMzEikKC29yaWVudGF0aW9uGAIgASgLMhQuZm94Z2xvdmUuUXVhdGVybmlvbiJXChJQYWNrZWRFbGVtZW50RmllbGQSDAoEbmFtZRgBIAEoCRIOCgZvZmZzZXQYAiABKA0SIwoEdHlwZRgDIAEoDjIVLmZveGdsb3ZlLk51bWVyaWNUeXBlIrEBCgpQb2ludENsb3VkEiEKCXRpbWVzdGFtcBgBIAEoCzIOLmZveGdsb3ZlLlRpbWUSEAoIZnJhbWVfaWQYAiABKAkSHAoEcG9zZRgDIAEoCzIOLmZveGdsb3ZlLlBvc2USFAoMcG9pbnRfc3RyaWRlGAQgASgNEiwKBmZpZWxkcxgFIAMoCzIcLmZveGdsb3ZlLlBhY2tlZEVsZW1lbnRGaWVsZBIMCgRkYXRhGAYgASgMKncKC051bWVyaWNUeXBlEgsKB1VOS05PV04QABIJCgVVSU5UOBABEggKBElOVDgQAhIKCgZVSU5UMTYQAxIJCgVJTlQxNhAEEgoKBlVJTlQzMhAFEgkKBUlOVDMyEAYSCwoHRkxPQVQzMhAHEgsKB0ZMT0FUNjQQCGIGcHJvdG8z"

def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)

def _field(num, wt, payload):
    return _varint((num << 3) | wt) + payload

def _ld(num, data):
    return _field(num, 2, _varint(len(data)) + data)

def encode_compressed_image(frame_id, fmt, data, ts_ns):
    """Hand-rolled foxglove.CompressedImage protobuf."""
    sec = ts_ns // 1_000_000_000
    nsec = ts_ns % 1_000_000_000
    ts = _field(1, 0, _varint(sec)) + _field(2, 0, _varint(nsec))
    body = _field(1, 2, _varint(len(ts)) + ts) + _ld(2, frame_id.encode()) + _ld(3, data) + _ld(4, fmt.encode())
    return body

def _dbl(num, v):
    return _field(num, 1, struct.pack("<d", float(v)))

def encode_point_cloud(frame_id, xyz, ts_ns):
    """xyz: (N,3) float32 array -> foxglove.PointCloud protobuf."""
    sec, nsec = ts_ns // 1_000_000_000, ts_ns % 1_000_000_000
    ts = _field(1, 0, _varint(sec)) + _field(2, 0, _varint(nsec))
    pos = _dbl(1, 0.0) + _dbl(2, 0.0) + _dbl(3, 0.0)
    ori = _dbl(1, 0.0) + _dbl(2, 0.0) + _dbl(3, 0.0) + _dbl(4, 1.0)
    pose = _field(1, 2, _varint(len(pos)) + pos) + _field(2, 2, _varint(len(ori)) + ori)

    def fld(name, off):
        body = _ld(1, name.encode()) + _field(2, 0, _varint(off)) + _field(3, 0, _varint(7))
        return _field(5, 2, _varint(len(body)) + body)

    fields = fld("x", 0) + fld("y", 4) + fld("z", 8)
    data = np.ascontiguousarray(xyz, dtype="<f4").tobytes()
    return (_field(1, 2, _varint(len(ts)) + ts) + _ld(2, frame_id.encode())
            + _field(3, 2, _varint(len(pose)) + pose)
            + _field(4, 0, _varint(12)) + fields + _ld(6, data))

COMPRESSED_IMAGE_SCHEMA = """{
  "title": "foxglove.CompressedImage",
  "type": "object",
  "properties": {
    "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
    "frame_id": {"type": "string"},
    "data": {"type": "string", "contentEncoding": "base64"},
    "format": {"type": "string"}
  }
}"""

POINTCLOUD_SCHEMA = """{
  "title": "foxglove.PointCloud",
  "type": "object",
  "properties": {
    "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
    "frame_id": {"type": "string"},
    "pose": {"type": "object", "properties": {
      "position": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}},
      "orientation": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}}}}},
    "point_stride": {"type": "integer"},
    "fields": {"type": "array", "items": {"type": "object", "properties": {
      "name": {"type": "string"}, "offset": {"type": "integer"}, "type": {"type": "integer"}}}},
    "data": {"type": "string", "contentEncoding": "base64"}
  }
}"""


def compressed_image_json(frame_id, fmt, data, ts_ns):
    return json.dumps({"timestamp": {"sec": ts_ns // 1_000_000_000, "nsec": ts_ns % 1_000_000_000},
                       "frame_id": frame_id, "format": fmt,
                       "data": base64.b64encode(data).decode()}).encode()


def point_cloud_json(frame_id, xyz, ts_ns):
    return json.dumps({
        "timestamp": {"sec": ts_ns // 1_000_000_000, "nsec": ts_ns % 1_000_000_000},
        "frame_id": frame_id,
        "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
        "point_stride": 12,
        "fields": [{"name": "x", "offset": 0, "type": 7},
                   {"name": "y", "offset": 4, "type": 7},
                   {"name": "z", "offset": 8, "type": 7}],
        "data": base64.b64encode(np.ascontiguousarray(xyz, dtype="<f4").tobytes()).decode(),
    }).encode()


class Channel:
    def __init__(self, cid, topic, schema_name, schema, encoding="protobuf"):
        self.id = cid
        self.topic = topic
        self.schema_name = schema_name
        self.schema = schema
        self.encoding = encoding

    def to_json(self):
        # schema is a base64-encoded protobuf FileDescriptorSet (what Foxglove
        # expects for encoding=protobuf); an empty string falls back to the
        # client's built-in definition for foxglove.* types
        out = {"id": self.id, "topic": self.topic, "encoding": self.encoding,
               "schemaName": self.schema_name}
        if self.schema:
            out["schemaEncoding"] = "protobuf"
            out["schema"] = self.schema
        # no schema -> Foxglove falls back to its built-in definition for
        # well-known foxglove.* message types
        return out

class _Client:
    def __init__(self, sock):
        self.sock = sock
        self.subs = {}          # subscription id -> channel id
        self.q = queue.Queue(maxsize=8)
        self.alive = True
        self.dropped = 0
        self.sent = 0

    def send_raw(self, opcode, payload):
        n = len(payload)
        hdr = bytearray([0x80 | opcode])
        if n < 126:
            hdr.append(n)
        elif n < 65536:
            hdr.append(126)
            hdr += struct.pack(">H", n)
        else:
            hdr.append(127)
            hdr += struct.pack(">Q", n)
        self.sock.sendall(bytes(hdr) + payload)

    def send_json(self, obj):
        self.send_raw(OP_TEXT, json.dumps(obj).encode())

    def writer(self):
        while self.alive:
            try:
                op, data = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.send_raw(op, data)
                self.sent += 1
            except Exception:
                self.alive = False
                break

    def push(self, opcode, payload):
        try:
            self.q.put_nowait((opcode, payload))
        except queue.Full:
            self.dropped += 1

class FoxgloveServer:
    def __init__(self, port=8765, host="0.0.0.0", name="OrangePi PromptDA"):
        self.port = port
        self.host = host
        self.name = name
        self.channels = {}
        self.clients = []
        self.lock = threading.Lock()
        self._srv = None
        self.running = False

    def add_channel(self, topic, schema_name, schema, encoding="protobuf"):
        cid = len(self.channels) + 1
        ch = Channel(cid, topic, schema_name, schema, encoding)
        self.channels[cid] = ch
        return ch

    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(5)
        self._srv = s
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while self.running:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        try:
            conn.settimeout(5.0)
            req = b""
            while b"\r\n\r\n" not in req:
                d = conn.recv(4096)
                if not d:
                    return
                req += d
            key = ""
            for line in req.decode("latin1").split("\r\n"):
                if line.lower().startswith("sec-websocket-key:"):
                    key = line.split(":", 1)[1].strip()
            if not key:
                conn.close()
                return
            acc = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
            conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                          "Connection: Upgrade\r\nSec-WebSocket-Accept: " + acc +
                          "\r\nSec-WebSocket-Protocol: foxglove.websocket.v1\r\n\r\n").encode())
            conn.settimeout(None)
            cl = _Client(conn)
            cl.send_json({"op": "serverInfo", "name": self.name, "capabilities": [],
                          "sessionId": str(int(time.time()))})
            cl.send_json({"op": "advertise", "channels": [c.to_json() for c in self.channels.values()]})
            threading.Thread(target=cl.writer, daemon=True).start()
            with self.lock:
                self.clients.append(cl)
            self._read_loop(cl)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _read_loop(self, cl):
        buf = b""
        while cl.alive:
            try:
                d = cl.sock.recv(65536)
            except Exception:
                break
            if not d:
                break
            buf += d
            while True:
                if len(buf) < 2:
                    break
                b0, b1 = buf[0], buf[1]
                op = b0 & 0x0F
                masked = b1 & 0x80
                ln = b1 & 0x7F
                pos = 2
                if ln == 126:
                    if len(buf) < 4:
                        break
                    ln = struct.unpack(">H", buf[2:4])[0]
                    pos = 4
                elif ln == 127:
                    if len(buf) < 10:
                        break
                    ln = struct.unpack(">Q", buf[2:10])[0]
                    pos = 10
                if masked:
                    if len(buf) < pos + 4 + ln:
                        break
                    mask = buf[pos:pos + 4]
                    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(buf[pos + 4:pos + 4 + ln]))
                    pos += 4 + ln
                else:
                    if len(buf) < pos + ln:
                        break
                    payload = buf[pos:pos + ln]
                    pos += ln
                buf = buf[pos:]
                if op == OP_TEXT:
                    self._on_json(cl, payload)
                elif op == OP_CLOSE:
                    cl.alive = False
                    return
                elif op == OP_PING:
                    cl.push(OP_PONG, payload)
        cl.alive = False
        with self.lock:
            if cl in self.clients:
                self.clients.remove(cl)

    def _on_json(self, cl, payload):
        try:
            msg = json.loads(payload.decode())
        except Exception:
            return
        op = msg.get("op")
        if op == "subscribe":
            for sub in msg.get("subscriptions", []):
                cl.subs[int(sub["id"])] = int(sub["channelId"])
        elif op == "unsubscribe":
            for sid in msg.get("subscriptionIds", []):
                cl.subs.pop(int(sid), None)

    def publish(self, channel_id, payload, ts_ns=None):
        if ts_ns is None:
            ts_ns = int(time.time() * 1e9)
        with self.lock:
            clients = list(self.clients)
        for cl in clients:
            if not cl.alive:
                continue
            for sid, cid in cl.subs.items():
                if cid == channel_id:
                    cl.push(OP_BIN, struct.pack("<IQ", sid, ts_ns) + payload)

    def close(self):
        self.running = False
        try:
            self._srv.close()
        except Exception:
            pass