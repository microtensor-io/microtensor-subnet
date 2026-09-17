from __future__ import annotations

import base64
import contextlib
import ctypes
import hashlib
import hmac
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time

F_INDEX = "index"
F_NAME = "name"
F_UUID = "uuid"
F_SERIAL = "serial"
F_MEMORY_TOTAL = "memory_total_mb"
F_MEMORY_USED = "memory_used_mb"
F_CAPABILITY = "compute_capability"
F_POWER_LIMIT = "power_limit_w"
F_POWER_DEFAULT = "power_default_w"
F_POWER_MIN = "power_min_w"
F_POWER_MAX = "power_max_w"
F_CLOCK_GRAPHICS = "clock_graphics_mhz"
F_CLOCK_SM = "clock_sm_mhz"
F_CLOCK_MEMORY = "clock_memory_mhz"
F_PCIE_WIDTH = "pcie_width"
F_PCIE_GENERATION = "pcie_generation"
F_UTILIZATION = "utilization"
F_MEMORY_UTILIZATION = "memory_utilization"
F_MIG_MODE = "mig_mode"
F_VIRTUALIZATION = "virtualization"
F_HOST_VGPU = "host_vgpu_mode"
F_ECC = "ecc_mode"
F_PCI_BUS = "pci_bus_id"
F_MINOR = "minor"
F_PROCESSES = "processes"
F_ERRORS = "errors"
FIELDS = (
    F_INDEX,
    F_NAME,
    F_UUID,
    F_SERIAL,
    F_MEMORY_TOTAL,
    F_MEMORY_USED,
    F_CAPABILITY,
    F_POWER_LIMIT,
    F_POWER_DEFAULT,
    F_POWER_MIN,
    F_POWER_MAX,
    F_CLOCK_GRAPHICS,
    F_CLOCK_SM,
    F_CLOCK_MEMORY,
    F_PCIE_WIDTH,
    F_PCIE_GENERATION,
    F_UTILIZATION,
    F_MEMORY_UTILIZATION,
    F_MIG_MODE,
    F_VIRTUALIZATION,
    F_HOST_VGPU,
    F_ECC,
    F_PCI_BUS,
    F_MINOR,
    F_PROCESSES,
    F_ERRORS,
)

HOST_ROOT = "/proc/1/root"
PROBE_TIMEOUT = 30
NVML_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/lib/libnvidia-ml.so.1",
)
NVML_NAMES = {
    0: "SUCCESS",
    1: "UNINITIALIZED",
    2: "INVALID_ARGUMENT",
    3: "NOT_SUPPORTED",
    4: "NO_PERMISSION",
    5: "ALREADY_INITIALIZED",
    6: "NOT_FOUND",
    7: "INSUFFICIENT_SIZE",
    8: "INSUFFICIENT_POWER",
    9: "DRIVER_NOT_LOADED",
    10: "TIMEOUT",
    11: "IRQ_ISSUE",
    12: "LIBRARY_NOT_FOUND",
    13: "FUNCTION_NOT_FOUND",
    14: "CORRUPTED_INFOROM",
    15: "GPU_IS_LOST",
    16: "RESET_REQUIRED",
    17: "OPERATING_SYSTEM",
    18: "LIB_RM_VERSION_MISMATCH",
    19: "IN_USE",
    20: "MEMORY",
    21: "NO_DATA",
    22: "VGPU_ECC_NOT_SUPPORTED",
    23: "INSUFFICIENT_RESOURCES",
    24: "FREQ_NOT_SUPPORTED",
    25: "ARGUMENT_VERSION_MISMATCH",
    26: "DEPRECATED",
    27: "NOT_READY",
    28: "GPU_NOT_FOUND",
    29: "INVALID_STATE",
    999: "UNKNOWN",
}
INSUFFICIENT_SIZE = 7
FUNCTION_NOT_FOUND = 13
CLOCK_GRAPHICS = 0
CLOCK_SM = 1
CLOCK_MEM = 2
VIRTUALIZATION_NAMES = {0: "NONE", 1: "PASSTHROUGH", 2: "VGPU", 3: "HOST_VGPU", 4: "HOST_VSGA"}
CONTAINER_ID = re.compile(r"([0-9a-f]{64})")
ROTATIONAL_UNKNOWN = None


def xtime(value):
    value <<= 1
    return (value ^ 0x11B) & 0xFF if value & 0x100 else value


def build_sbox():
    exp = [0] * 256
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x ^= xtime(x)
    sbox = [0] * 256
    for value in range(256):
        inverse = exp[(255 - log[value]) % 255] if value else 0
        s = inverse
        v = inverse
        for _ in range(4):
            v = ((v << 1) | (v >> 7)) & 0xFF
            s ^= v
        sbox[value] = s ^ 0x63
    return sbox


SBOX = build_sbox()
RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)
SHIFT = [((i // 4 + i % 4) % 4) * 4 + i % 4 for i in range(16)]


def expand_key(key):
    words = [list(key[i : i + 4]) for i in range(0, 16, 4)]
    for i in range(4, 44):
        temp = list(words[i - 1])
        if i % 4 == 0:
            temp = temp[1:] + temp[:1]
            temp = [SBOX[b] for b in temp]
            temp[0] ^= RCON[i // 4 - 1]
        words.append([words[i - 4][j] ^ temp[j] for j in range(4)])
    return [sum(words[4 * r : 4 * r + 4], []) for r in range(11)]


def encrypt_block(round_keys, block):
    state = [block[i] ^ round_keys[0][i] for i in range(16)]
    for rnd in range(1, 11):
        state = [SBOX[b] for b in state]
        state = [state[SHIFT[i]] for i in range(16)]
        if rnd != 10:
            mixed = []
            for c in range(0, 16, 4):
                a0, a1, a2, a3 = state[c : c + 4]
                mixed.extend(
                    (
                        xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3,
                        a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3,
                        a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3,
                        xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3),
                    )
                )
            state = mixed
        state = [state[i] ^ round_keys[rnd][i] for i in range(16)]
    return bytes(state)


def fernet_encrypt(key_text, data):
    key = base64.urlsafe_b64decode(key_text)
    signing_key, encryption_key = key[:16], key[16:32]
    iv = os.urandom(16)
    round_keys = expand_key(encryption_key)
    pad = 16 - len(data) % 16
    padded = data + bytes([pad]) * pad
    out = bytearray()
    previous = iv
    for offset in range(0, len(padded), 16):
        block = bytes(padded[offset + j] ^ previous[j] for j in range(16))
        previous = encrypt_block(round_keys, block)
        out += previous
    basic = b"\x80" + struct.pack(">Q", int(time.time())) + iv + bytes(out)
    tag = hmac.new(signing_key, basic, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(basic + tag).decode("ascii")


def derive_key():
    return base64.urlsafe_b64encode(
        hashlib.sha256("".join(FIELDS).encode("utf-8")).digest()
    ).decode("ascii")


def read(path, limit=65536):
    try:
        with open(path, "rb") as handle:
            return handle.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def run(argv, timeout=20):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "", str(exc)
    return done.returncode, done.stdout or "", done.stderr or ""


def file_hashes(path):
    if not path:
        return {"path": "", "error": "not found"}
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        return {"path": path, "error": str(exc)}
    return {
        "path": path,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "md5": hashlib.md5(raw).hexdigest(),
    }


def find_nvml():
    for candidate in NVML_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    found = []
    for base, _dirs, files in os.walk("/usr"):
        if "libnvidia-ml.so.1" in files:
            found.append(os.path.join(base, "libnvidia-ml.so.1"))
        if len(found) > 8:
            break
    found.sort(key=lambda path: (("x86_64" not in path) and ("lib64" not in path), len(path)))
    return found[0] if found else ""


class NvmlFailure(Exception):
    def __init__(self, code, function):
        Exception.__init__(
            self, "{} from {}".format(NVML_NAMES.get(code, f"CODE_{code}"), function)
        )
        self.code = code
        self.name = NVML_NAMES.get(code, f"CODE_{code}")
        self.function = function


class Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class ProcessV2(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class ProcessV1(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]


class Nvml:
    def __init__(self, path):
        with open(path, "rb") as handle:
            raw = handle.read()
        self.path = path
        self.hashes = {
            "path": path,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "md5": hashlib.md5(raw).hexdigest(),
        }
        fd, temp = tempfile.mkstemp(prefix=".mt-nvml-", suffix=".so")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
            self.lib = ctypes.CDLL(temp)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp)
        self.errors = []

    def function(self, name):
        try:
            return getattr(self.lib, name)
        except AttributeError:
            return None

    def call(self, name, *args):
        fn = self.function(name)
        if fn is None:
            return FUNCTION_NOT_FOUND
        fn.restype = ctypes.c_int
        return int(fn(*args))

    def check(self, name, *args):
        code = self.call(name, *args)
        if code != 0:
            raise NvmlFailure(code, name)

    def note(self, failure):
        if failure.name not in self.errors:
            self.errors.append(failure.name)

    def string(self, name, handle, size=96):
        buffer = ctypes.create_string_buffer(size)
        self.check(name, handle, buffer, ctypes.c_uint(size))
        return buffer.value.decode("utf-8", "replace")

    def uint(self, name, handle, *extra):
        value = ctypes.c_uint(0)
        self.check(name, handle, *(extra + (ctypes.byref(value),)))
        return int(value.value)

    def error_string(self, code):
        fn = self.function("nvmlErrorString")
        if fn is None:
            return ""
        fn.restype = ctypes.c_char_p
        raw = fn(ctypes.c_int(code))
        return raw.decode("utf-8", "replace") if raw else ""


def attempt(card, field, action):
    try:
        card[field] = action()
    except NvmlFailure as exc:
        card[field] = None
        card[F_ERRORS][field] = exc.name
    except Exception as exc:
        card[field] = None
        card[F_ERRORS][field] = f"PY_{type(exc).__name__}"


def container_of(pid):
    text = read(f"/proc/{pid}/cgroup", 4096)
    match = CONTAINER_ID.search(text)
    return match.group(1) if match else ""


def describe_process(pid, kind, memory_mb, instance):
    status = read(f"/proc/{pid}/status", 2048)
    uid = -1
    for line in status.splitlines():
        if line.startswith("Uid:"):
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                uid = int(parts[1])
            break
    return {
        "pid": pid,
        "kind": kind,
        "memory_mb": memory_mb,
        "comm": read(f"/proc/{pid}/comm", 64).strip(),
        "cmdline": read(f"/proc/{pid}/cmdline", 4096).replace("\0", " ").strip()[:200],
        "uid": uid,
        "container": container_of(pid),
        "gpu_instance": instance[0],
        "compute_instance": instance[1],
        "alive": os.path.exists(f"/proc/{pid}"),
    }


def running_processes(nvml, handle, family, kind):
    versions = (
        (f"{family}_v3", ProcessV2),
        (f"{family}_v2", ProcessV2),
        (family, ProcessV1),
    )
    last = None
    for name, struct_type in versions:
        if nvml.function(name) is None:
            continue
        count = ctypes.c_uint(0)
        code = nvml.call(name, handle, ctypes.byref(count), None)
        if code == 0:
            return [], ""
        if code != INSUFFICIENT_SIZE:
            last = NvmlFailure(code, name)
            continue
        size = int(count.value) * 2 + 5
        entries = (struct_type * size)()
        count = ctypes.c_uint(size)
        code = nvml.call(name, handle, ctypes.byref(count), entries)
        if code != 0:
            last = NvmlFailure(code, name)
            continue
        found = []
        for entry in entries[: int(count.value)]:
            memory = int(entry.usedGpuMemory)
            memory_mb = -1 if memory in (0xFFFFFFFFFFFFFFFF,) else memory // (1024 * 1024)
            instance = (
                int(getattr(entry, "gpuInstanceId", 0xFFFFFFFF)),
                int(getattr(entry, "computeInstanceId", 0xFFFFFFFF)),
            )
            found.append(describe_process(int(entry.pid), kind, memory_mb, instance))
        return found, ""
    if last is None:
        return [], "FUNCTION_NOT_FOUND"
    nvml.note(last)
    return [], last.name


def scrape_gpu(nvml, index):
    card = {F_INDEX: index, F_ERRORS: {}}
    handle = ctypes.c_void_p()
    try:
        nvml.check("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(index), ctypes.byref(handle))
    except NvmlFailure as exc:
        nvml.note(exc)
        card[F_ERRORS]["handle"] = exc.name
        return card
    attempt(card, F_NAME, lambda: nvml.string("nvmlDeviceGetName", handle, 96))
    attempt(card, F_UUID, lambda: nvml.string("nvmlDeviceGetUUID", handle, 96))
    attempt(card, F_SERIAL, lambda: nvml.string("nvmlDeviceGetSerial", handle, 64))

    def memory():
        info = Memory()
        nvml.check("nvmlDeviceGetMemoryInfo", handle, ctypes.byref(info))
        return (int(info.total) // (1024 * 1024), int(info.used) // (1024 * 1024))

    try:
        total, used = memory()
        card[F_MEMORY_TOTAL] = total
        card[F_MEMORY_USED] = used
    except NvmlFailure as exc:
        nvml.note(exc)
        card[F_MEMORY_TOTAL] = None
        card[F_MEMORY_USED] = None
        card[F_ERRORS]["memory"] = exc.name

    def capability():
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        nvml.check(
            "nvmlDeviceGetCudaComputeCapability", handle, ctypes.byref(major), ctypes.byref(minor)
        )
        return f"{major.value}.{minor.value}"

    attempt(card, F_CAPABILITY, capability)
    attempt(
        card, F_POWER_LIMIT, lambda: nvml.uint("nvmlDeviceGetPowerManagementLimit", handle) / 1000.0
    )
    attempt(
        card,
        F_POWER_DEFAULT,
        lambda: nvml.uint("nvmlDeviceGetPowerManagementDefaultLimit", handle) / 1000.0,
    )

    def constraints():
        low = ctypes.c_uint(0)
        high = ctypes.c_uint(0)
        nvml.check(
            "nvmlDeviceGetPowerManagementLimitConstraints",
            handle,
            ctypes.byref(low),
            ctypes.byref(high),
        )
        return (low.value / 1000.0, high.value / 1000.0)

    try:
        low, high = constraints()
        card[F_POWER_MIN] = low
        card[F_POWER_MAX] = high
    except NvmlFailure as exc:
        card[F_POWER_MIN] = None
        card[F_POWER_MAX] = None
        card[F_ERRORS]["power_constraints"] = exc.name
    attempt(
        card,
        F_CLOCK_GRAPHICS,
        lambda: nvml.uint("nvmlDeviceGetClockInfo", handle, ctypes.c_uint(CLOCK_GRAPHICS)),
    )
    attempt(
        card,
        F_CLOCK_SM,
        lambda: nvml.uint("nvmlDeviceGetClockInfo", handle, ctypes.c_uint(CLOCK_SM)),
    )
    attempt(
        card,
        F_CLOCK_MEMORY,
        lambda: nvml.uint("nvmlDeviceGetClockInfo", handle, ctypes.c_uint(CLOCK_MEM)),
    )
    attempt(card, F_PCIE_WIDTH, lambda: nvml.uint("nvmlDeviceGetCurrPcieLinkWidth", handle))
    attempt(
        card, F_PCIE_GENERATION, lambda: nvml.uint("nvmlDeviceGetCurrPcieLinkGeneration", handle)
    )

    def utilization():
        rates = Utilization()
        nvml.check("nvmlDeviceGetUtilizationRates", handle, ctypes.byref(rates))
        return (int(rates.gpu), int(rates.memory))

    try:
        gpu_rate, memory_rate = utilization()
        card[F_UTILIZATION] = gpu_rate
        card[F_MEMORY_UTILIZATION] = memory_rate
    except NvmlFailure as exc:
        card[F_UTILIZATION] = None
        card[F_MEMORY_UTILIZATION] = None
        card[F_ERRORS]["utilization"] = exc.name

    def mig():
        current = ctypes.c_uint(0)
        pending = ctypes.c_uint(0)
        nvml.check("nvmlDeviceGetMigMode", handle, ctypes.byref(current), ctypes.byref(pending))
        return {"current": int(current.value), "pending": int(pending.value)}

    attempt(card, F_MIG_MODE, mig)

    def virtualization():
        mode = ctypes.c_uint(0)
        nvml.check("nvmlDeviceGetVirtualizationMode", handle, ctypes.byref(mode))
        return VIRTUALIZATION_NAMES.get(int(mode.value), f"MODE_{mode.value}")

    attempt(card, F_VIRTUALIZATION, virtualization)
    attempt(card, F_HOST_VGPU, lambda: nvml.uint("nvmlDeviceGetHostVgpuMode", handle))

    def ecc():
        current = ctypes.c_uint(0)
        pending = ctypes.c_uint(0)
        nvml.check("nvmlDeviceGetEccMode", handle, ctypes.byref(current), ctypes.byref(pending))
        return int(current.value)

    attempt(card, F_ECC, ecc)

    def pci():
        buffer = ctypes.create_string_buffer(256)
        for name in ("nvmlDeviceGetPciInfo_v3", "nvmlDeviceGetPciInfo_v2", "nvmlDeviceGetPciInfo"):
            if nvml.function(name) is None:
                continue
            nvml.check(name, handle, buffer)
            legacy = buffer.raw[:16].split(b"\0", 1)[0].decode("ascii", "replace")
            if name.endswith("_v3"):
                full = buffer.raw[36:68].split(b"\0", 1)[0].decode("ascii", "replace")
                return full or legacy
            return legacy
        raise NvmlFailure(FUNCTION_NOT_FOUND, "nvmlDeviceGetPciInfo")

    attempt(card, F_PCI_BUS, pci)
    attempt(card, F_MINOR, lambda: nvml.uint("nvmlDeviceGetMinorNumber", handle))
    processes = []
    compute, error = running_processes(
        nvml, handle, "nvmlDeviceGetComputeRunningProcesses", "compute"
    )
    if error:
        card[F_ERRORS]["compute_processes"] = error
    processes.extend(compute)
    graphics, error = running_processes(
        nvml, handle, "nvmlDeviceGetGraphicsRunningProcesses", "graphics"
    )
    if error:
        card[F_ERRORS]["graphics_processes"] = error
    seen = {entry["pid"] for entry in processes}
    processes.extend(entry for entry in graphics if entry["pid"] not in seen)
    card[F_PROCESSES] = processes
    return card


def scrape_nvml():
    result = {
        "library": "",
        "hashes": {},
        "errors": [],
        "driver": "",
        "cuda": "",
        "nvml_version": "",
        "count": 0,
        "gpus": [],
    }
    path = find_nvml()
    if not path:
        result["errors"].append("LIBRARY_NOT_FOUND")
        return result
    result["library"] = path
    try:
        nvml = Nvml(path)
    except OSError as exc:
        result["errors"].append("LIBRARY_NOT_FOUND")
        result["load_error"] = str(exc)
        return result
    result["hashes"] = nvml.hashes
    try:
        code = nvml.call("nvmlInit_v2")
        if code != 0:
            failure = NvmlFailure(code, "nvmlInit_v2")
            nvml.note(failure)
            result["init_error"] = nvml.error_string(code)
            result["errors"] = list(nvml.errors)
            return result
        try:
            buffer = ctypes.create_string_buffer(96)
            if nvml.call("nvmlSystemGetDriverVersion", buffer, ctypes.c_uint(96)) == 0:
                result["driver"] = buffer.value.decode("utf-8", "replace")
            buffer = ctypes.create_string_buffer(96)
            if nvml.call("nvmlSystemGetNVMLVersion", buffer, ctypes.c_uint(96)) == 0:
                result["nvml_version"] = buffer.value.decode("utf-8", "replace")
            cuda = ctypes.c_int(0)
            if (
                nvml.call("nvmlSystemGetCudaDriverVersion_v2", ctypes.byref(cuda)) == 0
                or nvml.call("nvmlSystemGetCudaDriverVersion", ctypes.byref(cuda)) == 0
            ):
                result["cuda"] = f"{cuda.value // 1000}.{(cuda.value % 1000) // 10}"
            count = ctypes.c_uint(0)
            code = nvml.call("nvmlDeviceGetCount_v2", ctypes.byref(count))
            if code != 0:
                nvml.note(NvmlFailure(code, "nvmlDeviceGetCount_v2"))
            else:
                result["count"] = int(count.value)
                for index in range(int(count.value)):
                    result["gpus"].append(scrape_gpu(nvml, index))
        finally:
            nvml.call("nvmlShutdown")
    finally:
        result["errors"] = list(nvml.errors)
    return result


def os_release():
    text = read(os.path.join(HOST_ROOT, "etc/os-release"), 8192) or read("/etc/os-release", 8192)
    fields = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip().strip('"')
    return fields


def cpu_facts():
    info = read("/proc/cpuinfo", 1 << 20)
    model = ""
    for line in info.splitlines():
        if line.lower().startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 0
    load = read("/proc/loadavg", 128).split()
    return {"model": model, "count": count, "load": load[:3]}


def memory_facts():
    total = 0
    available = 0
    for line in read("/proc/meminfo", 8192).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            if parts[0] == "MemTotal:":
                total = int(parts[1]) // 1024
            elif parts[0] == "MemAvailable:":
                available = int(parts[1]) // 1024
    return {"total_mb": total, "available_mb": available}


def block_rotational(path):
    try:
        device = os.stat(path).st_dev
    except OSError:
        return ROTATIONAL_UNKNOWN, "stat failed"
    wanted = f"{os.major(device)}:{os.minor(device)}"
    base = "/sys/class/block"
    try:
        names = os.listdir(base)
    except OSError:
        return ROTATIONAL_UNKNOWN, "no sysfs"
    for name in names:
        if read(os.path.join(base, name, "dev"), 32).strip() != wanted:
            continue
        real = os.path.realpath(os.path.join(base, name))
        for candidate in (real, os.path.dirname(real)):
            flag = read(os.path.join(candidate, "queue", "rotational"), 8).strip()
            if flag in ("0", "1"):
                return flag == "1", name
        return ROTATIONAL_UNKNOWN, name
    code, out, _err = run(["lsblk", "-d", "-n", "-o", "NAME,ROTA,TYPE"], timeout=10)
    if code == 0:
        disks = [line.split() for line in out.splitlines() if line.strip()]
        disks = [parts for parts in disks if len(parts) >= 3 and parts[2] == "disk"]
        if disks and all(parts[1] == "0" for parts in disks):
            return False, "lsblk all disks"
        if disks and all(parts[1] == "1" for parts in disks):
            return True, "lsblk all disks"
    return ROTATIONAL_UNKNOWN, f"unresolved {wanted}"


def disk_facts(docker_root):
    host_root = HOST_ROOT if os.path.isdir(HOST_ROOT) else "/"
    result = {}
    for label, path in (
        ("root", host_root),
        ("docker_root", os.path.join(HOST_ROOT, docker_root.lstrip("/")) if docker_root else ""),
    ):
        if not path or not os.path.isdir(path):
            continue
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            result[label] = {"path": path, "error": str(exc)}
            continue
        rotational, method = block_rotational(path)
        result[label] = {
            "path": path,
            "total_gb": round(usage.total / 1e9, 1),
            "free_gb": round(usage.free / 1e9, 1),
            "rotational": rotational,
            "method": method,
        }
    return result


def docker_facts():
    code, out, err = run(["docker", "info", "--format", "{{json .}}"], timeout=20)
    if code != 0 or not out.strip():
        return {"available": False, "error": (err or "docker info failed").strip()[:300]}
    try:
        info = json.loads(out)
    except ValueError:
        return {"available": False, "error": "docker info is not JSON"}
    return {
        "available": True,
        "server_version": str(info.get("ServerVersion", "")),
        "runtimes": sorted((info.get("Runtimes") or {}).keys()),
        "default_runtime": str(info.get("DefaultRuntime", "")),
        "driver": str(info.get("Driver", "")),
        "driver_status": info.get("DriverStatus") or [],
        "root_dir": str(info.get("DockerRootDir", "")),
        "cgroup_driver": str(info.get("CgroupDriver", "")),
        "cgroup_version": str(info.get("CgroupVersion", "")),
        "containers_running": info.get("ContainersRunning"),
        "images": info.get("Images"),
    }


def own_container_id():
    match = CONTAINER_ID.search(read("/proc/self/cgroup", 4096))
    if match:
        return match.group(1)
    match = CONTAINER_ID.search(read("/proc/self/mountinfo", 1 << 16))
    return match.group(1) if match else ""


def probe_image():
    own = own_container_id()
    if own:
        code, out, _err = run(["docker", "inspect", "--format", "{{.Image}}", own], timeout=15)
        if code == 0 and out.strip():
            return out.strip(), "own container"
    code, out, _err = run(["docker", "images", "-q"], timeout=15)
    if code == 0:
        for line in out.splitlines():
            if line.strip():
                return line.strip(), "first local image"
    return "", "no local image"


def probe(argv, marker=None):
    started = time.time()
    code, out, err = run(argv, timeout=PROBE_TIMEOUT)
    ok = code == 0 and (marker is None or marker in out)
    return {
        "ok": ok,
        "code": code,
        "stdout": out.strip()[-300:],
        "stderr": err.strip()[-300:],
        "elapsed_ms": round((time.time() - started) * 1000.0, 1),
        "command": " ".join(argv),
    }


def idmapped_facts(kernel_tuple, runtimes):
    for argv in (
        ["chroot", HOST_ROOT, "journalctl", "-u", "sysbox-mgr", "-b", "--no-pager", "-o", "cat"],
        ["journalctl", "-u", "sysbox-mgr", "-b", "--no-pager", "-o", "cat"],
    ):
        code, out, _err = run(argv, timeout=15)
        if code != 0:
            continue
        verdict = None
        for line in out.splitlines():
            match = re.search(r"ID-mapped mounts supported by kernel:\s*(yes|no)", line)
            if match:
                verdict = match.group(1) == "yes"
        if verdict is not None:
            return {"idmapped": verdict, "method": "sysbox-mgr journal"}
    supported = kernel_tuple >= (5, 19) and "sysbox-runc" in runtimes
    return {"idmapped": supported, "method": "kernel version and runtime presence"}


def kernel_tuple(text):
    match = re.match(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def host_facts(docker):
    release = os_release()
    kernel = platform.release()
    hostname = read(os.path.join(HOST_ROOT, "etc/hostname"), 256).strip() or socket.gethostname()
    facts = {
        "cpu": cpu_facts(),
        "memory": memory_facts(),
        "disk": disk_facts(docker.get("root_dir", "")),
        "os": {
            "id": release.get("ID", "").lower(),
            "version": release.get("VERSION_ID", ""),
            "pretty": release.get("PRETTY_NAME", ""),
        },
        "kernel": kernel,
        "kernel_tuple": list(kernel_tuple(kernel)),
        "arch": platform.machine(),
        "hostname": hostname,
        "boot_id": read("/proc/sys/kernel/random/boot_id", 64).strip(),
        "machine_id": read(os.path.join(HOST_ROOT, "etc/machine-id"), 64).strip()
        or read("/etc/machine-id", 64).strip(),
        "uptime_seconds": float((read("/proc/uptime", 64).split() or ["0"])[0]),
        "rm_profiling_admin_only": read("/proc/driver/nvidia/params", 4096).strip()[:800],
        "nvidia_devices": sorted(
            name
            for name in (os.listdir("/dev") if os.path.isdir("/dev") else [])
            if name.startswith("nvidia")
        ),
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "python": sys.version.split()[0],
        "uid": os.getuid(),
    }
    return facts


def probes(docker, kernel):
    image, source = probe_image()
    runtimes = docker.get("runtimes") or []
    result = {"image": image, "image_source": source}
    if not docker.get("available"):
        result["sysbox"] = {"ok": False, "skipped": "docker unavailable"}
        result["storage_quota"] = {"ok": False, "skipped": "docker unavailable"}
    elif not image:
        result["sysbox"] = {"ok": False, "skipped": "no local image"}
        result["storage_quota"] = {"ok": False, "skipped": "no local image"}
    else:
        if "sysbox-runc" in runtimes:
            result["sysbox"] = probe(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--runtime=sysbox-runc",
                    "--gpus",
                    "all",
                    image,
                    "nvidia-smi",
                    "-L",
                ],
                marker="GPU 0",
            )
        else:
            result["sysbox"] = {"ok": False, "skipped": "sysbox-runc runtime not registered"}
        result["storage_quota"] = probe(
            ["docker", "run", "--rm", "--storage-opt", "size=1g", image, "true"]
        )
    result["idmapped"] = idmapped_facts(kernel, runtimes)
    return result


def main():
    if os.environ.get("LD_PRELOAD"):
        sys.stdout.write(json.dumps({"error": "LD_PRELOAD is set"}) + "\n")
        return 1
    started = time.time()
    try:
        nvml = scrape_nvml()
        docker = docker_facts()
        host = host_facts(docker)
        checks = probes(docker, tuple(host["kernel_tuple"]))
        hashes = {
            "nvidia_smi": file_hashes(shutil.which("nvidia-smi") or ""),
            "nvml": nvml.get("hashes") or {},
            "docker": file_hashes(shutil.which("docker") or ""),
        }
        payload = {
            "version": 1,
            "collected_at": round(time.time(), 3),
            "elapsed_ms": round((time.time() - started) * 1000.0, 1),
            "gpus": nvml.pop("gpus"),
            "nvml": nvml,
            "host": host,
            "docker": docker,
            "probes": checks,
            "hashes": hashes,
        }
        token = fernet_encrypt(
            derive_key(), json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        )
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
        return 1
    sys.stdout.write(token + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
