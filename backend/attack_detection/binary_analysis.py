"""Minimal read-only PE/DLL parser.

It reads bounded headers and sections only. It does not load or execute a PE file.
"""

from __future__ import annotations

import math
import struct
import time
import zlib
from collections import Counter
from typing import Any

from .contracts import EngineResult
from .static_analysis.strings_ioc import classify_iocs, printable_strings


SUSPICIOUS_IMPORTS = {
    "virtualalloc", "virtualprotect", "createremotethread", "writeprocessmemory",
    "winexec", "shellexecutea", "shellexecutew", "urlmon", "urldownloadtofilea",
    "internetopen", "winhttpopen", "regsetvalue", "adjusttokenprivileges",
}

MAX_DECOMPRESSED_PE_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSION_RATIO = 200


def _unwrap_zlib_pe(data: bytes) -> tuple[bytes, str | None]:
    """Unwrap a bounded zlib-compressed PE used by datasets such as SOREL.

    Dataset objects are compressed blobs even though their object names are PE
    hashes.  Treating the blob itself as an executable makes a valid sample
    appear to be an unknown non-PE file.  Decompression is bounded to avoid
    turning file analysis into an archive-bomb primitive.
    """

    if data[:2] == b"MZ" or len(data) < 2:
        return data, None
    # RFC 1950 header check: CM=deflate and the two-byte header is divisible
    # by 31. This avoids trying to inflate every arbitrary binary upload.
    if data[0] & 0x0F != 8 or int.from_bytes(data[:2], "big") % 31:
        return data, None
    try:
        inflater = zlib.decompressobj()
        output = inflater.decompress(data, MAX_DECOMPRESSED_PE_BYTES + 1)
        if inflater.unconsumed_tail or len(output) > MAX_DECOMPRESSED_PE_BYTES:
            return data, None
        output += inflater.flush(MAX_DECOMPRESSED_PE_BYTES + 1 - len(output))
    except zlib.error:
        return data, None
    if (
        not inflater.eof
        or len(output) > MAX_DECOMPRESSED_PE_BYTES
        or len(output) > max(1, len(data)) * MAX_DECOMPRESSION_RATIO
        or output[:2] != b"MZ"
    ):
        return data, None
    return output, "zlib"


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    size = len(data)
    return round(-sum((count / size) * math.log2(count / size) for count in counts.values()), 3)


def _read_c_string(data: bytes, offset: int, limit: int = 256) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset, min(len(data), offset + limit))
    return data[offset:(end if end >= 0 else min(len(data), offset + limit))].decode("ascii", errors="ignore")


def _rva_to_offset(rva: int, sections: list[dict[str, int]]) -> int | None:
    for section in sections:
        start = section["virtual_address"]
        end = start + max(section["virtual_size"], section["raw_size"])
        if start <= rva < end:
            return section["raw_pointer"] + (rva - start)
    return None


def _parse_import_table(
    data: bytes,
    optional_offset: int,
    optional_size: int,
    magic: int,
    sections: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Parse a bounded PE import table without loading the executable."""

    data_directory_offset = optional_offset + (112 if magic == 0x20B else 96)
    number_offset = data_directory_offset - 4
    optional_end = optional_offset + optional_size
    if number_offset + 4 > optional_end or data_directory_offset + 16 > optional_end:
        return []
    directory_count = struct.unpack_from("<I", data, number_offset)[0]
    if directory_count < 2:
        return []
    import_rva, import_size = struct.unpack_from("<II", data, data_directory_offset + 8)
    if not import_rva:
        return []
    descriptor_offset = _rva_to_offset(import_rva, sections)
    if descriptor_offset is None:
        return []

    imports: list[dict[str, Any]] = []
    thunk_size = 8 if magic == 0x20B else 4
    thunk_format = "<Q" if thunk_size == 8 else "<I"
    ordinal_mask = 1 << (63 if thunk_size == 8 else 31)
    maximum_descriptors = min(256, max(1, import_size // 20) if import_size else 256)
    total_functions = 0
    for descriptor_index in range(maximum_descriptors):
        offset = descriptor_offset + descriptor_index * 20
        if offset + 20 > len(data):
            break
        original_thunk, timestamp, forwarder, name_rva, first_thunk = struct.unpack_from(
            "<IIIII", data, offset
        )
        if not any((original_thunk, timestamp, forwarder, name_rva, first_thunk)):
            break
        name_offset = _rva_to_offset(name_rva, sections)
        library = _read_c_string(data, name_offset if name_offset is not None else -1, 260)
        if not library:
            library = f"unknown_{descriptor_index}"
        thunk_rva = original_thunk or first_thunk
        thunk_offset = _rva_to_offset(thunk_rva, sections)
        functions: list[str] = []
        if thunk_offset is not None:
            for thunk_index in range(2048):
                value_offset = thunk_offset + thunk_index * thunk_size
                if value_offset + thunk_size > len(data) or total_functions >= 4096:
                    break
                value = struct.unpack_from(thunk_format, data, value_offset)[0]
                if not value:
                    break
                if value & ordinal_mask:
                    functions.append(f"ordinal:{value & 0xFFFF}")
                else:
                    import_name_offset = _rva_to_offset(int(value), sections)
                    if import_name_offset is None or import_name_offset + 2 > len(data):
                        continue
                    function_name = _read_c_string(data, import_name_offset + 2, 260)
                    if function_name:
                        functions.append(function_name)
                total_functions += 1
        imports.append({"dll": library, "functions": functions})
        if total_functions >= 4096:
            break
    return imports


def parse_pe(data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"is_pe": False, "sections": [], "imports": [], "warnings": []}
    if len(data) < 64 or data[:2] != b"MZ":
        return result
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 24 > len(data) or data[pe_offset:pe_offset + 4] != b"PE\x00\x00":
        result["warnings"].append("检测到 MZ 文件头，但 PE 签名无效")
        return result
    machine, section_count, timestamp, _, _, optional_size, characteristics = struct.unpack_from("<HHIIIHH", data, pe_offset + 4)
    optional_offset = pe_offset + 24
    if optional_offset + optional_size > len(data) or optional_size < 2:
        result["warnings"].append("PE 可选头不完整")
        return result
    magic = struct.unpack_from("<H", data, optional_offset)[0]
    if magic not in {0x10B, 0x20B}:
        result["warnings"].append(f"未知的 PE 可选头标识 0x{magic:x}")
        return result
    result.update({"is_pe": True, "machine": f"0x{machine:04x}", "section_count": section_count, "timestamp": timestamp, "characteristics": f"0x{characteristics:04x}", "optional_magic": f"0x{magic:04x}"})
    section_offset = optional_offset + optional_size
    sections: list[dict[str, int]] = []
    for index in range(min(section_count, 96)):
        offset = section_offset + index * 40
        if offset + 40 > len(data):
            result["warnings"].append("PE 节区表不完整")
            break
        name = data[offset:offset + 8].split(b"\x00", 1)[0].decode("ascii", errors="replace")
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from("<IIII", data, offset + 8)
        section_bytes = data[raw_pointer:min(len(data), raw_pointer + raw_size)] if raw_pointer < len(data) else b""
        item = {"name": name, "virtual_size": virtual_size, "virtual_address": virtual_address, "raw_size": raw_size, "raw_pointer": raw_pointer, "entropy": _entropy(section_bytes)}
        sections.append(item)
    result["sections"] = sections
    result["imports"] = _parse_import_table(
        data,
        optional_offset,
        optional_size,
        magic,
        sections,
    )
    result["overlay_bytes"] = max(0, len(data) - max((item["raw_pointer"] + item["raw_size"] for item in sections), default=0))
    suspicious_sections = [item["name"] for item in sections if item["entropy"] >= 7.2]
    result["high_entropy_sections"] = suspicious_sections
    strings = [value for _, value in printable_strings(data, minimum=5, limit=1200)]
    lowered = "\n".join(strings).lower()
    result["suspicious_strings"] = [value[:240] for value in strings if any(token in value.lower() for token in ("powershell", "cmd.exe", "rundll32", "regsvr32", "appdata", "http://", "https://", "virtualalloc", "createremotethread"))][:80]
    imported_names = {
        str(function).lower()
        for library in result["imports"]
        for function in library.get("functions", [])
    }
    result["import_indicators"] = sorted({
        token
        for token in SUSPICIOUS_IMPORTS
        if token in lowered or token in imported_names
    })
    return result


class BinaryAnalysisEngine:
    name = "pe_static"

    def scan(self, filename: str, payload: bytes) -> dict[str, Any]:
        start = time.perf_counter()
        analysis_payload, container = _unwrap_zlib_pe(payload)
        parsed = parse_pe(analysis_payload)
        if not parsed["is_pe"]:
            return EngineResult(name=self.name, status="completed", decision="benign", duration_ms=int((time.perf_counter() - start) * 1000), metadata={"is_pe": False, "format": "unknown_or_non_pe"}).to_dict()
        findings: list[dict[str, Any]] = []
        extracted_strings = [value for _, value in printable_strings(analysis_payload, minimum=5, limit=1200)]
        ioc_findings = classify_iocs("\n".join(extracted_strings))
        findings.extend(ioc_findings[:40])
        for section in parsed["sections"]:
            if section["entropy"] >= 7.2:
                findings.append({
                    "source": self.name, "rule_id": "PE-HIGH-ENTROPY", "category": "PE 高熵节区", "risk_type": "context", "behavior": "packed_or_encrypted_section", "severity": 3,
                    "line": None, "snippet": section["name"], "evidence": f"节区 {section['name']} 熵为 {section['entropy']}，可能是压缩/加密或打包。",
                    "description": "高熵 PE 节区是结构线索，不等于恶意；需要结合签名、导入和动态行为复核。", "repair_advice": "核对发布链、签名和打包器来源，不要仅凭熵值定性。", "confidence": 0.5,
                })
        if parsed["import_indicators"]:
            findings.append({
                "source": self.name, "rule_id": "PE-SUSPICIOUS-IMPORT", "category": "PE 可疑导入", "risk_type": "context", "behavior": "suspicious_native_api", "severity": 4,
                "line": None, "snippet": ", ".join(parsed["import_indicators"]), "evidence": "字符串/导入线索包含高风险原生 API: " + ", ".join(parsed["import_indicators"]),
                "description": "可疑原生 API 常见于注入、下载或持久化能力，也可能用于合法安全软件。", "repair_advice": "结合调用点、签名和沙箱事件确认 API 的实际用途。", "confidence": 0.55,
            })
        if parsed["suspicious_strings"]:
            findings.append({
                "source": self.name, "rule_id": "PE-SUSPICIOUS-STRING", "category": "PE 可疑字符串", "risk_type": "context", "behavior": "suspicious_binary_string", "severity": 2,
                "line": None, "snippet": parsed["suspicious_strings"][0], "evidence": f"发现 {len(parsed['suspicious_strings'])} 条需要核对的二进制字符串。",
                "description": "二进制字符串可帮助定位网络、脚本或系统 API 线索，但不构成单独恶意结论。", "repair_advice": "核对字符串与导入表、签名及动态事件的关联。", "confidence": 0.45,
            })
        score = min(55, sum(int(item["severity"]) for item in findings) * 5)
        return EngineResult(name=self.name, status="completed", decision="unknown" if findings else "benign", risk_score=score, duration_ms=int((time.perf_counter() - start) * 1000), findings=findings, metadata={"is_pe": True, "format": "PE", "container": container, "parser": "bounded_read_only", "machine": parsed.get("machine"), "section_count": parsed.get("section_count"), "timestamp": parsed.get("timestamp"), "sections": parsed.get("sections", []), "overlay_bytes": parsed.get("overlay_bytes", 0), "import_indicators": parsed.get("import_indicators", []), "suspicious_strings": parsed.get("suspicious_strings", []), "string_count": len(extracted_strings), "ioc_count": len(ioc_findings), "warnings": parsed.get("warnings", [])}).to_dict()
