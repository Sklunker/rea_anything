import type { Element, Node } from "@xmldom/xmldom";
import { z } from "zod";
import type {
  BoardDeviceAttachment,
  BoardMemoryWindow,
} from "./boardPeripheralProfile.js";
import type { Devicetree } from "./devicetree.js";
import { firstKeyed, keyedBy, uniqueKeyed } from "../keyedCollection.js";
import { err, ok, type Result } from "../result.js";
import { countXmlNodes, parseXmlDocument } from "../xmlDocument.js";

export const CUSTOM_HARDWARE_SOURCES = [
  "devicetree",
  "cmsis_svd",
  "explicit",
] as const;
export type CustomHardwareSource = (typeof CUSTOM_HARDWARE_SOURCES)[number];

export interface CustomPeripheralRegister {
  readonly offset: bigint;
  readonly mnemonic: string;
  readonly name: string;
}

export interface CustomPeripheralWindow {
  readonly name: string;
  readonly baseAddress: string;
  readonly baseAddressValue: bigint;
  readonly sizeBytes: number;
  readonly bus: string | null;
  readonly compatible: readonly string[];
  readonly status: string | null;
  readonly registers: readonly CustomPeripheralRegister[];
}

export interface CustomHardwareProfile {
  readonly id: string;
  readonly sourceKind: CustomHardwareSource;
  readonly sourceSha256: string | null;
  readonly model: string | null;
  readonly memory: readonly BoardMemoryWindow[];
  readonly peripherals: readonly CustomPeripheralWindow[];
  readonly devices: readonly BoardDeviceAttachment[];
  readonly limitations: readonly string[];
}

const customPeripheralRegisterSchema = z.strictObject({
  offset: z.bigint().nonnegative(),
  mnemonic: z.string().min(1),
  name: z.string().min(1),
});

const customPeripheralWindowSchema = z.strictObject({
  name: z.string().min(1),
  baseAddress: z.string(),
  baseAddressValue: z.bigint().nonnegative(),
  sizeBytes: z.number().int().positive(),
  bus: z.string().nullable(),
  compatible: z.array(z.string()),
  status: z.string().nullable(),
  registers: z.array(customPeripheralRegisterSchema),
});

export const customHardwareProfileSchema = z.strictObject({
  id: z.string().min(1),
  sourceKind: z.enum(CUSTOM_HARDWARE_SOURCES),
  sourceSha256: z.string().nullable(),
  model: z.string().nullable(),
  memory: z.array(
    z.strictObject({
      id: z.string(),
      name: z.string(),
      kind: z.enum([
        "code",
        "sram",
        "peripheral_mmio",
        "external_device",
        "system_control",
        "reserved",
      ]),
      baseAddress: z.string(),
      baseAddressValue: z.bigint(),
      sizeBytes: z.number().int().positive(),
      evidence: z.string(),
    }),
  ),
  peripherals: z.array(customPeripheralWindowSchema),
  devices: z.array(
    z.strictObject({
      id: z.string(),
      name: z.string(),
      kind: z.enum(["listed_resource", "bsp_option"]),
      peripheralName: z.string().nullable(),
      role: z.string(),
      kconfigDefault: z.boolean().nullable(),
      pins: z.array(z.string()),
      evidence: z.string(),
      uncertainty: z.string().optional(),
      partNames: z.array(z.string()).optional(),
    }),
  ),
  limitations: z.array(z.string()),
});

const SVD_REGISTER_PRECEDENCE =
  "CMSIS-SVD alternate registers may share an addressOffset; select primary declaration";
const FDT_PROPERTY_PRECEDENCE =
  "Devicetree specification v0.4 §2.2.4 requires unique property names per node; first declaration selected";

export type CustomPeripheralLookupResult =
  | {
      readonly kind: "matched";
      readonly peripheral: CustomPeripheralWindow;
      readonly register: CustomPeripheralRegister | null;
    }
  | {
      readonly kind: "ambiguous";
      readonly baseAddress: string;
      readonly baseAddressValue: bigint;
      readonly candidates: readonly CustomPeripheralWindow[];
    };

export const formatAmbiguousPeripheralCandidates = (
  candidates: readonly CustomPeripheralWindow[],
): readonly string[] => {
  const counts = new Map<string, number>();
  for (const item of candidates) {
    counts.set(item.name, (counts.get(item.name) ?? 0) + 1);
  }
  return candidates
    .map((item) =>
      (counts.get(item.name) ?? 0) > 1 && item.bus !== null
        ? `${item.name} (${item.bus})`
        : item.name,
    )
    .sort();
};

/**
 * Look up the most specific peripheral window and optional register offset in a
 * custom profile. When a parent bus/syscon window contains a smaller child
 * controller window, the narrower child window wins; equal-sized overlapping
 * windows return `kind: "ambiguous"` so MMIO semantics survive while controller
 * naming stays unresolved.
 */
export const lookupCustomPeripheral = (
  address: bigint,
  profile: CustomHardwareProfile,
): CustomPeripheralLookupResult | null => {
  const containing = profile.peripherals.filter((peripheral) => {
    const end = peripheral.baseAddressValue + BigInt(peripheral.sizeBytes);
    return address >= peripheral.baseAddressValue && address < end;
  });
  if (containing.length === 0) return null;

  let minSize = containing[0]?.sizeBytes ?? 0;
  for (const candidate of containing) {
    if (candidate.sizeBytes < minSize) minSize = candidate.sizeBytes;
  }
  const narrowest = containing.filter((item) => item.sizeBytes === minSize);
  const selected = narrowest[0];
  if (selected === undefined) return null;
  if (narrowest.length > 1) {
    let baseAddressValue = selected.baseAddressValue;
    for (const item of narrowest) {
      if (item.baseAddressValue > baseAddressValue) {
        baseAddressValue = item.baseAddressValue;
      }
    }
    return {
      kind: "ambiguous",
      baseAddress: `0x${baseAddressValue.toString(16)}`,
      baseAddressValue,
      candidates: narrowest,
    };
  }

  const offset = address - selected.baseAddressValue;
  const keyedRegs = keyedBy(selected.registers, (entry) =>
    entry.offset.toString(16),
  );
  const register =
    firstKeyed(keyedRegs, offset.toString(16), SVD_REGISTER_PRECEDENCE) ?? null;
  return { kind: "matched", peripheral: selected, register };
};

interface RangeTuple {
  readonly childStart: bigint;
  readonly parentStart: bigint;
  readonly length: bigint;
}

type BusMapping =
  | { readonly kind: "identity" }
  | { readonly kind: "translated"; readonly tuples: readonly RangeTuple[] }
  | { readonly kind: "untranslated"; readonly reason: string }
  | { readonly kind: "device_bus" };

const UNSIGNED_INT_TEXT = /^(?:0x[0-9a-fA-F]+|\d+)$/u;

const parseUnsignedBigInt = (raw: string | null): bigint | null => {
  if (raw === null) return null;
  const trimmed = raw.trim();
  if (!UNSIGNED_INT_TEXT.test(trimmed)) return null;
  try {
    const parsed = BigInt(trimmed);
    return parsed < 0n ? null : parsed;
  } catch {
    return null;
  }
};

const readUintCells = (
  cells: readonly number[],
  cellOffset: number,
  cellCount: number,
): bigint => {
  let value = 0n;
  for (let index = 0; index < cellCount; index += 1) {
    value = (value << 32n) | BigInt((cells[cellOffset + index] ?? 0) >>> 0);
  }
  return value;
};

const readHexCells = (
  hex: string,
  cellOffset: number,
  cellCount: number,
): bigint => {
  let value = 0n;
  for (let index = 0; index < cellCount; index += 1) {
    const start = (cellOffset + index) * 8;
    value = (value << 32n) | BigInt(`0x${hex.slice(start, start + 8)}`);
  }
  return value;
};

const nodeShortName = (path: string): string => {
  const leaf = path.split("/").at(-1) ?? path;
  const base = leaf.split("@")[0];
  return (base && base.length > 0 ? base : leaf).toUpperCase();
};

const nodeProperty = (
  node: Devicetree["nodes"][number],
  name: string,
): Devicetree["nodes"][number]["properties"][number] | undefined =>
  firstKeyed(
    keyedBy(node.properties, (prop) => prop.name),
    name,
    FDT_PROPERTY_PRECEDENCE,
  );

const validateAcyclicDevicetree = (
  nodes: Devicetree["nodes"],
  byPath: ReadonlyMap<string, Devicetree["nodes"][number]>,
): Result<void, string> => {
  for (const node of nodes) {
    if (node.path === "/") {
      if (node.parent !== null) {
        return err("Devicetree root node '/' must have a null parent");
      }
      continue;
    }
    if (node.parent === null) {
      return err(`Devicetree non-root node '${node.path}' is missing a parent`);
    }
    const seen = new Set<string>([node.path]);
    let cursor: string | null = node.parent;
    while (cursor !== null) {
      if (seen.has(cursor)) {
        return err(
          `Devicetree contains a cyclic parent link involving '${node.path}'`,
        );
      }
      seen.add(cursor);
      const parentNode = byPath.get(cursor);
      if (parentNode === undefined) {
        return err(
          `Devicetree node '${node.path}' references missing parent '${cursor}'`,
        );
      }
      cursor = parentNode.parent;
    }
  }
  return ok(undefined);
};

const parseBusMapping = (
  node: Devicetree["nodes"][number],
  byPath: ReadonlyMap<string, Devicetree["nodes"][number]>,
): BusMapping => {
  if (node.path === "/") return { kind: "identity" };
  const ranges = nodeProperty(node, "ranges");
  if (ranges === undefined) return { kind: "device_bus" };
  if (ranges.bytes === 0) return { kind: "identity" };
  const cells = ranges.derived.cells;
  if (
    cells === null &&
    (ranges.truncated ||
      ranges.hex.length < ranges.bytes * 2 ||
      !/^[0-9a-fA-F]*$/u.test(ranges.hex))
  ) {
    return {
      kind: "untranslated",
      reason: `ranges at ${node.path} exceeds 256-byte preview; child reg addresses were not mapped to CPU space.`,
    };
  }
  const childAddrCells =
    nodeProperty(node, "#address-cells")?.derived.cells?.[0] ?? 2;
  const childSizeCells =
    nodeProperty(node, "#size-cells")?.derived.cells?.[0] ?? 1;
  const parentNode = node.parent === null ? undefined : byPath.get(node.parent);
  const parentAddrCells =
    (parentNode === undefined
      ? undefined
      : nodeProperty(parentNode, "#address-cells")
    )?.derived.cells?.[0] ?? 2;
  const tupleCells = childAddrCells + parentAddrCells + childSizeCells;
  if (
    tupleCells === 0 ||
    childSizeCells === 0 ||
    ranges.bytes % (tupleCells * 4) !== 0 ||
    (cells !== null && cells.length % tupleCells !== 0)
  ) {
    return {
      kind: "untranslated",
      reason: `ranges at ${node.path} has unsupported cell widths.`,
    };
  }
  const tuples: RangeTuple[] = [];
  const totalCells = cells !== null ? cells.length : ranges.bytes / 4;
  const readRangeCells = (offset: number, count: number): bigint =>
    cells !== null
      ? readUintCells(cells, offset, count)
      : readHexCells(ranges.hex, offset, count);
  for (let cell = 0; cell < totalCells; cell += tupleCells) {
    tuples.push({
      childStart: readRangeCells(cell, childAddrCells),
      parentStart: readRangeCells(cell + childAddrCells, parentAddrCells),
      length: readRangeCells(
        cell + childAddrCells + parentAddrCells,
        childSizeCells,
      ),
    });
  }
  return { kind: "translated", tuples };
};

const translateToCpuAddress = (
  startParentPath: string | null,
  rawAddress: bigint,
  rawSize: bigint,
  ctx: {
    readonly byPath: ReadonlyMap<string, Devicetree["nodes"][number]>;
    readonly mappings: ReadonlyMap<string, BusMapping>;
  },
): Result<bigint, "device_bus" | string> => {
  let currentPath = startParentPath;
  let currentAddr = rawAddress;
  while (currentPath !== null) {
    const mapping = ctx.mappings.get(currentPath);
    const busNode = ctx.byPath.get(currentPath);
    if (mapping === undefined || busNode === undefined) {
      return err(`Missing parent node ${currentPath}`);
    }
    if (mapping.kind === "device_bus") return err("device_bus");
    if (mapping.kind === "untranslated") return err(mapping.reason);
    if (mapping.kind === "translated") {
      let matchedTuple: RangeTuple | undefined;
      for (const tuple of mapping.tuples) {
        if (
          currentAddr >= tuple.childStart &&
          currentAddr + rawSize <= tuple.childStart + tuple.length
        ) {
          matchedTuple = tuple;
          break;
        }
      }
      if (matchedTuple === undefined) {
        return err(
          `Address 0x${currentAddr.toString(16)} is outside ranges of ${currentPath}.`,
        );
      }
      currentAddr =
        matchedTuple.parentStart + (currentAddr - matchedTuple.childStart);
    }
    currentPath = busNode.parent;
  }
  return ok(currentAddr);
};

const NON_MMIO_PREFIXES = [
  "/cpus",
  "/chosen",
  "/aliases",
  "/reserved-memory",
  "/__symbols__",
] as const;

/**
 * Build a dynamic hardware peripheral, memory, and child-device profile from a
 * parsed Flattened Devicetree (`Devicetree`), validating parent links and `reg`
 * encodings before translating bus `ranges`.
 */
export const hardwareProfileFromDevicetree = (
  dt: Devicetree,
): Result<CustomHardwareProfile, string> => {
  const byPath = new Map(dt.nodes.map((node) => [node.path, node]));
  const treeCheck = validateAcyclicDevicetree(dt.nodes, byPath);
  if (!treeCheck.ok) return treeCheck;

  const mappings = new Map(
    dt.nodes.map((node) => [node.path, parseBusMapping(node, byPath)]),
  );
  const memory: BoardMemoryWindow[] = [];
  const peripherals: CustomPeripheralWindow[] = [];
  const devices: BoardDeviceAttachment[] = [];
  const limitations = [...dt.limitations];
  const controllerNameByPath = new Map<string, string>();

  for (const node of dt.nodes) {
    if (
      node.path === "/" ||
      NON_MMIO_PREFIXES.some(
        (prefix) => node.path === prefix || node.path.startsWith(`${prefix}/`),
      )
    ) {
      continue;
    }
    const regEntries = nodeProperty(node, "reg")?.derived.reg ?? null;
    if (regEntries === null || regEntries.length === 0) continue;

    const compatible = nodeProperty(node, "compatible")?.derived.strings ?? [];
    const status = nodeProperty(node, "status")?.derived.strings?.[0] ?? null;
    if (status === "disabled") continue;
    const isMemory =
      node.path === "/memory" ||
      node.path.startsWith("/memory@") ||
      compatible.includes("mmio-sram");

    if (
      node.parent !== null &&
      mappings.get(node.parent)?.kind === "device_bus"
    ) {
      const firstAddr = regEntries[0]?.address ?? null;
      if (firstAddr !== null) {
        if (parseUnsignedBigInt(firstAddr) === null) {
          return err(
            `Devicetree node '${node.path}' has invalid reg address '${firstAddr}'`,
          );
        }
        devices.push({
          id: node.path,
          name: compatible[0] ?? nodeShortName(node.path),
          kind: "listed_resource",
          peripheralName: controllerNameByPath.get(node.parent) ?? null,
          role: `Devicetree child at ${firstAddr}`,
          kconfigDefault: status === null || status === "okay",
          pins: [],
          evidence: `DTB ${dt.artifact.sha256} node ${node.path}`,
        });
      }
      continue;
    }

    for (const [regIndex, reg] of regEntries.entries()) {
      if (reg.address === null || reg.size === null) continue;
      const rawAddr = parseUnsignedBigInt(reg.address);
      const rawSize = parseUnsignedBigInt(reg.size);
      if (rawAddr === null || rawSize === null) {
        return err(
          `Devicetree node '${node.path}' has invalid reg entry (${String(reg.address)}, ${String(reg.size)})`,
        );
      }
      if (rawSize <= 0n || rawSize > BigInt(Number.MAX_SAFE_INTEGER)) continue;

      const translated = translateToCpuAddress(node.parent, rawAddr, rawSize, {
        byPath,
        mappings,
      });
      if (!translated.ok) {
        if (translated.error !== "device_bus") {
          limitations.push(translated.error);
        }
        continue;
      }

      const cpuAddr = translated.value;
      const sizeBytes = Number(rawSize);
      if (isMemory) {
        memory.push({
          id: regIndex === 0 ? node.path : `${node.path}#${regIndex}`,
          name: node.path,
          kind: "sram",
          baseAddress: `0x${cpuAddr.toString(16)}`,
          baseAddressValue: cpuAddr,
          sizeBytes,
          evidence: `DTB ${dt.artifact.sha256} memory node ${node.path}`,
        });
        continue;
      }

      const baseLabel = nodeShortName(node.path);
      const name =
        regEntries.length > 1
          ? `${baseLabel}_${regIndex}`
          : `${baseLabel}@0x${cpuAddr.toString(16)}`;
      if (regIndex === 0) controllerNameByPath.set(node.path, name);
      peripherals.push({
        name,
        baseAddress: `0x${cpuAddr.toString(16)}`,
        baseAddressValue: cpuAddr,
        sizeBytes,
        bus: node.parent,
        compatible,
        status,
        registers: [],
      });
    }
  }

  return ok({
    id: dt.summary.model ?? dt.summary.compatible?.[0] ?? dt.artifact.path,
    sourceKind: "devicetree",
    sourceSha256: dt.artifact.sha256,
    model: dt.summary.model,
    memory,
    peripherals,
    devices,
    limitations,
  });
};

const MAX_SVD_CHARS = 16 * 1024 * 1024;
const MAX_SVD_XML_NODES = 500_000;
const MAX_SVD_DIM_ELEMENTS = 1024;
const MAX_SVD_TOTAL_REGISTERS = 32768;
const MAX_SVD_EXPANSION_STEPS = 65536;
const MAX_SVD_CLUSTER_DEPTH = 64;

interface SvdExpansionBudget {
  remainingRegisters: number;
  remainingSteps: number;
}

const elementTagLower = (element: Element): string =>
  (element.localName ?? element.tagName).toLowerCase();

const directChildElements = (parent: Element, tagLower?: string): Element[] => {
  const out: Element[] = [];
  for (
    let child: Node | null = parent.firstChild;
    child !== null;
    child = child.nextSibling
  ) {
    if (child.nodeType !== 1) continue;
    const element = child as Element;
    if (tagLower === undefined || elementTagLower(element) === tagLower) {
      out.push(element);
    }
  }
  return out;
};

const directChildText = (parent: Element, tagLower: string): string | null => {
  for (
    let child: Node | null = parent.firstChild;
    child !== null;
    child = child.nextSibling
  ) {
    if (child.nodeType !== 1) continue;
    const element = child as Element;
    if (elementTagLower(element) !== tagLower) continue;
    const trimmed = (element.textContent ?? "").trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  return null;
};

const elementAttributeCaseInsensitive = (
  element: Element,
  attrLower: string,
): string | null => {
  for (let idx = 0; idx < element.attributes.length; idx += 1) {
    const attr = element.attributes.item(idx);
    if (attr === null) continue;
    if ((attr.localName ?? attr.name).toLowerCase() === attrLower) {
      const trimmed = attr.value.trim();
      return trimmed.length > 0 ? trimmed : null;
    }
  }
  return null;
};

const parseSvdInt = (raw: string | null): bigint | null => {
  if (raw === null || raw.length === 0) return null;
  const trimmed = raw.trim();
  const normalized = trimmed.startsWith("#")
    ? `0b${trimmed.slice(1)}`
    : trimmed;
  if (!/^(?:0x[0-9a-fA-F]+|0b[01]+|\d+)$/u.test(normalized)) return null;
  try {
    const val = BigInt(normalized);
    return val < 0n ? null : val;
  } catch {
    return null;
  }
};

const expandDimIndices = (
  dim: number,
  dimIndexRaw: string | null,
): string[] => {
  if (dimIndexRaw !== null && dimIndexRaw.length > 0) {
    if (dimIndexRaw.includes(",")) {
      const parts = dimIndexRaw.split(",").map((part) => part.trim());
      if (parts.length === dim) return parts;
    }
    const rangeMatch = /^(\d+)-(\d+)$/u.exec(dimIndexRaw);
    if (rangeMatch?.[1] && rangeMatch[2]) {
      const start = Number(rangeMatch[1]);
      const end = Number(rangeMatch[2]);
      if (end >= start && end - start + 1 === dim) {
        return Array.from({ length: dim }, (_, idx) => String(start + idx));
      }
    }
  }
  return Array.from({ length: dim }, (_, idx) => String(idx));
};

const formatDimName = (template: string, indexToken: string): string =>
  template.includes("%s")
    ? template.replaceAll("%s", indexToken)
    : `${template}${indexToken}`;

const parseSvdBitWidthToBytes = (raw: string | null): number | null => {
  const bits = parseSvdInt(raw);
  if (bits === null || bits <= 0n || bits > 1024n) return null;
  return Math.max(1, Math.ceil(Number(bits) / 8));
};

interface RawSvdRegister extends CustomPeripheralRegister {
  readonly explicitByteWidth: number | null;
}

type SvdRegisterWorkItem =
  | {
      readonly kind: "container";
      readonly element: Element;
      readonly baseOffset: bigint;
      readonly prefix: string;
      readonly inheritedByteWidth: number | null;
      readonly depth: number;
    }
  | {
      readonly kind: "register";
      readonly value: RawSvdRegister;
    };

const parseSvdRegisterTree = (
  registersContainer: Element,
  ctx: {
    readonly limitations: string[];
    readonly budget: SvdExpansionBudget;
  },
): Result<RawSvdRegister[], string> => {
  const results: RawSvdRegister[] = [];
  const stack: SvdRegisterWorkItem[] = [
    {
      kind: "container",
      element: registersContainer,
      baseOffset: 0n,
      prefix: "",
      inheritedByteWidth: null,
      depth: 0,
    },
  ];

  while (stack.length > 0) {
    const work = stack.pop()!;
    if (work.kind === "register") {
      results.push(work.value);
      continue;
    }
    if (work.depth > MAX_SVD_CLUSTER_DEPTH) {
      return err(
        `CMSIS-SVD cluster nesting exceeds maximum depth ${MAX_SVD_CLUSTER_DEPTH}`,
      );
    }

    const emitted: SvdRegisterWorkItem[] = [];
    for (const itemEl of directChildElements(work.element)) {
      const tag = elementTagLower(itemEl);
      if (tag !== "cluster" && tag !== "register") continue;

      const rawName = directChildText(itemEl, "name");
      const offset = parseSvdInt(directChildText(itemEl, "addressoffset"));
      if (rawName === null || offset === null) continue;
      const itemByteWidth =
        parseSvdBitWidthToBytes(directChildText(itemEl, "size")) ??
        work.inheritedByteWidth;

      const dimRaw = directChildText(itemEl, "dim");
      const dimVal = dimRaw === null ? null : parseSvdInt(dimRaw);
      const dimInc = parseSvdInt(directChildText(itemEl, "dimincrement"));
      if (dimRaw !== null) {
        if (
          dimVal === null ||
          dimVal <= 0n ||
          dimVal > BigInt(MAX_SVD_DIM_ELEMENTS) ||
          dimInc === null
        ) {
          ctx.limitations.push(
            `Skipped SVD ${tag} '${rawName}' with unsupported dim/dimIncrement bounds.`,
          );
          continue;
        }
      }

      const count = dimVal === null ? 1 : Number(dimVal);
      ctx.budget.remainingSteps -= count;
      if (ctx.budget.remainingSteps < 0) {
        return err(
          `CMSIS-SVD expanded cluster/register count exceeds maximum ${MAX_SVD_EXPANSION_STEPS} steps`,
        );
      }
      if (tag === "register") {
        ctx.budget.remainingRegisters -= count;
        if (ctx.budget.remainingRegisters < 0) {
          return err(
            `CMSIS-SVD expanded register count exceeds maximum ${MAX_SVD_TOTAL_REGISTERS} registers`,
          );
        }
      }

      const step = dimInc ?? 0n;
      const tokens =
        dimVal === null
          ? [""]
          : expandDimIndices(count, directChildText(itemEl, "dimindex"));
      const rawDesc = directChildText(itemEl, "description");

      for (let idx = 0; idx < count; idx += 1) {
        const token = tokens[idx] ?? String(idx);
        const instanceName =
          dimVal === null ? rawName : formatDimName(rawName, token);
        const instanceOffset = work.baseOffset + offset + BigInt(idx) * step;
        if (tag === "cluster") {
          const cleanPrefix = instanceName.replace(/[^\w]/gu, "_");
          emitted.push({
            kind: "container",
            element: itemEl,
            baseOffset: instanceOffset,
            prefix: `${work.prefix}${cleanPrefix}_`,
            inheritedByteWidth: itemByteWidth,
            depth: work.depth + 1,
          });
        } else {
          const desc = (rawDesc ?? instanceName).replace(/\s+/gu, " ");
          emitted.push({
            kind: "register",
            value: {
              offset: instanceOffset,
              mnemonic: `${work.prefix}${instanceName}`,
              name: desc,
              explicitByteWidth: itemByteWidth,
            },
          });
        }
      }
    }

    for (let idx = emitted.length - 1; idx >= 0; idx -= 1) {
      stack.push(emitted[idx]!);
    }
  }
  return ok(results);
};

interface RawSvdPeripheral {
  readonly name: string;
  readonly derivedFrom: string | null;
  readonly base: bigint;
  readonly declaredSize: number | null;
  readonly declaredByteWidth: number | null;
  readonly group: string | null;
  readonly registers: readonly RawSvdRegister[];
}

interface ResolvedSvdEntry {
  readonly window: CustomPeripheralWindow;
  readonly declaredSize: number | null;
  readonly defaultRegBytes: number;
  readonly rawRegisters: readonly RawSvdRegister[];
}

const buildResolvedSvdEntry = (
  item: RawSvdPeripheral,
  parentEntry: ResolvedSvdEntry | null,
  budget: SvdExpansionBudget,
  deviceDefaultRegBytes: number,
): Result<ResolvedSvdEntry, string> => {
  const parentWindow = parentEntry?.window ?? null;
  const defaultRegBytes =
    item.declaredByteWidth ??
    parentEntry?.defaultRegBytes ??
    deviceDefaultRegBytes;
  const sourceRegs =
    item.registers.length > 0
      ? item.registers
      : (parentEntry?.rawRegisters ?? []);
  if (item.registers.length === 0 && sourceRegs.length > 0) {
    budget.remainingRegisters -= sourceRegs.length;
    if (budget.remainingRegisters < 0) {
      return err(
        `CMSIS-SVD expanded register count exceeds maximum ${MAX_SVD_TOTAL_REGISTERS} registers`,
      );
    }
  }

  const rawRegisters: RawSvdRegister[] = sourceRegs.map((reg) => ({
    offset: reg.offset,
    mnemonic: reg.mnemonic.startsWith(`${item.name}_`)
      ? reg.mnemonic
      : `${item.name}_${reg.mnemonic.replace(
          parentWindow ? new RegExp(`^${parentWindow.name}_`, "u") : /^$/u,
          "",
        )}`,
    name: reg.name.startsWith(`${item.name} `)
      ? reg.name
      : `${item.name} ${reg.name.replace(
          parentWindow ? new RegExp(`^${parentWindow.name} `, "u") : /^$/u,
          "",
        )}`,
    explicitByteWidth: reg.explicitByteWidth,
  }));
  const registers: CustomPeripheralRegister[] = rawRegisters.map((reg) => ({
    offset: reg.offset,
    mnemonic: reg.mnemonic,
    name: reg.name,
  }));

  const maxRegEnd = rawRegisters.reduce((max, reg) => {
    const end = reg.offset + BigInt(reg.explicitByteWidth ?? defaultRegBytes);
    return end > max ? end : max;
  }, 0n);
  if (maxRegEnd > BigInt(Number.MAX_SAFE_INTEGER)) {
    return err(
      `CMSIS-SVD peripheral '${item.name}' register span exceeds safe integer bounds`,
    );
  }
  const inferredSize = maxRegEnd > 0n ? Number(maxRegEnd) : 0x400;
  const declaredSize = item.declaredSize ?? parentEntry?.declaredSize ?? null;
  const sizeBytes = declaredSize ?? inferredSize;
  const window: CustomPeripheralWindow = {
    name: item.name,
    baseAddress: `0x${item.base.toString(16)}`,
    baseAddressValue: item.base,
    sizeBytes,
    bus: item.group ?? parentWindow?.bus ?? null,
    compatible: [],
    status: "okay",
    registers,
  };
  return ok({
    window,
    declaredSize,
    defaultRegBytes,
    rawRegisters,
  });
};

const resolveSvdPeripheral = (
  name: string,
  keyedRaw: ReturnType<typeof keyedBy<RawSvdPeripheral, string>>,
  ctx: {
    readonly memo: Map<string, ResolvedSvdEntry>;
    readonly budget: SvdExpansionBudget;
    readonly deviceDefaultRegBytes: number;
  },
): Result<ResolvedSvdEntry, string> => {
  const existing = ctx.memo.get(name);
  if (existing !== undefined) return ok(existing);

  const chain: RawSvdPeripheral[] = [];
  const visiting = new Set<string>();
  let cursor: string | null = name;

  while (cursor !== null && !ctx.memo.has(cursor)) {
    const currentCursor: string = cursor;
    if (visiting.has(currentCursor)) {
      return err(
        `CMSIS-SVD peripheral derivedFrom cycle involving '${currentCursor}'`,
      );
    }
    const lookup = uniqueKeyed(keyedRaw, currentCursor);
    if (!lookup.ok) {
      return err(
        `CMSIS-SVD contains duplicate peripheral name '${currentCursor}'`,
      );
    }
    const item = lookup.value;
    if (item === undefined) {
      return err(
        `CMSIS-SVD references unknown derivedFrom peripheral '${currentCursor}'`,
      );
    }
    visiting.add(currentCursor);
    chain.push(item);
    cursor = item.derivedFrom;
  }

  for (let idx = chain.length - 1; idx >= 0; idx -= 1) {
    const item = chain[idx]!;
    const parentEntry =
      item.derivedFrom === null
        ? null
        : (ctx.memo.get(item.derivedFrom) ?? null);
    const built = buildResolvedSvdEntry(
      item,
      parentEntry,
      ctx.budget,
      ctx.deviceDefaultRegBytes,
    );
    if (!built.ok) return built;
    ctx.memo.set(item.name, built.value);
  }

  return ok(ctx.memo.get(name)!);
};

/**
 * Parse a CMSIS-SVD XML description into a `CustomHardwareProfile`, resolving
 * peripheral base addresses, `<addressBlock>` spans, chained `derivedFrom`
 * inheritance, nested `<cluster>` offsets, and `<dim>` register arrays.
 */
export const parseCmsisSvd = (
  xml: string,
  options?: { readonly sha256?: string | null },
): Result<CustomHardwareProfile, string> => {
  if (typeof xml !== "string" || xml.trim().length === 0) {
    return err("CMSIS-SVD input must be a non-empty XML string");
  }
  if (xml.length > MAX_SVD_CHARS) {
    return err("CMSIS-SVD input exceeds maximum 16 MiB text size");
  }
  if (countXmlNodes(xml, MAX_SVD_XML_NODES) > MAX_SVD_XML_NODES) {
    return err(
      `CMSIS-SVD XML node count exceeds maximum ${MAX_SVD_XML_NODES} nodes`,
    );
  }
  const parsedDoc = parseXmlDocument(xml, {
    doctype: "external",
    lineEndings: "normalize",
  });
  if (!parsedDoc.ok) {
    return err(`CMSIS-SVD XML is invalid: ${parsedDoc.error.message}`);
  }
  const rootElement = parsedDoc.value.documentElement;
  if (rootElement === null || elementTagLower(rootElement) !== "device") {
    return err("CMSIS-SVD input is missing a <device> root element");
  }

  const deviceName = directChildText(rootElement, "name") ?? "svd-device";
  const deviceDefaultRegBytes =
    parseSvdBitWidthToBytes(directChildText(rootElement, "size")) ?? 4;
  const peripheralsEl = directChildElements(rootElement, "peripherals")[0];
  if (peripheralsEl === undefined) {
    return err("CMSIS-SVD <device> does not contain a <peripherals> section");
  }

  const budget: SvdExpansionBudget = {
    remainingRegisters: MAX_SVD_TOTAL_REGISTERS,
    remainingSteps: MAX_SVD_EXPANSION_STEPS,
  };
  const limitations: string[] = [];
  const rawList: RawSvdPeripheral[] = [];

  for (const periphEl of directChildElements(peripheralsEl, "peripheral")) {
    const name = directChildText(periphEl, "name");
    const base = parseSvdInt(directChildText(periphEl, "baseaddress"));
    if (name === null || base === null) continue;
    const declaredByteWidth = parseSvdBitWidthToBytes(
      directChildText(periphEl, "size"),
    );
    const derivedFrom = elementAttributeCaseInsensitive(
      periphEl,
      "derivedfrom",
    );

    let totalSpan: number | null = null;
    for (const abEl of directChildElements(periphEl, "addressblock")) {
      const blockOffset = parseSvdInt(directChildText(abEl, "offset")) ?? 0n;
      const blockSize = parseSvdInt(directChildText(abEl, "size"));
      if (blockSize !== null && blockSize > 0n) {
        const end = Number(blockOffset + blockSize);
        if (Number.isSafeInteger(end) && end > (totalSpan ?? 0)) {
          totalSpan = end;
        }
      }
    }

    let registers: RawSvdRegister[] = [];
    const regsEl = directChildElements(periphEl, "registers")[0];
    if (regsEl !== undefined) {
      const parsedRegs = parseSvdRegisterTree(regsEl, {
        limitations,
        budget,
      });
      if (!parsedRegs.ok) return parsedRegs;
      registers = parsedRegs.value;
    }

    rawList.push({
      name,
      derivedFrom,
      base,
      declaredSize: totalSpan,
      declaredByteWidth,
      group: directChildText(periphEl, "groupname"),
      registers,
    });
  }

  if (rawList.length === 0) {
    return err("CMSIS-SVD <peripherals> did not define any valid peripheral");
  }

  const keyedRaw = keyedBy(rawList, (item) => item.name);
  if (keyedRaw.duplicates.length > 0) {
    return err(
      `CMSIS-SVD contains duplicate peripheral name '${keyedRaw.duplicates[0]}'`,
    );
  }
  const memo = new Map<string, ResolvedSvdEntry>();
  const peripherals: CustomPeripheralWindow[] = [];
  for (const item of rawList) {
    const resolved = resolveSvdPeripheral(item.name, keyedRaw, {
      memo,
      budget,
      deviceDefaultRegBytes,
    });
    if (!resolved.ok) return resolved;
    peripherals.push(resolved.value.window);
  }

  return ok({
    id: deviceName,
    sourceKind: "cmsis_svd",
    sourceSha256: options?.sha256 ?? null,
    model: deviceName,
    memory: [],
    peripherals,
    devices: [],
    limitations,
  });
};
