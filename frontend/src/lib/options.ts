/** Turn a task's pydantic JSON schema into form fields, defaults and a minimal request. */

export type Choice = { value: string; label: string };

type Base = { key: string; title: string; description: string };
export type Field = Base &
  (
    | { kind: "range"; min: number; max: number; step: number; default: number }
    | {
        kind: "number";
        integer: boolean;
        min?: number;
        max?: number;
        /** Exclusive bounds (pydantic's gt / lt): the value must be above / below them. */
        above?: number;
        below?: number;
        /** "any" for decimals: a fixed step would make the browser refuse 1.5. */
        step: number | "any";
        default: number | null;
        /** The task takes no value (null) as "its own default", so empty is allowed. */
        optional: boolean;
      }
    | { kind: "boolean"; default: boolean }
    | { kind: "enum"; choices: Choice[]; default: string | null }
    | { kind: "multi"; choices: Choice[]; minItems: number; default: string[] }
    | { kind: "text"; default: string | null }
  );

export type Values = Record<string, unknown>;
type Json = Record<string, unknown>;

const isObject = (v: unknown): v is Json =>
  typeof v === "object" && v !== null && !Array.isArray(v);

export function humanize(key: string): string {
  const text = key.replace(/_/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function resolve(node: unknown, defs: Json): Json {
  if (!isObject(node)) return {};
  const ref = node.$ref;
  if (typeof ref === "string") {
    const target = defs[ref.split("/").pop() ?? ""];
    return { ...resolve(target, defs), ...node, $ref: undefined };
  }
  return node;
}

/** `string | null` arrives as anyOf [{type: string}, {type: null}]; unwrap to the non-null part. */
function unwrapNullable(node: Json): { node: Json; nullable: boolean } {
  const any = node.anyOf;
  if (!Array.isArray(any)) return { node, nullable: false };
  const rest = any.filter((a) => !(isObject(a) && a.type === "null"));
  const first = rest[0];
  if (rest.length === 1 && isObject(first)) return { node: { ...node, ...first }, nullable: true };
  return { node, nullable: false };
}

function choicesOf(enumValues: unknown): Choice[] {
  return Array.isArray(enumValues)
    ? enumValues.map((v) => ({ value: String(v), label: humanize(String(v)) }))
    : [];
}

function parseField(key: string, raw: unknown, defs: Json): Field | undefined {
  const { node, nullable } = unwrapNullable(resolve(raw, defs));
  const base: Base = {
    key,
    title: typeof node.title === "string" ? node.title : humanize(key),
    description: typeof node.description === "string" ? node.description : "",
  };
  const def = node.default;
  const min = typeof node.minimum === "number" ? node.minimum : undefined;
  const max = typeof node.maximum === "number" ? node.maximum : undefined;
  const above = typeof node.exclusiveMinimum === "number" ? node.exclusiveMinimum : undefined;
  const below = typeof node.exclusiveMaximum === "number" ? node.exclusiveMaximum : undefined;

  if (node.type === "boolean") return { ...base, kind: "boolean", default: def === true };

  if (node.type === "number" || node.type === "integer") {
    const integer = node.type === "integer";
    const value = typeof def === "number" ? def : null;
    if (!integer && min === 0 && max === 1 && value !== null) {
      return { ...base, kind: "range", min, max, step: 0.01, default: value };
    }
    return {
      ...base,
      kind: "number",
      integer,
      ...(min !== undefined && { min }),
      ...(max !== undefined && { max }),
      ...(above !== undefined && { above }),
      ...(below !== undefined && { below }),
      step: integer ? 1 : "any",
      default: value,
      optional: nullable,
    };
  }

  if (Array.isArray(node.enum)) {
    return {
      ...base,
      kind: "enum",
      choices: choicesOf(node.enum),
      default: typeof def === "string" ? def : null,
    };
  }

  if (node.type === "array") {
    const items = resolve(node.items, defs);
    if (Array.isArray(items.enum)) {
      return {
        ...base,
        kind: "multi",
        choices: choicesOf(items.enum),
        minItems: typeof node.minItems === "number" ? node.minItems : 0,
        default: Array.isArray(def) ? def.map(String) : [],
      };
    }
    return undefined;
  }

  if (node.type === "string") {
    return { ...base, kind: "text", default: typeof def === "string" ? def : null };
  }
  return undefined;
}

/** Fields in the schema's declaration order. Unsupported shapes are skipped. */
export function parseOptionsSchema(schema: unknown): Field[] {
  if (!isObject(schema) || !isObject(schema.properties)) return [];
  const defs = isObject(schema.$defs) ? schema.$defs : {};
  const fields: Field[] = [];
  for (const [key, raw] of Object.entries(schema.properties)) {
    const field = parseField(key, raw, defs);
    if (field) fields.push(field);
  }
  return fields;
}

export function defaultValues(fields: readonly Field[]): Values {
  return Object.fromEntries(
    fields.map((f) => [f.key, f.kind === "text" ? (f.default ?? "") : f.default]),
  );
}

const sameList = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && [...a].sort().join("\0") === [...b].sort().join("\0");

function isDefault(field: Field, value: unknown): boolean {
  switch (field.kind) {
    case "multi":
      return Array.isArray(value) && sameList(value.map(String), field.default);
    case "text":
      return (typeof value === "string" ? value.trim() : "") === (field.default ?? "");
    default:
      return value === field.default;
  }
}

/** A number input's value: cleared, an optional number is null (the task's default) and
 * a required one is NaN, which `validateOptions` refuses. */
export function numberFromInput(
  field: Extract<Field, { kind: "number" }>,
  text: string,
  valueAsNumber: number,
): number | null {
  if (text === "") return field.optional ? null : Number.NaN;
  return valueAsNumber;
}

/** Per-field problems that should block submitting. */
export function validateOptions(fields: readonly Field[], values: Values): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const f of fields) {
    const value = values[f.key];
    if (f.kind === "multi") {
      const count = Array.isArray(value) ? value.length : 0;
      if (count < f.minItems) {
        errors[f.key] = f.minItems === 1 ? "Pick at least one." : `Pick at least ${f.minItems}.`;
      }
    } else if (f.kind === "number" || f.kind === "range") {
      const min = f.min;
      const max = f.max;
      if (f.kind === "number" && f.optional && value === null) {
        continue;
      }
      if (typeof value !== "number" || !Number.isFinite(value)) {
        errors[f.key] = "Enter a number.";
      } else if (min !== undefined && value < min) {
        errors[f.key] = `At least ${min}.`;
      } else if (max !== undefined && value > max) {
        errors[f.key] = `At most ${max}.`;
      } else if (f.kind === "number" && f.above !== undefined && value <= f.above) {
        errors[f.key] = `More than ${f.above}.`;
      } else if (f.kind === "number" && f.below !== undefined && value >= f.below) {
        errors[f.key] = `Less than ${f.below}.`;
      } else if (f.kind === "number" && f.integer && !Number.isInteger(value)) {
        errors[f.key] = "Whole numbers only.";
      }
    }
  }
  return errors;
}

/** Only values that differ from the task's defaults; empty text becomes null. */
export function changedOptions(fields: readonly Field[], values: Values): Values {
  const out: Values = {};
  for (const f of fields) {
    const value = values[f.key];
    if (isDefault(f, value)) continue;
    if (f.kind === "text")
      out[f.key] = typeof value === "string" && value.trim() ? value.trim() : null;
    else out[f.key] = value;
  }
  return out;
}
