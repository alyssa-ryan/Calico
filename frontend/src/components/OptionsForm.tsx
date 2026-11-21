import { type Field, numberFromInput, type Values } from "../lib/options";

type Props = {
  fields: readonly Field[];
  values: Values;
  errors: Record<string, string>;
  onChange: (key: string, value: unknown) => void;
};

/** Task options, generated from the task's schema in the order the task declares them. */
export function OptionsForm({ fields, values, errors, onChange }: Props) {
  return (
    <div className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
      {fields.map((field) => (
        <div key={field.key} className="min-w-0 space-y-2" data-option={field.key}>
          <FieldControl
            field={field}
            value={values[field.key]}
            onChange={(v) => onChange(field.key, v)}
          />
          {field.description && (
            <p className="text-xs leading-snug text-zinc-500 dark:text-zinc-400">
              {field.description}
            </p>
          )}
          {errors[field.key] && (
            <p role="alert" className="text-xs font-medium text-red-600 dark:text-red-400">
              {errors[field.key]}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

const inputClass =
  "w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 dark:border-zinc-700 dark:bg-zinc-900";

function FieldControl({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const id = `opt-${field.key}`;
  const label = (
    <label htmlFor={id} className="block text-sm font-medium">
      {field.title}
    </label>
  );
  switch (field.kind) {
    case "range": {
      const n = typeof value === "number" ? value : field.default;
      return (
        <>
          <div className="flex items-baseline justify-between gap-2">
            {label}
            <span className="font-mono text-xs text-zinc-500 tabular-nums">{n.toFixed(2)}</span>
          </div>
          <input
            id={id}
            type="range"
            min={field.min}
            max={field.max}
            step={field.step}
            value={n}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full accent-teal-600"
          />
        </>
      );
    }
    case "number":
      return (
        <>
          {label}
          <input
            id={id}
            type="number"
            min={field.min}
            max={field.max}
            step={field.step}
            value={typeof value === "number" && Number.isFinite(value) ? value : ""}
            onChange={(e) =>
              onChange(numberFromInput(field, e.target.value, e.target.valueAsNumber))
            }
            className={`${inputClass} max-w-40`}
          />
        </>
      );
    case "boolean": {
      const on = value === true;
      return (
        <div className="flex items-center justify-between gap-3">
          {label}
          <button
            id={id}
            type="button"
            role="switch"
            aria-checked={on}
            onClick={() => onChange(!on)}
            className={`relative h-6 w-11 shrink-0 rounded-full transition ${
              on ? "bg-accent-600" : "bg-zinc-300 dark:bg-zinc-700"
            } focus-visible:outline-2 focus-visible:outline-accent-500`}
          >
            <span
              className={`absolute top-0.5 left-0.5 size-5 rounded-full bg-white shadow transition-transform ${
                on ? "translate-x-5" : ""
              }`}
            />
          </button>
        </div>
      );
    }
    case "enum":
      return (
        <fieldset>
          <legend className="mb-2 text-sm font-medium">{field.title}</legend>
          <div className="inline-flex flex-wrap rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
            {field.choices.map((c) => (
              <label
                key={c.value}
                className={`cursor-pointer rounded-md px-3 py-1 text-sm has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent-500 ${
                  value === c.value ? "bg-white shadow-sm dark:bg-zinc-950" : "text-zinc-500"
                }`}
              >
                <input
                  type="radio"
                  name={id}
                  value={c.value}
                  checked={value === c.value}
                  onChange={() => onChange(c.value)}
                  className="sr-only"
                />
                {c.label}
              </label>
            ))}
          </div>
        </fieldset>
      );
    case "multi": {
      const selected = Array.isArray(value) ? value.map(String) : [];
      return (
        <fieldset>
          <legend className="mb-2 text-sm font-medium">{field.title}</legend>
          <div className="flex flex-wrap gap-2">
            {field.choices.map((c) => {
              const on = selected.includes(c.value);
              return (
                <label
                  key={c.value}
                  className={`cursor-pointer rounded-full border px-3 py-1 text-sm transition has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent-500 ${
                    on
                      ? "border-accent-500 bg-accent-50 text-accent-700 dark:bg-accent-500/15 dark:text-accent-400"
                      : "border-zinc-300 text-zinc-600 hover:border-zinc-400 dark:border-zinc-700 dark:text-zinc-400"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() =>
                      onChange(on ? selected.filter((s) => s !== c.value) : [...selected, c.value])
                    }
                    className="sr-only"
                  />
                  {c.label}
                </label>
              );
            })}
          </div>
        </fieldset>
      );
    }
    case "text":
      return (
        <>
          {label}
          <input
            id={id}
            type="text"
            value={typeof value === "string" ? value : ""}
            onChange={(e) => onChange(e.target.value)}
            className={inputClass}
          />
        </>
      );
  }
}
