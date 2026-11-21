import type { Template } from "../api/types";
import { TEMPLATE } from "../lib/format";

type Props = { value: Template; trained: boolean; onChange: (template: Template) => void };

/** General or Yours: whose taste ranks photos. Yours waits until it is trained. */
export function TasteChoice({ value, trained, onChange }: Props) {
  const options: { name: Template; title: string; help: string; disabled: boolean }[] = [
    { name: "general", title: TEMPLATE.general, help: TEMPLATE.generalHelp, disabled: false },
    {
      name: "yours",
      title: TEMPLATE.yours,
      help: trained ? TEMPLATE.yoursHelp : TEMPLATE.untrained,
      disabled: !trained,
    },
  ];
  return (
    <div role="radiogroup" aria-labelledby="taste-legend" className="mt-4 space-y-2">
      <p id="taste-legend" className="text-sm font-medium">
        {TEMPLATE.legend}
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        {options.map((option) => (
          <label
            key={option.name}
            className={`flex items-start gap-3 rounded-xl border p-3 text-sm has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent-500 ${
              option.disabled
                ? "cursor-not-allowed border-zinc-200 text-zinc-500 dark:border-zinc-800"
                : value === option.name
                  ? "cursor-pointer border-accent-500 bg-accent-50/60 dark:bg-accent-500/10"
                  : "cursor-pointer border-zinc-200 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900"
            }`}
          >
            <input
              type="radio"
              name="template"
              value={option.name}
              checked={value === option.name}
              disabled={option.disabled}
              onChange={() => onChange(option.name)}
              className="mt-0.5 size-4 shrink-0 accent-accent-600"
            />
            <span>
              <span className="font-medium">{option.title}</span>
              <span className="block text-zinc-500 dark:text-zinc-400">{option.help}</span>
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
