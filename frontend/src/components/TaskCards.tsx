import { GalleryHorizontal, History, type LucideIcon, Repeat, Sparkles, Wand2 } from "lucide-react";
import type { TaskInfo } from "../api/types";

const ICONS: Record<string, LucideIcon> = {
  best: Sparkles,
  same_view: History,
  recurring_views: Repeat,
  triptych: GalleryHorizontal,
};

type Props = { tasks: readonly TaskInfo[]; value: string; onChange: (task: string) => void };

/** The option menu: one card per registered task (new tasks appear automatically). */
export function TaskCards({ tasks, value, onChange }: Props) {
  return (
    <div role="radiogroup" aria-label="What do you need?" className="grid gap-3 sm:grid-cols-2">
      {tasks.map((task) => {
        const Icon = ICONS[task.name] ?? Wand2;
        const selected = task.name === value;
        return (
          <label
            key={task.name}
            data-testid="task-card"
            className={`group relative flex cursor-pointer flex-col gap-2 rounded-xl border p-4 transition has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent-500 ${
              selected
                ? "border-accent-500 bg-accent-50/60 shadow-sm dark:border-accent-500 dark:bg-accent-500/10"
                : "border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:border-zinc-700 dark:hover:bg-zinc-900"
            }`}
          >
            <input
              type="radio"
              name="task"
              value={task.name}
              checked={selected}
              onChange={() => onChange(task.name)}
              className="sr-only"
            />
            <Icon
              className={`size-5 ${selected ? "text-accent-600 dark:text-accent-400" : "text-zinc-500"}`}
              aria-hidden
            />
            <span className="font-medium">{task.title}</span>
            <span className="text-sm leading-snug text-zinc-500 dark:text-zinc-400">
              {task.description}
            </span>
          </label>
        );
      })}
    </div>
  );
}
