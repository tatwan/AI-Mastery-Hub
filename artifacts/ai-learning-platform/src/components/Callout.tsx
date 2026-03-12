import { Info, AlertTriangle, Lightbulb, AlertOctagon } from "lucide-react";
import { cn } from "@/lib/utils";

interface CalloutProps {
  type: "info" | "warning" | "tip" | "important";
  title?: string;
  children: React.ReactNode;
}

const config = {
  info: {
    icon: Info,
    classes: "bg-blue-500/10 border-blue-500/30 text-blue-200",
    iconClass: "text-blue-400",
    title: "Information"
  },
  warning: {
    icon: AlertTriangle,
    classes: "bg-amber-500/10 border-amber-500/30 text-amber-200",
    iconClass: "text-amber-400",
    title: "Warning"
  },
  tip: {
    icon: Lightbulb,
    classes: "bg-emerald-500/10 border-emerald-500/30 text-emerald-200",
    iconClass: "text-emerald-400",
    title: "Pro Tip"
  },
  important: {
    icon: AlertOctagon,
    classes: "bg-rose-500/10 border-rose-500/30 text-rose-200",
    iconClass: "text-rose-400",
    title: "Crucial Concept"
  }
};

export function Callout({ type, title, children }: CalloutProps) {
  const style = config[type];
  const Icon = style.icon;

  return (
    <div className={cn("my-6 p-4 rounded-xl border flex gap-4 backdrop-blur-sm", style.classes)}>
      <div className="mt-0.5 shrink-0">
        <Icon className={cn("w-5 h-5", style.iconClass)} />
      </div>
      <div className="space-y-1">
        <h5 className={cn("font-semibold leading-none mb-2", style.iconClass)}>
          {title || style.title}
        </h5>
        <div className="text-sm leading-relaxed opacity-90 prose-academic">
          {children}
        </div>
      </div>
    </div>
  );
}
