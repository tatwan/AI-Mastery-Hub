import { AppLayout } from "@/components/Layout";
import { useGetModule } from "@workspace/api-client-react";
import { useParams, Link } from "wouter";
import { ArrowLeft, Clock, PlayCircle, FileText, Code2, HelpCircle, LayoutTemplate } from "lucide-react";
import { cn, formatMinutes } from "@/lib/utils";

export function ModulePage() {
  const params = useParams();
  const moduleId = params.moduleId || "";
  
  const { data: moduleData, isLoading, error } = useGetModule(moduleId);

  if (isLoading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </AppLayout>
    );
  }

  if (error || !moduleData) {
    return (
      <AppLayout>
        <div className="p-10 text-center">Module not found.</div>
      </AppLayout>
    );
  }

  const getLessonIcon = (type: string) => {
    switch(type) {
      case 'concept': return FileText;
      case 'coding': return Code2;
      case 'quiz': return HelpCircle;
      case 'project': return LayoutTemplate;
      default: return FileText;
    }
  };

  return (
    <AppLayout>
      <div className="max-w-4xl mx-auto px-6 py-10">
        <Link href={`/track/${moduleData.trackId}`} className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8 transition-colors">
          <ArrowLeft className="w-4 h-4" /> Back to Track
        </Link>

        <div className="mb-10 space-y-4">
          <div className="text-primary font-medium text-sm tracking-wider uppercase">Module {moduleData.order}</div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">{moduleData.title}</h1>
          <p className="text-lg text-muted-foreground leading-relaxed max-w-3xl">
            {moduleData.description}
          </p>
        </div>

        <div className="relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-border before:to-transparent">
          <div className="space-y-6 relative z-10">
            {moduleData.lessons.map((lesson, index) => {
              const Icon = getLessonIcon(lesson.type);
              
              return (
                <div key={lesson.id} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-background bg-secondary text-muted-foreground group-hover:bg-primary group-hover:text-primary-foreground transition-colors shadow-sm shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10">
                    <Icon className="w-4 h-4" />
                  </div>
                  
                  <Link href={`/lesson/${lesson.id}`} className="w-[calc(100%-3rem)] md:w-[calc(50%-2.5rem)] cursor-pointer">
                    <div className={cn(
                      "bg-card border border-border rounded-2xl p-5 hover:border-primary/50 hover:shadow-lg transition-all",
                      "group-hover:-translate-y-1"
                    )}>
                      <div className="flex justify-between items-start mb-2">
                        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-2">
                          Lesson {index + 1}
                          <span className="w-1 h-1 rounded-full bg-border" />
                          <span className="text-primary/80 capitalize">{lesson.type}</span>
                        </div>
                        <span className="text-xs font-medium text-muted-foreground flex items-center gap-1 bg-background px-2 py-1 rounded-md border border-border/50">
                          <Clock className="w-3 h-3" /> {formatMinutes(lesson.estimatedMinutes)}
                        </span>
                      </div>
                      
                      <h3 className="text-lg font-bold mb-2 group-hover:text-primary transition-colors">{lesson.title}</h3>
                      <p className="text-sm text-muted-foreground line-clamp-2">{lesson.description}</p>
                      
                      <div className="mt-4 flex items-center text-primary text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity transform translate-y-2 group-hover:translate-y-0">
                        Start Lesson <PlayCircle className="w-4 h-4 ml-1.5" />
                      </div>
                    </div>
                  </Link>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
