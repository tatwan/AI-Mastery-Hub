import { AppLayout } from "@/components/Layout";
import { useGetTrack } from "@workspace/api-client-react";
import { useParams, Link } from "wouter";
import { BookOpen, Clock, ArrowLeft, Layers, PlayCircle } from "lucide-react";
import { ProgressRing } from "@/components/ProgressRing";

export function TrackPage() {
  const params = useParams();
  const trackId = params.trackId || "";
  
  const { data: track, isLoading, error } = useGetTrack(trackId);

  if (isLoading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </AppLayout>
    );
  }

  if (error || !track) {
    return (
      <AppLayout>
        <div className="p-10 text-center">Track not found.</div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="max-w-5xl mx-auto px-6 py-10">
        <Link href="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8 transition-colors">
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </Link>

        {/* Track Header */}
        <div className="bg-card border border-border rounded-3xl p-8 md:p-10 mb-10 shadow-xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-96 h-96 bg-primary/5 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/3 pointer-events-none" />
          
          <div className="flex flex-col md:flex-row gap-8 items-start justify-between relative z-10">
            <div className="space-y-4 max-w-2xl">
              <div className="w-16 h-16 rounded-2xl bg-secondary flex items-center justify-center text-3xl shadow-inner border border-border/50 mb-6">
                {track.icon || "🔬"}
              </div>
              <h1 className="text-3xl md:text-5xl font-bold tracking-tight text-foreground">
                {track.title}
              </h1>
              <p className="text-lg text-muted-foreground leading-relaxed">
                {track.description}
              </p>
              
              <div className="flex flex-wrap items-center gap-4 pt-4">
                <div className="flex items-center gap-2 text-sm font-medium px-3 py-1.5 rounded-lg bg-secondary text-secondary-foreground">
                  <Layers className="w-4 h-4 text-primary" />
                  {track.moduleCount} Modules
                </div>
                <div className="flex items-center gap-2 text-sm font-medium px-3 py-1.5 rounded-lg bg-secondary text-secondary-foreground">
                  <Clock className="w-4 h-4 text-primary" />
                  ~{track.estimatedHours} Hours
                </div>
                <div className="flex flex-wrap gap-2 ml-4">
                  {track.tags.map(tag => (
                    <span key={tag} className="text-xs font-medium text-muted-foreground bg-background px-2.5 py-1 rounded-full border border-border">
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="bg-background border border-border p-6 rounded-2xl flex flex-col items-center justify-center min-w-[200px] shrink-0 shadow-lg">
              <ProgressRing progress={0} size={100} strokeWidth={8} className="mb-4" />
              <div className="text-center">
                <div className="text-sm text-muted-foreground">Track Completion</div>
                <div className="font-semibold mt-1">0 / {track.lessonCount} Lessons</div>
              </div>
            </div>
          </div>
        </div>

        {/* Modules List */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold">Curriculum Modules</h2>
          
          <div className="space-y-4">
            {track.modules.map((module, index) => (
              <Link key={module.id} href={`/module/${module.id}`}>
                <div className="group bg-card border border-border rounded-2xl p-6 hover:border-primary/50 hover:bg-muted/30 transition-all cursor-pointer flex gap-6 items-center">
                  <div className="w-12 h-12 rounded-xl bg-background border border-border flex items-center justify-center font-bold text-xl text-muted-foreground group-hover:text-primary group-hover:border-primary/30 transition-colors shrink-0">
                    {index + 1}
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="text-xl font-bold mb-2 group-hover:text-primary transition-colors">{module.title}</h3>
                    <p className="text-muted-foreground text-sm leading-relaxed mb-3 line-clamp-2">
                      {module.description}
                    </p>
                    <div className="flex items-center gap-4 text-xs font-medium text-muted-foreground">
                      <span className="flex items-center gap-1.5"><BookOpen className="w-3.5 h-3.5" /> {module.lessonCount} Lessons</span>
                      <span className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> {module.estimatedHours}h</span>
                    </div>
                  </div>

                  <div className="shrink-0 p-4 rounded-full bg-primary/5 text-primary opacity-0 group-hover:opacity-100 transition-all transform translate-x-4 group-hover:translate-x-0 hidden md:flex">
                    <PlayCircle className="w-6 h-6" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
