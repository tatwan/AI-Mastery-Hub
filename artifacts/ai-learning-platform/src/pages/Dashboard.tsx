import { AppLayout } from "@/components/Layout";
import { useGetTracks, useGetProgress } from "@workspace/api-client-react";
import { ProgressRing } from "@/components/ProgressRing";
import { Link } from "wouter";
import { BookOpen, Clock, Zap, Target, ArrowRight, Flame } from "lucide-react";
import { cn } from "@/lib/utils";

export function Dashboard() {
  const { data: tracks, isLoading: tracksLoading } = useGetTracks();
  const { data: progress, isLoading: progressLoading } = useGetProgress();

  if (tracksLoading || progressLoading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </AppLayout>
    );
  }

  const completedCount = progress?.completedLessons.length || 0;
  const totalXP = progress?.totalXp || 0;
  const streak = progress?.streakDays || 0;

  return (
    <AppLayout>
      <div className="max-w-6xl mx-auto px-6 py-10 space-y-12">
        {/* Hero Section */}
        <section className="relative rounded-3xl overflow-hidden bg-card border border-border shadow-2xl">
          <div className="absolute inset-0">
            <img 
              src={`${import.meta.env.BASE_URL}images/dashboard-hero.png`} 
              alt="Neural Network Background" 
              className="w-full h-full object-cover opacity-20"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-card via-card/80 to-transparent" />
            <div className="absolute inset-0 bg-gradient-to-r from-card via-card/50 to-transparent" />
          </div>
          
          <div className="relative p-10 md:p-12 z-10 flex flex-col md:flex-row gap-8 justify-between items-center">
            <div className="space-y-4 max-w-xl">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-medium uppercase tracking-wider">
                <Zap className="w-3.5 h-3.5" /> Post-Graduate Curriculum
              </div>
              <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-white leading-tight">
                Master Advanced AI & Machine Learning.
              </h1>
              <p className="text-lg text-muted-foreground leading-relaxed">
                Dive deep into the mathematics and code behind modern architectures. From optimization theory to diffusion models.
              </p>
            </div>

            {/* Stats Widget */}
            <div className="bg-background/80 backdrop-blur-xl border border-border rounded-2xl p-6 flex flex-col gap-6 min-w-[280px] shadow-xl">
              <div className="flex justify-between items-end border-b border-border/50 pb-4">
                <div>
                  <p className="text-sm text-muted-foreground font-medium mb-1">Total Experience</p>
                  <div className="text-3xl font-bold text-white tracking-tight flex items-center gap-2">
                    {totalXP.toLocaleString()} <span className="text-primary text-xl">XP</span>
                  </div>
                </div>
                <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                  <Target className="w-6 h-6 text-primary" />
                </div>
              </div>
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-orange-500/10 flex items-center justify-center">
                    <Flame className="w-5 h-5 text-orange-500" />
                  </div>
                  <div>
                    <div className="text-lg font-bold text-white">{streak} Days</div>
                    <div className="text-xs text-muted-foreground">Current Streak</div>
                  </div>
                </div>
                <div className="h-10 w-px bg-border/50 mx-2" />
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-blue-500/10 flex items-center justify-center">
                    <BookOpen className="w-5 h-5 text-blue-500" />
                  </div>
                  <div>
                    <div className="text-lg font-bold text-white">{completedCount}</div>
                    <div className="text-xs text-muted-foreground">Lessons</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Tracks Grid */}
        <section className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold tracking-tight">Learning Tracks</h2>
              <p className="text-muted-foreground mt-1">Structured paths from foundations to state-of-the-art.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {tracks?.map((track, i) => {
              // Mocking track progress since we don't have aggregated track progress in API
              const mockProgress = Math.min(100, Math.floor((completedCount / (track.lessonCount || 10)) * 100));
              
              return (
                <Link key={track.id} href={`/track/${track.id}`}>
                  <div className="group h-full bg-card rounded-2xl border border-border p-6 hover:shadow-2xl hover:shadow-primary/5 hover:border-primary/30 transition-all duration-300 flex flex-col relative overflow-hidden cursor-pointer">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    
                    <div className="flex justify-between items-start mb-6">
                      <div className="w-14 h-14 rounded-xl bg-secondary flex items-center justify-center text-2xl shadow-inner border border-border/50">
                        {track.icon || "📚"}
                      </div>
                      <ProgressRing progress={mockProgress} size={52} strokeWidth={4} />
                    </div>
                    
                    <div className="flex-1 space-y-3">
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          "text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full",
                          track.difficulty === 'beginner' ? "bg-green-500/10 text-green-400" :
                          track.difficulty === 'intermediate' ? "bg-blue-500/10 text-blue-400" :
                          track.difficulty === 'advanced' ? "bg-purple-500/10 text-purple-400" :
                          "bg-red-500/10 text-red-400"
                        )}>
                          {track.difficulty}
                        </span>
                        <span className="text-xs text-muted-foreground flex items-center gap-1">
                          <Clock className="w-3 h-3" /> {track.estimatedHours}h
                        </span>
                      </div>
                      
                      <h3 className="text-xl font-bold group-hover:text-primary transition-colors line-clamp-2">
                        {track.title}
                      </h3>
                      <p className="text-sm text-muted-foreground line-clamp-2 leading-relaxed">
                        {track.description}
                      </p>
                    </div>

                    <div className="mt-6 pt-6 border-t border-border/50 flex items-center justify-between text-sm">
                      <div className="flex items-center gap-4 text-muted-foreground">
                        <span className="flex items-center gap-1.5"><BookOpen className="w-4 h-4"/> {track.moduleCount} modules</span>
                      </div>
                      <span className="text-primary font-medium flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                        Explore <ArrowRight className="w-4 h-4" />
                      </span>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      </div>
    </AppLayout>
  );
}
