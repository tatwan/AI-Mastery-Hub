import { AppLayout } from "@/components/Layout";
import { useGetProgress } from "@workspace/api-client-react";
import { Trophy, Target, Flame, Calendar, Award } from "lucide-react";
import { format } from "date-fns";

export function ProgressPage() {
  const { data: progress, isLoading } = useGetProgress();

  if (isLoading || !progress) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </AppLayout>
    );
  }

  // Generate mock heatmap data for demonstration (since real history isn't in API schema)
  const generateHeatmap = () => {
    const days = [];
    const today = new Date();
    for(let i = 0; i < 90; i++) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      // Random intensity for visual effect
      const intensity = Math.random() > 0.7 ? Math.floor(Math.random() * 4) + 1 : 0;
      days.unshift({ date: d, intensity });
    }
    return days;
  };
  const heatmap = generateHeatmap();

  return (
    <AppLayout>
      <div className="max-w-6xl mx-auto px-6 py-10 space-y-10">
        
        <div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">Your Learning Journey</h1>
          <p className="text-muted-foreground">Track your academic progress, experience points, and recent activities.</p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-card border border-border p-6 rounded-2xl flex flex-col gap-4 relative overflow-hidden">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-primary/5 rounded-full blur-xl" />
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-primary/10 text-primary">
                <Trophy className="w-5 h-5" />
              </div>
              <div className="font-medium text-muted-foreground">Total XP</div>
            </div>
            <div className="text-4xl font-black">{progress.totalXp.toLocaleString()}</div>
          </div>

          <div className="bg-card border border-border p-6 rounded-2xl flex flex-col gap-4 relative overflow-hidden">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-orange-500/5 rounded-full blur-xl" />
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-orange-500/10 text-orange-500">
                <Flame className="w-5 h-5" />
              </div>
              <div className="font-medium text-muted-foreground">Day Streak</div>
            </div>
            <div className="text-4xl font-black">{progress.streakDays}</div>
          </div>

          <div className="bg-card border border-border p-6 rounded-2xl flex flex-col gap-4 relative overflow-hidden">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-blue-500/5 rounded-full blur-xl" />
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-blue-500/10 text-blue-500">
                <Target className="w-5 h-5" />
              </div>
              <div className="font-medium text-muted-foreground">Lessons Completed</div>
            </div>
            <div className="text-4xl font-black">{progress.completedLessons.length}</div>
          </div>

          <div className="bg-card border border-border p-6 rounded-2xl flex flex-col gap-4 relative overflow-hidden">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-green-500/5 rounded-full blur-xl" />
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-green-500/10 text-green-500">
                <Award className="w-5 h-5" />
              </div>
              <div className="font-medium text-muted-foreground">Exercises Solved</div>
            </div>
            <div className="text-4xl font-black">
              {progress.exerciseResults.filter(r => r.isCorrect).length}
            </div>
          </div>
        </div>

        {/* Activity Heatmap */}
        <div className="bg-card border border-border p-8 rounded-2xl">
          <div className="flex items-center gap-3 mb-6">
            <Calendar className="w-5 h-5 text-muted-foreground" />
            <h2 className="text-xl font-bold">Activity Map</h2>
          </div>
          
          <div className="flex gap-2">
            {/* Simple CSS grid heatmap representation */}
            <div className="grid grid-rows-7 grid-flow-col gap-1.5 w-full overflow-x-auto pb-4">
              {heatmap.map((day, i) => (
                <div 
                  key={i}
                  title={`${format(day.date, 'MMM d, yyyy')}`}
                  className={`w-3.5 h-3.5 rounded-sm transition-colors cursor-pointer hover:ring-2 hover:ring-primary/50 ring-offset-1 ring-offset-card
                    ${day.intensity === 0 ? 'bg-secondary' : 
                      day.intensity === 1 ? 'bg-primary/30' : 
                      day.intensity === 2 ? 'bg-primary/50' : 
                      day.intensity === 3 ? 'bg-primary/80' : 
                      'bg-primary'}
                  `}
                />
              ))}
            </div>
          </div>
          <div className="flex items-center justify-end gap-2 text-xs text-muted-foreground mt-2">
            <span>Less</span>
            <div className="flex gap-1">
              <div className="w-3 h-3 rounded-sm bg-secondary" />
              <div className="w-3 h-3 rounded-sm bg-primary/30" />
              <div className="w-3 h-3 rounded-sm bg-primary/50" />
              <div className="w-3 h-3 rounded-sm bg-primary/80" />
              <div className="w-3 h-3 rounded-sm bg-primary" />
            </div>
            <span>More</span>
          </div>
        </div>

        {/* Recent Achievements / Exercises */}
        <div className="bg-card border border-border rounded-2xl overflow-hidden">
          <div className="p-6 border-b border-border bg-muted/20">
            <h2 className="text-xl font-bold">Recent Exercise Results</h2>
          </div>
          <div className="divide-y divide-border">
            {progress.exerciseResults.length === 0 ? (
              <div className="p-8 text-center text-muted-foreground">No exercises completed yet.</div>
            ) : (
              progress.exerciseResults.slice(0, 5).map((result, idx) => (
                <div key={idx} className="p-6 flex items-center justify-between hover:bg-muted/10 transition-colors">
                  <div className="flex items-start gap-4">
                    <div className={`mt-1 p-2 rounded-full ${result.isCorrect ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                      {result.isCorrect ? <CheckCircle2 className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
                    </div>
                    <div>
                      <div className="font-semibold text-foreground mb-1">Exercise: {result.exerciseId}</div>
                      <div className="text-sm text-muted-foreground line-clamp-1">{result.explanation}</div>
                      {result.submittedAt && (
                        <div className="text-xs text-muted-foreground/60 mt-2">
                          {format(new Date(result.submittedAt), 'MMM d, yyyy • h:mm a')}
                        </div>
                      )}
                    </div>
                  </div>
                  {result.isCorrect && (
                    <div className="text-right font-bold text-primary shrink-0">
                      +{result.xpEarned} XP
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </AppLayout>
  );
}

// Ensure XCircle is imported for the progress page since it was missing in the top block
import { XCircle } from "lucide-react";
