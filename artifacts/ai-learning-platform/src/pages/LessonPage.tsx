import { useState, useEffect } from "react";
import { useParams, Link } from "wouter";
import { useGetLesson, useUpdateLessonProgress } from "@workspace/api-client-react";
import { MathRenderer } from "@/components/MathRenderer";
import { CodeBlock } from "@/components/CodeBlock";
import { Callout } from "@/components/Callout";
import { ExercisePanel } from "@/components/ExercisePanel";
import { Menu, X, ChevronLeft, ChevronRight, CheckCircle2, ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";

export function LessonPage() {
  const params = useParams();
  const lessonId = params.lessonId || "";
  
  const { data: lesson, isLoading } = useGetLesson(lessonId);
  const updateProgress = useUpdateLessonProgress();
  
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeExerciseIdx, setActiveExerciseIdx] = useState(0);
  const [completed, setCompleted] = useState(false);

  // Auto-collapse sidebar on smaller screens
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) setSidebarOpen(false);
      else setSidebarOpen(true);
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  if (isLoading || !lesson) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  const handleCompleteLesson = () => {
    updateProgress.mutate({
      lessonId: lesson.id,
      data: { status: "completed", timeSpentMinutes: 30 }
    });
    setCompleted(true);
  };

  const hasExercises = lesson.exercises && lesson.exercises.length > 0;
  const currentExercise = hasExercises ? lesson.exercises[activeExerciseIdx] : null;

  return (
    <div className="h-screen bg-background text-foreground flex overflow-hidden selection:bg-primary/30 selection:text-primary-foreground">
      
      {/* Sidebar TOC */}
      <aside className={cn(
        "fixed lg:static inset-y-0 left-0 z-40 w-72 bg-card border-r border-border transition-transform duration-300 flex flex-col",
        sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0 lg:w-0 lg:border-none"
      )}>
        <div className="p-4 border-b border-border flex items-center justify-between">
          <Link href={`/module/${lesson.moduleId}`} className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors font-medium">
            <ArrowLeft className="w-4 h-4" /> Back to Module
          </Link>
          <button className="lg:hidden p-2" onClick={() => setSidebarOpen(false)}>
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="p-5 flex-1 overflow-y-auto">
          <div className="text-xs font-bold text-primary uppercase tracking-wider mb-2">Lesson Outline</div>
          <h2 className="font-bold text-lg mb-6 leading-tight">{lesson.title}</h2>
          
          <nav className="space-y-1 relative border-l border-border/50 ml-2">
            {lesson.sections.map((section, idx) => {
              if (!section.title) return null;
              return (
                <a 
                  key={section.id} 
                  href={`#section-${section.id}`}
                  className="block py-2 pl-4 pr-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-r-lg border-l-2 border-transparent hover:border-primary/50 transition-colors"
                >
                  {section.title}
                </a>
              );
            })}
            {hasExercises && (
              <a 
                href="#exercises"
                className="block py-2 pl-4 pr-2 text-sm font-medium text-primary hover:bg-primary/5 rounded-r-lg border-l-2 border-primary transition-colors mt-4"
              >
                Interactive Exercises
              </a>
            )}
          </nav>

          {lesson.prerequisites?.length > 0 && (
            <div className="mt-8 pt-6 border-t border-border/50">
              <h4 className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-3">Prerequisites</h4>
              <ul className="space-y-2">
                {lesson.prerequisites.map((req, i) => (
                  <li key={i} className="text-sm flex items-start gap-2 text-muted-foreground">
                    <div className="w-1.5 h-1.5 rounded-full bg-border mt-1.5 shrink-0" />
                    <span>{req}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-full relative">
        {/* Top Navbar */}
        <header className="h-14 border-b border-border bg-background/80 backdrop-blur flex items-center justify-between px-4 shrink-0 z-30">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button className="p-2 hover:bg-muted rounded-md text-muted-foreground" onClick={() => setSidebarOpen(true)}>
                <Menu className="w-5 h-5" />
              </button>
            )}
            <div className="text-sm font-medium text-muted-foreground hidden sm:block">
              {lesson.trackId} <span className="mx-2 text-border">/</span> {lesson.title}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={handleCompleteLesson}
              disabled={completed}
              className={cn(
                "flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-semibold transition-all",
                completed 
                  ? "bg-green-500/10 text-green-500 cursor-default" 
                  : "bg-primary text-primary-foreground hover:bg-primary/90"
              )}
            >
              {completed ? <><CheckCircle2 className="w-4 h-4" /> Completed</> : "Mark Complete"}
            </button>
          </div>
        </header>

        {/* Reader Layout Split */}
        <div className="flex-1 flex overflow-hidden">
          
          {/* Article / Reading pane */}
          <div className={cn(
            "flex-1 overflow-y-auto scroll-smooth",
            hasExercises ? "border-r border-border max-w-4xl" : "w-full"
          )}>
            <article className="max-w-3xl mx-auto px-8 py-12 lg:px-12">
              <header className="mb-12">
                <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tight mb-4 text-foreground font-sans">
                  {lesson.title}
                </h1>
                <p className="text-xl text-muted-foreground prose-academic">
                  {lesson.description}
                </p>
              </header>

              <div className="prose-academic">
                {lesson.sections.map((section) => (
                  <section key={section.id} id={`section-${section.id}`} className="mb-10 scroll-mt-20">
                    {section.title && <h2 className="text-2xl font-bold font-sans text-foreground mb-4">{section.title}</h2>}
                    
                    {section.type === "text" && (
                      <div dangerouslySetInnerHTML={{ __html: section.content }} className="space-y-4" />
                    )}
                    
                    {section.type === "math" && (
                      <MathRenderer content={section.content} block={true} />
                    )}
                    
                    {section.type === "code" && (
                      <CodeBlock code={section.content} language={section.language} runnable={true} />
                    )}
                    
                    {section.type === "callout" && (
                      <Callout type={section.calloutType || "info"} title={section.caption}>
                        <div dangerouslySetInnerHTML={{ __html: section.content }} />
                      </Callout>
                    )}

                    {section.caption && section.type !== "callout" && (
                      <div className="text-center text-sm text-muted-foreground mt-2 italic">
                        {section.caption}
                      </div>
                    )}
                  </section>
                ))}
              </div>

              {lesson.keyTakeaways?.length > 0 && (
                <div className="mt-16 p-8 rounded-2xl bg-secondary/50 border border-border">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <CheckCircle2 className="w-6 h-6 text-primary" /> Key Takeaways
                  </h3>
                  <ul className="space-y-3">
                    {lesson.keyTakeaways.map((point, i) => (
                      <li key={i} className="flex items-start gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 shrink-0" />
                        <span className="prose-academic text-base">{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Bottom Navigation */}
              <div className="mt-16 pt-8 border-t border-border flex items-center justify-between">
                {lesson.prevLessonId ? (
                  <Link href={`/lesson/${lesson.prevLessonId}`} className="group flex flex-col items-start p-4 rounded-xl hover:bg-muted/50 transition-colors">
                    <span className="text-xs text-muted-foreground font-semibold uppercase tracking-wider mb-1 flex items-center gap-1">
                      <ChevronLeft className="w-4 h-4 transition-transform group-hover:-translate-x-1" /> Previous
                    </span>
                    <span className="font-medium text-foreground">View previous lesson</span>
                  </Link>
                ) : <div />}
                
                {lesson.nextLessonId ? (
                  <Link href={`/lesson/${lesson.nextLessonId}`} className="group flex flex-col items-end p-4 rounded-xl hover:bg-muted/50 transition-colors text-right">
                    <span className="text-xs text-muted-foreground font-semibold uppercase tracking-wider mb-1 flex items-center gap-1">
                      Next <ChevronRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                    </span>
                    <span className="font-medium text-foreground">Continue to next lesson</span>
                  </Link>
                ) : (
                  <div className="group flex flex-col items-end p-4 rounded-xl bg-primary/10 border border-primary/20 text-right">
                    <span className="text-xs text-primary font-semibold uppercase tracking-wider mb-1">
                      Module Complete
                    </span>
                    <Link href={`/module/${lesson.moduleId}`} className="font-medium text-foreground hover:text-primary">Return to Module</Link>
                  </div>
                )}
              </div>
            </article>
          </div>

          {/* Interactive Pane (Right side) */}
          {hasExercises && (
            <div id="exercises" className="hidden lg:flex w-[400px] xl:w-[500px] bg-background flex-col shrink-0">
              <div className="flex-1 p-6 overflow-hidden flex flex-col">
                <div className="flex items-center justify-between mb-4 shrink-0">
                  <h3 className="font-bold text-lg">Practice</h3>
                  <div className="text-sm font-medium text-muted-foreground bg-secondary px-2 py-1 rounded-md">
                    {activeExerciseIdx + 1} / {lesson.exercises.length}
                  </div>
                </div>
                
                <div className="flex-1 min-h-0">
                  {currentExercise && (
                    <ExercisePanel 
                      key={currentExercise.id} 
                      exercise={currentExercise} 
                      onSuccess={() => {
                        // Auto advance after short delay if not last
                        if (activeExerciseIdx < lesson.exercises.length - 1) {
                          setTimeout(() => setActiveExerciseIdx(prev => prev + 1), 2000);
                        } else {
                          handleCompleteLesson();
                        }
                      }}
                    />
                  )}
                </div>
                
                {lesson.exercises.length > 1 && (
                  <div className="flex justify-between items-center mt-4 shrink-0">
                    <button 
                      onClick={() => setActiveExerciseIdx(prev => Math.max(0, prev - 1))}
                      disabled={activeExerciseIdx === 0}
                      className="p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg disabled:opacity-30 transition-colors"
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </button>
                    <div className="flex gap-1.5">
                      {lesson.exercises.map((_, i) => (
                        <div key={i} className={cn(
                          "w-2 h-2 rounded-full transition-all duration-300",
                          i === activeExerciseIdx ? "bg-primary w-4" : "bg-border"
                        )} />
                      ))}
                    </div>
                    <button 
                      onClick={() => setActiveExerciseIdx(prev => Math.min(lesson.exercises.length - 1, prev + 1))}
                      disabled={activeExerciseIdx === lesson.exercises.length - 1}
                      className="p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg disabled:opacity-30 transition-colors"
                    >
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
