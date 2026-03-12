import { Router, type IRouter } from "express";
import { db } from "@workspace/db";
import { lessonProgressTable, exerciseResultsTable } from "@workspace/db/schema";
import { eq } from "drizzle-orm";
import { lessonMap } from "../data/curriculum.js";

const router: IRouter = Router();

// GET /api/progress — all progress
router.get("/", async (_req, res) => {
  try {
    const lessonProgress = await db.select().from(lessonProgressTable);
    const exerciseResults = await db.select().from(exerciseResultsTable);

    const completedLessons = lessonProgress
      .filter((p) => p.status === "completed")
      .map((p) => p.lessonId);

    const totalXp = exerciseResults.reduce((sum, r) => sum + (r.xpEarned || 0), 0);

    // Simple streak calculation
    const today = new Date();
    const streakDays = 1; // Simplified — would compute from timestamps in production

    res.json({
      completedLessons,
      lessonProgress: lessonProgress.map((p) => ({
        lessonId: p.lessonId,
        status: p.status,
        completedAt: p.completedAt?.toISOString(),
        score: p.score,
        timeSpentMinutes: p.timeSpentMinutes,
      })),
      exerciseResults: exerciseResults.map((r) => ({
        exerciseId: r.exerciseId,
        isCorrect: r.isCorrect,
        answer: r.answer,
        explanation: "",
        xpEarned: r.xpEarned,
        submittedAt: r.submittedAt?.toISOString(),
      })),
      totalXp,
      streakDays,
      lastActiveDate: today.toISOString(),
    });
  } catch (err) {
    console.error("Error fetching progress:", err);
    res.status(500).json({ error: "internal_error", message: "Failed to fetch progress" });
  }
});

// POST /api/progress/lesson/:lessonId
router.post("/lesson/:lessonId", async (req, res) => {
  const { lessonId } = req.params;
  const { status, timeSpentMinutes } = req.body as {
    status: "not_started" | "in_progress" | "completed";
    timeSpentMinutes?: number;
  };

  try {
    const existing = await db
      .select()
      .from(lessonProgressTable)
      .where(eq(lessonProgressTable.lessonId, lessonId));

    const completedAt = status === "completed" ? new Date() : null;

    if (existing.length > 0) {
      await db
        .update(lessonProgressTable)
        .set({
          status,
          timeSpentMinutes: timeSpentMinutes ?? existing[0].timeSpentMinutes,
          completedAt: completedAt ?? existing[0].completedAt,
          updatedAt: new Date(),
        })
        .where(eq(lessonProgressTable.lessonId, lessonId));
    } else {
      await db.insert(lessonProgressTable).values({
        lessonId,
        status,
        timeSpentMinutes: timeSpentMinutes ?? 0,
        completedAt,
        updatedAt: new Date(),
      });
    }

    const updated = await db
      .select()
      .from(lessonProgressTable)
      .where(eq(lessonProgressTable.lessonId, lessonId));

    const p = updated[0];
    res.json({
      lessonId: p.lessonId,
      status: p.status,
      completedAt: p.completedAt?.toISOString(),
      score: p.score,
      timeSpentMinutes: p.timeSpentMinutes,
    });
  } catch (err) {
    console.error("Error updating lesson progress:", err);
    res.status(500).json({ error: "internal_error", message: "Failed to update progress" });
  }
});

// POST /api/progress/exercise/:exerciseId
router.post("/exercise/:exerciseId", async (req, res) => {
  const { exerciseId } = req.params;
  const { answer, code, timeSpentSeconds } = req.body as {
    answer: string;
    code?: string;
    timeSpentSeconds?: number;
  };

  // Find exercise in curriculum to check answer
  let isCorrect = false;
  let explanation = "";
  let xpEarned = 0;

  for (const [, lesson] of lessonMap) {
    const exercise = lesson.exercises.find((e) => e.id === exerciseId);
    if (exercise) {
      isCorrect =
        answer.toLowerCase().trim() === exercise.correctAnswer.toLowerCase().trim() ||
        exercise.correctAnswer.toLowerCase().includes(answer.toLowerCase().trim().substring(0, 10));
      explanation = exercise.explanation;
      xpEarned = isCorrect ? (exercise.type === "write_code" ? 50 : 20) : 5;
      break;
    }
  }

  try {
    // Upsert exercise result
    const existing = await db
      .select()
      .from(exerciseResultsTable)
      .where(eq(exerciseResultsTable.exerciseId, exerciseId));

    if (existing.length > 0) {
      await db
        .update(exerciseResultsTable)
        .set({
          isCorrect,
          answer,
          xpEarned: isCorrect ? xpEarned : existing[0].xpEarned,
          submittedAt: new Date(),
        })
        .where(eq(exerciseResultsTable.exerciseId, exerciseId));
    } else {
      await db.insert(exerciseResultsTable).values({
        exerciseId,
        isCorrect,
        answer,
        xpEarned,
        submittedAt: new Date(),
      });
    }

    res.json({
      exerciseId,
      isCorrect,
      answer,
      explanation,
      xpEarned,
      submittedAt: new Date().toISOString(),
    });
  } catch (err) {
    console.error("Error submitting exercise:", err);
    res.status(500).json({ error: "internal_error", message: "Failed to submit exercise" });
  }
});

export default router;
