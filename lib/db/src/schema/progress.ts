import { pgTable, text, real, timestamp, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const lessonProgressTable = pgTable("lesson_progress", {
  lessonId: text("lesson_id").primaryKey(),
  status: text("status").notNull().default("not_started"),
  completedAt: timestamp("completed_at"),
  score: real("score"),
  timeSpentMinutes: real("time_spent_minutes").default(0),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const exerciseResultsTable = pgTable("exercise_results", {
  exerciseId: text("exercise_id").primaryKey(),
  isCorrect: boolean("is_correct").notNull(),
  answer: text("answer").notNull(),
  xpEarned: real("xp_earned").notNull().default(0),
  submittedAt: timestamp("submitted_at").defaultNow(),
});

export const insertLessonProgressSchema = createInsertSchema(lessonProgressTable);
export type InsertLessonProgress = z.infer<typeof insertLessonProgressSchema>;
export type LessonProgressRow = typeof lessonProgressTable.$inferSelect;

export const insertExerciseResultSchema = createInsertSchema(exerciseResultsTable);
export type InsertExerciseResult = z.infer<typeof insertExerciseResultSchema>;
export type ExerciseResultRow = typeof exerciseResultsTable.$inferSelect;
