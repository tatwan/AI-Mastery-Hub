import { Router, type IRouter } from "express";
import {
  allTracks,
  trackMap,
  moduleMap,
  lessonMap,
  getTrackSummary,
  getModuleSummary,
} from "../data/curriculum.js";

const router: IRouter = Router();

// GET /api/curriculum/tracks — all tracks (summary)
router.get("/tracks", (_req, res) => {
  const tracks = allTracks.map(getTrackSummary);
  tracks.sort((a, b) => a.order - b.order);
  res.json(tracks);
});

// GET /api/curriculum/tracks/:trackId — single track (summary with modules)
router.get("/tracks/:trackId", (req, res) => {
  const track = trackMap.get(req.params.trackId);
  if (!track) {
    res.status(404).json({ error: "not_found", message: `Track '${req.params.trackId}' not found` });
    return;
  }
  res.json(getTrackSummary(track));
});

// GET /api/curriculum/modules/:moduleId — module with lesson summaries
router.get("/modules/:moduleId", (req, res) => {
  const module = moduleMap.get(req.params.moduleId);
  if (!module) {
    res.status(404).json({ error: "not_found", message: `Module '${req.params.moduleId}' not found` });
    return;
  }
  res.json(getModuleSummary(module));
});

// GET /api/curriculum/lessons/:lessonId — full lesson content
router.get("/lessons/:lessonId", (req, res) => {
  const lesson = lessonMap.get(req.params.lessonId);
  if (!lesson) {
    res.status(404).json({ error: "not_found", message: `Lesson '${req.params.lessonId}' not found` });
    return;
  }
  res.json(lesson);
});

export default router;
