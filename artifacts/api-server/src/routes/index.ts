import { Router, type IRouter } from "express";
import healthRouter from "./health.js";
import curriculumRouter from "./curriculum.js";
import progressRouter from "./progress.js";

const router: IRouter = Router();

router.use(healthRouter);
router.use("/curriculum", curriculumRouter);
router.use("/progress", progressRouter);

export default router;
