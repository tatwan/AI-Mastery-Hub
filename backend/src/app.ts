import express from 'express';
import cors from 'cors';
import curriculumRouter from './routes/curriculum.js';
import lessonsRouter from './routes/lessons.js';
import progressRouter from './routes/progress.js';

const app = express();
app.use(cors());
app.use(express.json());

app.use('/api/curriculum', curriculumRouter);
app.use('/api/lessons', lessonsRouter);
app.use('/api/progress', progressRouter);

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, data: { status: 'ok' } });
});

const PORT = process.env.PORT ?? 3001;
app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

export default app;
