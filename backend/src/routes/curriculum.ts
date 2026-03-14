import { Router } from 'express';
import { loadCurriculum, CONTENT_DIR } from '../lib/content.js';

const router = Router();

router.get('/', async (_req, res) => {
  try {
    const curriculum = await loadCurriculum(CONTENT_DIR);
    res.json({ ok: true, data: curriculum });
  } catch (err) {
    console.error('Failed to load curriculum:', err);
    res.status(500).json({
      ok: false,
      error: { code: 'INTERNAL_ERROR', message: 'Failed to load curriculum' }
    });
  }
});

export default router;
