import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api.ts';
import type { Lesson } from '../types.ts';

export function useLesson(semId: string, modId: string, lessonId: string) {
  return useQuery({
    queryKey: ['lesson', semId, modId, lessonId],
    queryFn: () => api.get<Lesson>(`/api/lessons/${semId}/${modId}/${lessonId}`),
    enabled: Boolean(semId && modId && lessonId),
  });
}
