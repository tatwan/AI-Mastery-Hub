import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../lib/api.ts';
import type { ProgressData } from '../types.ts';

export function useProgress() {
  return useQuery({
    queryKey: ['progress'],
    queryFn: () => api.get<ProgressData>('/api/progress'),
  });
}

export function useMarkComplete() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (lessonId: string) =>
      api.post<null>(`/api/progress/${lessonId}`, { status: 'completed' }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });
}

export function useSaveQuizAnswer() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      lessonId,
      quizIndex,
      selectedOption,
    }: {
      lessonId: string;
      quizIndex: number;
      selectedOption: number;
    }) =>
      api.post<null>(`/api/progress/${lessonId}`, {
        quizAnswer: { quizIndex, selectedOption },
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });
}
