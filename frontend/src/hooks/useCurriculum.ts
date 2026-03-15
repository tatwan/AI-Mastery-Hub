import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api.ts';
import type { Curriculum } from '../types.ts';

export function useCurriculum() {
  return useQuery({
    queryKey: ['curriculum'],
    queryFn: () => api.get<Curriculum>('/api/curriculum'),
    staleTime: Infinity,
  });
}
