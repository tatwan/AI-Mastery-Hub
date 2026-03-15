const SESSION_KEY = 'amh_session_id';
const FIRST_VISIT_KEY = 'amh_first_visit';

function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = generateUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function isFirstVisit(): boolean {
  return localStorage.getItem(FIRST_VISIT_KEY) === null;
}

export function markVisited(): void {
  localStorage.setItem(FIRST_VISIT_KEY, 'true');
}
