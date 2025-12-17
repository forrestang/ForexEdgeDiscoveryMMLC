export const SESSION_OPTIONS = [
  { value: 'full_day', label: 'Full Day (00:00-22:00 UTC)' },
  { value: 'asia', label: 'Asia (00:00-09:00 UTC)' },
  { value: 'london', label: 'London (08:00-17:00 UTC)' },
  { value: 'ny', label: 'New York (13:00-22:00 UTC)' },
] as const;

export const TIMEFRAME_OPTIONS = [
  { value: 'M1', label: 'M1 (1 Minute)' },
  { value: 'M5', label: 'M5 (5 Minutes)' },
  { value: 'M10', label: 'M10 (10 Minutes)' },
  { value: 'M15', label: 'M15 (15 Minutes)' },
  { value: 'M30', label: 'M30 (30 Minutes)' },
  { value: 'H1', label: 'H1 (1 Hour)' },
  { value: 'H4', label: 'H4 (4 Hours)' },
] as const;

export const WAVE_COLORS = {
  1: '#FFD700',  // Yellow
  2: '#00FFFF',  // Cyan
  3: '#FF0000',  // Red
  4: '#800080',  // Purple
  5: '#90EE90',  // Light Green
} as const;

export const API_BASE_URL = '/api';

export const DEFAULT_WORKING_DIRECTORY = 'C:\\Users\\lawfp\\Desktop\\Data4';
