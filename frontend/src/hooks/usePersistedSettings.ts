import { useState, useEffect, useCallback } from 'react'

const STORAGE_PREFIX = 'ForexEdgeDiscoveryMMLC_'

/**
 * Hook for persisting settings to localStorage with the ForexEdgeDiscoveryMMLC_ prefix.
 * Settings are automatically loaded on mount and saved on change.
 */
export function usePersistedState<T>(
  key: string,
  defaultValue: T
): [T, (value: T | ((prev: T) => T)) => void] {
  const storageKey = `${STORAGE_PREFIX}${key}`

  // Initialize from localStorage or use default
  const [state, setState] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(storageKey)
      if (stored !== null) {
        return JSON.parse(stored)
      }
    } catch (error) {
      console.warn(`Error reading ${storageKey} from localStorage:`, error)
    }
    return defaultValue
  })

  // Save to localStorage whenever state changes
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(state))
    } catch (error) {
      console.warn(`Error saving ${storageKey} to localStorage:`, error)
    }
  }, [storageKey, state])

  return [state, setState]
}

/**
 * Application-wide settings with persistence.
 */
export interface AppSettings {
  workingDirectory: string
  selectedPairs: string[]
  sessionType: string
  timeframe: string
  modelName: string
  trainingEpochs: number
  latentDim: number
  kNeighbors: number
  advancedPipelineOpen: boolean
  advancedEdgeFinderOpen: boolean
}

const DEFAULT_SETTINGS: AppSettings = {
  workingDirectory: 'C:\\Users\\lawfp\\Desktop\\Data4',
  selectedPairs: [],
  sessionType: 'ny',
  timeframe: 'M10',
  modelName: 'vae_default',
  trainingEpochs: 100,
  latentDim: 32,
  kNeighbors: 500,
  advancedPipelineOpen: false,
  advancedEdgeFinderOpen: false,
}

/**
 * Hook that provides all persisted app settings at once.
 * More efficient than calling usePersistedState for each setting.
 */
export function useAppSettings() {
  const [settings, setSettings] = usePersistedState<AppSettings>(
    'settings',
    DEFAULT_SETTINGS
  )

  const updateSettings = useCallback(
    (updates: Partial<AppSettings>) => {
      setSettings((prev) => ({ ...prev, ...updates }))
    },
    [setSettings]
  )

  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS)
  }, [setSettings])

  return {
    settings,
    updateSettings,
    resetSettings,
  }
}

/**
 * Individual setting hooks for backward compatibility.
 */
export function useWorkingDirectory() {
  return usePersistedState('workingDirectory', DEFAULT_SETTINGS.workingDirectory)
}

export function useSelectedPairs() {
  return usePersistedState<string[]>('selectedPairs', DEFAULT_SETTINGS.selectedPairs)
}

export function useSessionType() {
  return usePersistedState('sessionType', DEFAULT_SETTINGS.sessionType)
}

export function useTimeframe() {
  return usePersistedState('timeframe', DEFAULT_SETTINGS.timeframe)
}

export function useModelName() {
  return usePersistedState('modelName', DEFAULT_SETTINGS.modelName)
}

export function useTrainingEpochs() {
  return usePersistedState('trainingEpochs', DEFAULT_SETTINGS.trainingEpochs)
}

export function useLatentDim() {
  return usePersistedState('latentDim', DEFAULT_SETTINGS.latentDim)
}

export function useAdvancedPipelineOpen() {
  return usePersistedState('advancedPipelineOpen', DEFAULT_SETTINGS.advancedPipelineOpen)
}

export function useAdvancedEdgeFinderOpen() {
  return usePersistedState('advancedEdgeFinderOpen', DEFAULT_SETTINGS.advancedEdgeFinderOpen)
}

export function useKNeighbors() {
  return usePersistedState('kNeighbors', DEFAULT_SETTINGS.kNeighbors)
}
