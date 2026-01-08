import { motion } from 'framer-motion'

type TabType = 'ae-knn' | 'lstm' | 'stpm' | 'sandbox'

interface TopNavigationProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

const tabs: { id: TabType; title: string; subtitle: string }[] = [
  { id: 'ae-knn', title: 'AE+KNN', subtitle: 'Similarity Model' },
  { id: 'lstm', title: 'LSTM', subtitle: 'Symbolic Time Series Analysis' },
  { id: 'stpm', title: 'STPM', subtitle: 'State Transition Probability Model' },
  { id: 'sandbox', title: 'Sandbox', subtitle: 'Development Environment' },
]

export function TopNavigation({ activeTab, onTabChange }: TopNavigationProps) {
  return (
    <div className="top-nav">
      <div className="top-nav-brand">
        <span className="top-nav-brand-mark">FED</span>
        <span className="top-nav-brand-text">Forex Edge Discovery</span>
      </div>

      <nav className="top-nav-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`top-nav-tab ${activeTab === tab.id ? 'active' : ''}`}
          >
            <span className="top-nav-tab-title">{tab.title}</span>
            <span className="top-nav-tab-subtitle">{tab.subtitle}</span>
            {activeTab === tab.id && (
              <motion.div
                layoutId="activeTab"
                className="top-nav-tab-indicator"
                transition={{ type: 'spring', stiffness: 500, damping: 35 }}
              />
            )}
          </button>
        ))}
      </nav>

      <div className="top-nav-status">
        <div className="top-nav-status-dot" />
        <span>System Active</span>
      </div>
    </div>
  )
}
