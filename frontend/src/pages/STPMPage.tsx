import { motion } from 'framer-motion'

export function STPMPage() {
  return (
    <div className="stpm-page">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="stpm-hero"
      >
        <div className="stpm-hero-badge">Coming Soon</div>
        <h1 className="stpm-hero-title">STPM</h1>
        <p className="stpm-hero-subtitle">State Transition Probability Model</p>

        <div className="stpm-description">
          <p>
            A probabilistic framework for modeling market state transitions
            and predicting regime changes based on historical patterns.
          </p>
        </div>

        <div className="stpm-features">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="stpm-feature"
          >
            <div className="stpm-feature-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M3 3v18h18" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M7 14l4-4 4 4 5-5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <div className="stpm-feature-content">
              <h3>Markov Chain Analysis</h3>
              <p>State-based probability transitions</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="stpm-feature"
          >
            <div className="stpm-feature-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="3"/>
                <circle cx="12" cy="12" r="8" strokeDasharray="4 2"/>
                <path d="M12 2v2M12 20v2M2 12h2M20 12h2" strokeLinecap="round"/>
              </svg>
            </div>
            <div className="stpm-feature-content">
              <h3>Regime Detection</h3>
              <p>Identify market regime shifts in real-time</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="stpm-feature"
          >
            <div className="stpm-feature-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 3l9 4.5v9L12 21l-9-4.5v-9L12 3z"/>
                <path d="M12 12l9-4.5M12 12v9M12 12L3 7.5"/>
              </svg>
            </div>
            <div className="stpm-feature-content">
              <h3>Multi-Dimensional States</h3>
              <p>Complex state representations</p>
            </div>
          </motion.div>
        </div>
      </motion.div>

      <div className="stpm-grid-bg" />
    </div>
  )
}
