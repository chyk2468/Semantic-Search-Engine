import { useState, useEffect } from 'react'
import './App.css'

interface SearchResult {
  id: string
  text: string
  metadata: {
    category: string
    word_count: number
    cluster_id: number
  }
  similarity: number
}

interface QueryResponse {
  query: string
  cache_hit: boolean
  matched_query?: string
  similarity_score?: number
  result: string
  results_detail: SearchResult[]
  dominant_cluster: number
  dominant_cluster_name: string
  cluster_distribution: number[]
  response_time_ms: number
}

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark')
  }

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      if (!response.ok) throw new Error('Search failed')
      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <h1>Semantic Search Engine</h1>
        <button className="theme-toggle-btn" onClick={toggleTheme}>
          {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
        </button>
      </header>

      <main>
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className="search-input"
          />
          <button type="submit" disabled={loading} className="search-button">
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {results && (
          <div className="results-container">
            <div className="search-stats">
              <span>Time: {results.response_time_ms}ms</span>
              <span>Cluster: {results.dominant_cluster_name}</span>
              {results.cache_hit && <span className="cache-badge">Cache Hit</span>}
            </div>

            <div className="results-list">
              {results.results_detail.map((res) => (
                <div key={res.id} className="result-card">
                  <div className="result-header">
                    <span className="category">{res.metadata.category}</span>
                    <span className="similarity">Match: {(res.similarity * 100).toFixed(1)}%</span>
                  </div>
                  <p className="result-text">{res.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
