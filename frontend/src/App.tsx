import { useState } from 'react';
import { Container, Card, Form, Button, Alert, Spinner } from 'react-bootstrap';
import { api } from './api';
import type { AnalyzeResponse } from './api';

export default function App() {
  const [mode, setMode] = useState<'url' | 'text'>('url');
  const [url, setUrl] = useState('');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);


  async function analyzeUrl() {
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await api.analyzeUrl(url);
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  async function analyzeText() {
    setLoading(true); setError(''); setResult(null);
    try {
      const data = await api.analyzeText(text);
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  // one submit handler for both modes
  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (mode === 'url' && url) return void analyzeUrl();
    if (mode === 'text' && text) return void analyzeText();
  }

  return (
    // Full-viewport, centered both ways 
    <main className="position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center">
      <Container style={{ maxWidth: 820 }}>
        <Card className="shadow-sm border-0">
          <Card.Body className="p-4 p-md-5">

            {/* Header */}
            <h1 className="text-center fw-bold display-6">NewsLens</h1>
            <p>NewsLens is a tool that helps readers detect bias and emotionally charged language in news articles. It highlights sentiment and potential political leanings (liberal, conservative, or neutral), making it easier to spot subtle narrative-shaping tactics that might be missed when skimming. </p>
            <p className="text-center text-muted mx-auto" style={{ maxWidth: 720 }}>
              Paste a URL or full text to analyze sentiment, political leaning, and charged words.
            </p>

            {/* Mode toggle */}
            <div className="d-flex justify-content-center gap-2 mb-3">
              <Button
                variant={mode === 'url' ? 'primary' : 'outline-secondary'}
                onClick={() => setMode('url')}>
                URL
              </Button>
              <Button
                variant={mode === 'text' ? 'primary' : 'outline-secondary'}
                onClick={() => setMode('text')}
              >
                Paste Text
              </Button>
            </div>

            {/* Form */}
            <Form onSubmit={onSubmit}>
              {mode === 'url' ? (
                <div className="d-flex gap-2">
                  <Form.Control
                    type="url"
                    placeholder="https://news-site.com/article"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    required
                  />
                  <Button type="submit" disabled={!url || loading}>
                    {loading ? (<><Spinner animation="border" size="sm" className="me-2" />Analyzing…</>) : 'Analyze URL'}
                  </Button>
                </div>
              ) : (
                <>
                  <Form.Control
                    as="textarea"
                    rows={10}
                    placeholder="Paste the full article text here…"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                  />
                  <div className="d-flex gap-2 mt-2">
                    <Button type="submit" disabled={!text || loading}>
                      {loading ? (<><Spinner animation="border" size="sm" className="me-2" />Analyzing…</>) : 'Analyze Text'}
                    </Button>
                    <Button variant="outline-secondary" onClick={() => setText('')} disabled={loading}>
                      Clear
                    </Button>
                  </div>
                </>
              )}
            </Form>

            {/* Error */}
            {error && <Alert variant="danger" className="mt-3">{error}</Alert>}

            {/* Result */}
            {result && (
              <div className="mt-3 overflow-auto" style={{ maxHeight: '60vh', paddingRight: '0.25rem' }}>
                <h5 className="text-center mb-3">Analysis Result</h5>

                <div className="d-flex flex-wrap justify-content-center gap-2 mb-3">
                  <span className="badge text-bg-primary">Sentiment: {result.sentiment}</span>
                  <span className="badge text-bg-secondary">Bias: {result.bias}</span>
                  <span className="badge text-bg-warning text-dark">Charged words: {result.charged_total}</span>
                </div>

                {Array.isArray(result.top_positive) && result.top_positive.length > 0 && (
                  <>
                    <h6 className="mt-3">Top Positive</h6>
                    {result.top_positive.map((h, i) => (
                      <div key={`pos-${i}`} className="small mb-1">
                        <strong>{h.score.toFixed(3)}</strong> — {h.text}
                      </div>
                    ))}
                  </>
                )}

                {Array.isArray(result.top_negative) && result.top_negative.length > 0 && (
                  <>
                    <h6 className="mt-3">Top Negative</h6>
                    {result.top_negative.map((h, i) => (
                      <div key={`neg-${i}`} className="small mb-1">
                        <strong>{h.score.toFixed(3)}</strong> — {h.text}
                      </div>
                    ))}
                  </>
                )}

                {Array.isArray(result.top_biased) && result.top_biased.length > 0 && (
                  <>
                    <h6 className="mt-3">Top Biased</h6>
                    {result.top_biased.map((h, i) => (
                      <div key={`biased-${i}`} className="small mb-1">
                        <strong>{(h.side ?? 'biased')} {h.score.toFixed(3)}</strong> — {h.text}
                      </div>
                    ))}
                  </>
                )}
                {Array.isArray(result.charged_unique) && result.charged_unique.length > 0 && (
                <>
                  <div className="d-flex flex-wrap gap-2 mt-3">
                    <small className="text-muted">Charged words found (Note: The interpretation of charged words may vary depending on context, and in some cases, the highlighted words may not actually be charged.):</small>
                    {result.charged_unique.map((w, i) => (
                      <span key={i} className="badge text-bg-warning text-dark">{w}</span>
                    ))}
                  </div>
                </>
              )}

              </div>
            )}
          </Card.Body>
        </Card>
      </Container>
    </main>
  );
}
