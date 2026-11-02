import { useState, type ChangeEvent } from 'react'

type Checkbox = {
  bbox: [number, number, number, number]
  is_checked: boolean
}

type Status = 'idle' | 'loading' | 'done' | 'error'

const CHECKED_COLOR = '#22c55e'
const UNCHECKED_COLOR = '#7dd3fc'

function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [size, setSize] = useState<{ width: number; height: number } | null>(null)
  const [boxes, setBoxes] = useState<Checkbox[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState('')

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImageUrl(URL.createObjectURL(file))
    setSize(null)
    setBoxes([])
    setStatus('loading')

    const body = new FormData()
    body.append('image', file)
    try {
      const response = await fetch('/api/detect', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail ?? `Request failed (${response.status})`)
      setBoxes(data)
      setStatus('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  const checked = boxes.filter((box) => box.is_checked).length

  return (
    <main>
      <h1>Checkbox detection</h1>
      <input type="file" accept="image/*" onChange={handleFile} disabled={status === 'loading'} />

      <p className="status">
        {status === 'loading' && 'Detecting…'}
        {status === 'error' && <span className="error">{error}</span>}
        {status === 'done' && (
          <>
            {boxes.length} checkboxes
            <span className="legend">
              <span className="swatch" style={{ borderColor: CHECKED_COLOR }} /> {checked} checked
            </span>
            <span className="legend">
              <span className="swatch" style={{ borderColor: UNCHECKED_COLOR }} /> {boxes.length - checked} unchecked
            </span>
          </>
        )}
      </p>

      {imageUrl && (
        <div className="document">
          <img
            src={imageUrl}
            alt="Uploaded document"
            onLoad={(e) => setSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
          />
          {size && status === 'done' && (
            <svg viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="none">
              {boxes.map(({ bbox: [x, y, w, h], is_checked }, i) => (
                <rect
                  key={i}
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  fill="none"
                  stroke={is_checked ? CHECKED_COLOR : UNCHECKED_COLOR}
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                >
                  <title>{is_checked ? 'Checked' : 'Unchecked'}</title>
                </rect>
              ))}
            </svg>
          )}
        </div>
      )}
    </main>
  )
}

export default App
