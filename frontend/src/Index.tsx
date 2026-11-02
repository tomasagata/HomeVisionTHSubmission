import { useState, type DragEvent } from 'react'

type Checkbox = {
  bbox: [number, number, number, number] // [x1, y1, x2, y2]
  is_checked: boolean
}

type Status = 'idle' | 'loading' | 'done' | 'error'

const CHECKED_COLOR = '#22c55e'
const UNCHECKED_COLOR = '#7dd3fc'

function DropZone({ onFile, error }: { onFile: (file: File) => void; error: string }) {
  const [dragging, setDragging] = useState(false)

  function handleDrop(event: DragEvent) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) onFile(file)
  }

  return (
    <div className="centered">
      <label
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="image/*"
          className="visually-hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) onFile(file)
            e.target.value = ''
          }}
        />
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
        <strong>Drop an image here</strong>
        <span className="muted">or click to browse</span>
      </label>
      {error && <p className="error">{error}</p>}
    </div>
  )
}

function Loading({ fileName }: { fileName: string }) {
  return (
    <div className="centered">
      <p>
        Detecting checkboxes in <strong>{fileName}</strong>…
      </p>
      <div className="progress" role="progressbar" aria-busy="true" aria-label="Detecting checkboxes">
        <div />
      </div>
    </div>
  )
}

function ErrorView({ message, onRetry, onBack }: { message: string; onRetry: () => void; onBack: () => void }) {
  return (
    <div className="centered">
      <h2>Something went wrong</h2>
      <p className="muted">{message}</p>
      <div className="actions">
        <button onClick={onRetry}>Try again</button>
        <button onClick={onBack}>Back</button>
      </div>
    </div>
  )
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [size, setSize] = useState<{ width: number; height: number } | null>(null)
  const [boxes, setBoxes] = useState<Checkbox[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState('')
  const [pickError, setPickError] = useState('')

  async function detect(file: File) {
    if (!file.type.startsWith('image/')) {
      setPickError('Please choose an image file')
      return
    }

    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setFile(file)
    setImageUrl(URL.createObjectURL(file))
    setSize(null)
    setBoxes([])
    setPickError('')
    setStatus('loading')

    const body = new FormData()
    body.append('image', file)
    try {
      const response = await fetch('/api/detect', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail ?? `Request failed (${response.status})`)
      setBoxes(data.boxes)
      setStatus('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  function reset() {
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setFile(null)
    setImageUrl(null)
    setSize(null)
    setBoxes([])
    setError('')
    setPickError('')
    setStatus('idle')
  }

  const checked = boxes.filter((box) => box.is_checked).length

  return (
    <main>
      <h1>Checkbox detection</h1>

      {status === 'idle' && <DropZone onFile={detect} error={pickError} />}
      {status === 'loading' && file && <Loading fileName={file.name} />}
      {status === 'error' && file && <ErrorView message={error} onRetry={() => detect(file)} onBack={reset} />}

      {status === 'done' && imageUrl && (
        <>
          <div className="toolbar">
            <button onClick={reset}>← Back</button>
            <p className="status">
              {boxes.length} checkboxes
              <span className="legend">
                <span className="swatch" style={{ borderColor: CHECKED_COLOR }} /> {checked} checked
              </span>
              <span className="legend">
                <span className="swatch" style={{ borderColor: UNCHECKED_COLOR }} /> {boxes.length - checked} unchecked
              </span>
            </p>
          </div>

          <div className="document">
            <img
              src={imageUrl}
              alt="Uploaded document"
              onLoad={(e) => setSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
            />
            {size && (
              <svg viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="none">
                {boxes.map(({ bbox: [x1, y1, x2, y2], is_checked }, i) => (
                  <rect
                    key={i}
                    x={x1}
                    y={y1}
                    width={x2 - x1}
                    height={y2 - y1}
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
        </>
      )}
    </main>
  )
}

export default App
