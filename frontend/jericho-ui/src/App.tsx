import React, { useState, useRef, useEffect } from 'react'

type Screen = 'login' | 'chat'

type ChatMessage = {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: any[]
  confidence?: number
}

type SessionInfo = {
  session_id: number
  session_name: string
}

function App() {
  const [screen, setScreen] = useState<Screen>('login')
  const [username, setUsername] = useState<string>('')
  const [password, setPassword] = useState<string>('')
  const [role, setRole] = useState<string | null>(null)
  const [loginError, setLoginError] = useState<string | null>(null)
  const [loginLoading, setLoginLoading] = useState(false)

  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [nextId, setNextId] = useState(1)

  // Sessions
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null)
  const [sessionsLoading, setSessionsLoading] = useState(false)

  // Rename/delete UI
  const [sessionMenuOpenId, setSessionMenuOpenId] = useState<number | null>(
    null
  )
  const [showRenameModal, setShowRenameModal] = useState(false)
  const [renameSessionId, setRenameSessionId] = useState<number | null>(null)
  const [renameValue, setRenameValue] = useState('')

  // Upload state
  const [showUpload, setShowUpload] = useState(false)
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [uploadPrivate, setUploadPrivate] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)

  // Admin console state
  const [adminView, setAdminView] = useState<'chat' | 'admin'>('chat')
  const [adminStats, setAdminStats] = useState<any | null>(null)
  const [adminStatsLoading, setAdminStatsLoading] = useState(false)
  const [adminTestQuery, setAdminTestQuery] = useState('')
  const [adminTestResponse, setAdminTestResponse] = useState<any | null>(null)
  const [adminTestLoading, setAdminTestLoading] = useState(false)

  // Auto-scroll anchor
  const messagesEndRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  // ---------- LOGIN ----------

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoginError(null)
    if (!username.trim() || !password.trim()) {
      setLoginError('Username and password are required.')
      return
    }
    setLoginLoading(true)
    try {
      const resp = await fetch('http://localhost:8000/apiv1/react-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      setRole(data.role)
      setScreen('chat')
    } catch (err) {
      console.error(err)
      setLoginError('Invalid username or password.')
    } finally {
      setLoginLoading(false)
    }
  }

  // Once we are on chat, load sessions
  useEffect(() => {
    if (screen === 'chat') {
      loadSessions()
    }
  }, [screen])

  // Load admin stats when switching into admin view
  useEffect(() => {
    if (screen === 'chat' && role === 'admin' && adminView === 'admin') {
      loadAdminStats()
    }
  }, [screen, role, adminView])

  // ---------- SESSIONS + HISTORY ----------

  const loadSessions = async () => {
    setSessionsLoading(true)
    try {
      const resp = await fetch('http://localhost:8000/user_sessions', {
        method: 'GET',
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const list: SessionInfo[] = data.sessions || []
      setSessions(list)

      if (list.length > 0) {
        const last = list[list.length - 1]
        setCurrentSessionId(last.session_id)
        await loadHistory(last.session_id)
      } else {
        await handleNewChat()
      }
    } catch (err) {
      console.error(err)
    } finally {
      setSessionsLoading(false)
    }
  }

  const loadHistory = async (sessionId: number) => {
    try {
      const resp = await fetch(
        `http://localhost:8000/history?session_id=${sessionId}`,
        {
          method: 'GET',
          credentials: 'include',
        }
      )
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const history = data.history || []
      const mapped: ChatMessage[] = []
      let idCounter = 1
      for (const item of history) {
        if (item.question) {
          mapped.push({
            id: idCounter++,
            role: 'user',
            content: item.question,
          })
        }
        if (item.answer) {
          mapped.push({
            id: idCounter++,
            role: 'assistant',
            content: item.answer,
          })
        }
      }
      setMessages(mapped)
      setNextId(idCounter)
    } catch (err) {
      console.error(err)
      setMessages([])
      setNextId(1)
    }
  }

  const handleNewChat = async () => {
    try {
      const resp = await fetch('http://localhost:8000/new_session', {
        method: 'POST',
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const sessionId: number = data.session_id
      const newSession: SessionInfo = {
        session_id: sessionId,
        session_name: 'New Chat',
      }
      setSessions((prev) => [...prev, newSession])
      setCurrentSessionId(sessionId)
      setMessages([])
      setNextId(1)
    } catch (err) {
      console.error(err)
    }
  }

  const handleSelectSession = async (sessionId: number) => {
    if (sessionId === currentSessionId) return
    setCurrentSessionId(sessionId)
    await loadHistory(sessionId)
  }

  // ---------- SESSION RENAME / DELETE ----------

  const openRenameModal = (session: SessionInfo) => {
    setRenameSessionId(session.session_id)
    setRenameValue(session.session_name)
    setShowRenameModal(true)
    setSessionMenuOpenId(null)
  }

  const handleRenameSubmit = async () => {
    if (!renameSessionId) return
    const name = renameValue.trim() || 'New Chat'
    try {
      const form = new FormData()
      form.append('session_id', String(renameSessionId))
      form.append('new_name', name)

      const resp = await fetch('http://localhost:8000/rename_session', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      setSessions((prev) =>
        prev.map((s) =>
          s.session_id === renameSessionId ? { ...s, session_name: name } : s
        )
      )
      setShowRenameModal(false)
    } catch (err) {
      console.error(err)
    }
  }

  const handleDeleteSession = async (sessionId: number) => {
    try {
      const form = new FormData()
      form.append('session_id', String(sessionId))

      const resp = await fetch('http://localhost:8000/delete_session', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }

      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
      setSessionMenuOpenId(null)

      if (currentSessionId === sessionId) {
        const remaining = sessions.filter((s) => s.session_id !== sessionId)
        if (remaining.length > 0) {
          const last = remaining[remaining.length - 1]
          setCurrentSessionId(last.session_id)
          await loadHistory(last.session_id)
        } else {
          setCurrentSessionId(null)
          setMessages([])
          setNextId(1)
          await handleNewChat()
        }
      }
    } catch (err) {
      console.error(err)
    }
  }

  // ---------- CHAT SEND USING /react-query (with sources) ----------

  const handleSend = async () => {
    const q = input.trim()
    if (!q || !currentSessionId) return
    setInput('')

    const userMessage: ChatMessage = {
      id: nextId,
      role: 'user',
      content: q,
    }
    setNextId(nextId + 1)
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    try {
      const resp = await fetch('http://localhost:8000/react-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          sessionid: currentSessionId,
          private: false,
        }),
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const assistantMessage: ChatMessage = {
        id: nextId + 1,
        role: 'assistant',
        content: data.answer ?? 'No answer field in response.',
        sources: data.sources || [],
        confidence:
          typeof data.confidence === 'number' ? data.confidence : undefined,
      }
      setNextId(nextId + 2)
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      console.error(err)
      const errorMessage: ChatMessage = {
        id: nextId + 1,
        role: 'assistant',
        content: 'Error calling backend. Check server logs.',
      }
      setNextId(nextId + 2)
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  // ---------- UPLOAD USING /upload ----------

  const handleUpload = async () => {
    if (!uploadFiles || uploadFiles.length === 0) {
      setUploadStatus('Please select at least one file.')
      return
    }
    if (!currentSessionId) {
      setUploadStatus('No active session selected.')
      return
    }
    setUploadLoading(true)
    setUploadStatus(null)
    try {
      const form = new FormData()
      Array.from(uploadFiles).forEach((file) => {
        form.append('files', file)
      })
      form.append('session_id', String(currentSessionId))
      form.append('private', uploadPrivate ? 'true' : 'false')

      const resp = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: form,
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      const msg =
        data.message ||
        `Processed ${data.processed_files?.length ?? 0} file(s).`
      setUploadStatus(msg)
    } catch (err) {
      console.error(err)
      setUploadStatus('Upload failed. Check server logs.')
    } finally {
      setUploadLoading(false)
    }
  }

  // ---------- ADMIN STATS + TEST QUERY ----------

  const loadAdminStats = async () => {
    setAdminStatsLoading(true)
    try {
      const resp = await fetch('http://localhost:8000/admin/stats', {
        method: 'GET',
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      setAdminStats(data)
    } catch (err) {
      console.error(err)
      setAdminStats(null)
    } finally {
      setAdminStatsLoading(false)
    }
  }

  const handleAdminTestQuery = async () => {
    const q = adminTestQuery.trim()
    if (!q) return
    setAdminTestLoading(true)
    setAdminTestResponse(null)
    try {
      const resp = await fetch('http://localhost:8000/react-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          sessionid: currentSessionId || 1,
          private: false,
        }),
        credentials: 'include',
      })
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`)
      }
      const data = await resp.json()
      setAdminTestResponse(data)
    } catch (err) {
      console.error(err)
      setAdminTestResponse({ error: 'Request failed. Check logs.' })
    } finally {
      setAdminTestLoading(false)
    }
  }

  // ---------- LOGOUT ----------

  const handleLogout = async () => {
    try {
      await fetch('http://localhost:8000/logout', {
        method: 'POST',
        credentials: 'include',
      })
    } catch (e) {
      console.error(e)
    }
    setUsername('')
    setPassword('')
    setRole(null)
    setSessions([])
    setCurrentSessionId(null)
    setMessages([])
    setScreen('login')
  }

  // ---------- RENDER ----------

  if (screen === 'login') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="absolute top-6 left-6 flex items-center gap-3">
          <img
            src="https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png"
            alt="Diné College"
            className="h-10 rounded-md object-contain"
          />
        </div>

        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl shadow-2xl p-8 w-full max-w-md text-white">
          <h1 className="text-2xl font-semibold mb-2">
            Diné College Assistant
          </h1>

          {loginError && (
            <div className="mb-4 text-sm text-red-300 bg-red-900/40 border border-red-400/60 rounded px-3 py-2">
              {loginError}
            </div>
          )}

          <form className="space-y-4" onSubmit={handleLogin}>
            <div>
              <label className="block text-sm mb-1">Username</label>
              <input
                type="text"
                className="w-full rounded-md border border-slate-500 bg-slate-900/60 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Password</label>
              <input
                type="password"
                className="w-full rounded-md border border-slate-500 bg-slate-900/60 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <button
              type="submit"
              className="w-full mt-2 rounded-md bg-amber-500 hover:bg-amber-400 text-slate-900 font-semibold py-2 text-sm transition disabled:opacity-60"
              disabled={loginLoading}
            >
              {loginLoading ? 'Signing in...' : 'Login'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-100">
      <header className="h-14 px-4 flex items-center justify-between bg-white border-b">
        <div className="flex items-center gap-3">
          <img
            src="https://www.dinecollege.edu/wp-content/uploads/2024/12/dc_logoFooter.png"
            alt="Diné College"
            className="h-8 rounded-md object-contain"
          />
          <span className="font-semibold text-slate-800">
            Diné College Assistant (Jericho)
          </span>
        </div>
        <div className="flex items-center gap-3 text-sm text-slate-600">
          <span>{username || 'User'}</span>
          {role === 'admin' && (
            <span className="px-2 py-0.5 rounded-full bg-slate-900 text-amber-300 text-xs">
              Admin
            </span>
          )}
          <button
            className="text-xs px-3 py-1 rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100"
            onClick={handleLogout}
          >
            Logout
          </button>
        </div>
      </header>

      <main className="flex-1 flex">
        <aside className="w-64 border-r bg-white p-3 flex flex-col">
          <button
            className="w-full mb-3 rounded-md bg-amber-500 hover:bg-amber-400 text-sm font-semibold py-2 text-slate-900 disabled:opacity-60"
            onClick={handleNewChat}
            disabled={sessionsLoading}
          >
            New chat
          </button>

          <div className="text-xs text-slate-500 px-1 mb-2 flex items-center justify-between">
            <span>Chats</span>
            {sessionsLoading && <span className="text-[10px]">Loading…</span>}
          </div>

          {role === 'admin' && (
            <button
              className={`w-full mb-3 mt-1 rounded-md text-sm font-semibold py-2 ${
                adminView === 'admin'
                  ? 'bg-slate-900 text-amber-200'
                  : 'bg-slate-200 text-slate-800 hover:bg-slate-300'
              }`}
              onClick={() =>
                setAdminView(adminView === 'admin' ? 'chat' : 'admin')
              }
            >
              {adminView === 'admin' ? 'Back to chat' : 'Admin console'}
            </button>
          )}

          <div className="flex-1 overflow-auto text-sm">
            {sessions.length === 0 && !sessionsLoading && (
              <div className="text-xs text-slate-400 px-1">
                No sessions yet. Click “New chat”.
              </div>
            )}
            {sessions.map((s) => (
              <div
                key={s.session_id}
                className={`flex items-center mb-1 rounded-md ${
                  s.session_id === currentSessionId
                    ? 'bg-slate-900 text-amber-200'
                    : 'bg-slate-100 text-slate-700'
                }`}
              >
                <button
                  className="flex-1 text-left px-2 py-1.5 truncate hover:bg-slate-800/10"
                  onClick={() => handleSelectSession(s.session_id)}
                >
                  {s.session_name || `Session ${s.session_id}`}
                </button>
                <button
                  className="px-2 text-xs hover:bg-slate-800/20 rounded-r-md"
                  onClick={() =>
                    setSessionMenuOpenId(
                      sessionMenuOpenId === s.session_id ? null : s.session_id
                    )
                  }
                >
                  ⋮
                </button>
                {sessionMenuOpenId === s.session_id && (
                  <div className="absolute ml-40 mt-10 z-50 bg-white border border-slate-200 rounded-md shadow-md text-xs text-slate-700">
                    <button
                      className="block w-full px-3 py-1 hover:bg-slate-100 text-left"
                      onClick={() => openRenameModal(s)}
                    >
                      Rename
                    </button>
                    <button
                      className="block w-full px-3 py-1 hover:bg-red-50 text-left text-red-600"
                      onClick={() => handleDeleteSession(s.session_id)}
                    >
                      Delete
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </aside>

        <section className="flex-1 flex flex-col">
          {adminView === 'chat' || role !== 'admin' ? (
            <>
              {/* Chat view */}
              <div className="flex-1 p-4 overflow-auto bg-slate-50">
                <div className="max-w-3xl mx-auto space-y-3">
                  {messages.length === 0 && (
                    <div className="text-slate-500 text-sm">
                      Start a conversation by asking a question about Diné
                      College.
                    </div>
                  )}

                  {messages.map((m) => (
                    <div
                    className={`${
                      m.role === 'user' ? 'justify-end' : 'justify-start'
                    } flex`}
                  >
                    <div
                      className={`rounded-lg shadow-sm ${
                        m.role === 'user'
                          ? 'bg-amber-500 text-slate-900 px-4 py-2 max-w-xl'
                          : 'bg-white text-slate-800 p-0 max-w-2xl w-full'
                      }`}
                    >
                      {/* Render HTML content for assistant, plain text for user */}
                      {m.role === 'assistant' ? (
                        <div className="assistant-message">
                          <div
                            className="prose prose-sm max-w-none p-4"
                            dangerouslySetInnerHTML={{ __html: m.content }}
                          />
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap text-sm">{m.content}</div>
                      )}

                      {m.role === 'assistant' &&
                        m.sources &&
                        m.sources.length > 0 && (
                          <div className="mt-0 border-t border-slate-200 px-4 py-3 bg-slate-50 rounded-b-lg">
                            <div className="font-semibold mb-1.5 text-xs text-slate-700">
                              📚 Sources
                            </div>
                            <ul className="space-y-1">
                              {m.sources
                                .slice(0, 4)
                                .map((s: any, idx: number) => (
                                  <li
                                    key={idx}
                                    className="text-xs text-slate-600 flex items-start gap-1.5"
                                  >
                                    <span className="text-amber-500 mt-0.5">•</span>
                                    <span>
                                      {s.filename || s.title || 'Source'}{' '}
                                      {s.page && (
                                        <span className="text-slate-500">(p. {s.page})</span>
                                      )}
                                    </span>
                                  </li>
                                ))}
                            </ul>
                            {typeof m.confidence === 'number' && (
                              <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-500">
                                <span>Confidence:</span>
                                <div className="flex-1 bg-slate-200 rounded-full h-1.5 max-w-[100px]">
                                  <div
                                    className="bg-amber-500 h-1.5 rounded-full transition-all"
                                    style={{
                                      width: `${Math.round(m.confidence * 100)}%`,
                                    }}
                                  />
                                </div>
                                <span className="font-medium">
                                  {Math.round(m.confidence * 100)}%
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                    </div>
                  </div>
                  ))}

                  {loading && (
                    <div className="flex justify-start">
                      <div className="inline-flex items-center gap-3 px-4 py-2 rounded-lg bg-white text-xs text-slate-600 shadow-sm">
                        <div className="flex items-center gap-1">
                          <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-bounce"></span>
                          <span className="h-1.5 w-1.5 rounded-full bg-amber-300 animate-[bounce_1s_infinite_200ms]"></span>
                          <span className="h-1.5 w-1.5 rounded-full bg-amber-200 animate-[bounce_1s_infinite_400ms]"></span>
                        </div>
                        <span className="font-medium">
                          Jericho is generating an answer...
                        </span>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              </div>

              <div className="border-t bg-white p-3 flex items-center gap-2">
                <button
                  className="rounded-md border border-slate-300 text-sm px-3 py-2 text-slate-700 hover:bg-slate-100"
                  onClick={() => {
                    setShowUpload(true)
                    setUploadFiles(null)
                    setUploadStatus(null)
                  }}
                  disabled={!currentSessionId}
                >
                  Upload
                </button>
                <textarea
                  className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                  rows={1}
                  placeholder={
                    currentSessionId
                      ? 'Ask a question about Diné College…'
                      : 'Waiting for session…'
                  }
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      if (!loading) handleSend()
                    }
                  }}
                  disabled={!currentSessionId}
                />
                <button
                  className="rounded-md bg-amber-500 hover:bg-amber-400 text-sm font-semibold px-4 py-2 text-slate-900 disabled:opacity-60"
                  disabled={loading || !input.trim() || !currentSessionId}
                  onClick={handleSend}
                >
                  {loading ? 'Sending...' : 'Send'}
                </button>
              </div>
            </>
          ) : (
            // ADMIN VIEW
            <div className="flex-1 p-4 overflow-auto bg-slate-50">
              <div className="max-w-4xl mx-auto space-y-4">
                <h2 className="text-lg font-semibold text-slate-800">
                  Admin console
                </h2>

                {/* Stats card */}
                <div className="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-slate-800">
                      RAG / documents stats
                    </h3>
                    <button
                      className="text-xs px-2 py-1 rounded-md border border-slate-300 text-slate-600 hover:bg-slate-100"
                      onClick={loadAdminStats}
                      disabled={adminStatsLoading}
                    >
                      {adminStatsLoading ? 'Refreshing...' : 'Refresh'}
                    </button>
                  </div>
                  {adminStatsLoading && (
                    <div className="text-xs text-slate-500">
                      Loading stats...
                    </div>
                  )}
                  {!adminStatsLoading && adminStats && (
                    <div className="text-sm text-slate-700">
                      <div>
                        <span className="font-medium">Total documents: </span>
                        {adminStats.documents?.total ?? 'N/A'}
                      </div>
                      {adminStats.documents?.by_type && (
                        <ul className="mt-1 text-xs text-slate-600 list-disc pl-4">
                          {Object.entries(
                            adminStats.documents.by_type as Record<
                              string,
                              number
                            >
                          ).map(([ext, count]) => (
                            <li key={ext}>
                              {ext}: {count}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                  {!adminStatsLoading && !adminStats && (
                    <div className="text-xs text-slate-500">
                      No stats available or failed to load.
                    </div>
                  )}
                </div>

                {/* Test query card */}
                <div className="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">
                  <h3 className="text-sm font-semibold text-slate-800 mb-2">
                    Orchestrator test query
                  </h3>
                  <p className="text-xs text-slate-500 mb-2">
                    Send a test question and inspect tools_used, confidence, and
                    sources.
                  </p>
                  <div className="flex items-center gap-2 mb-2">
                    <input
                      type="text"
                      className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                      placeholder="E.g. What is the check date for Pay period 3"
                      value={adminTestQuery}
                      onChange={(e) => setAdminTestQuery(e.target.value)}
                    />
                    <button
                      className="rounded-md bg-amber-500 hover:bg-amber-400 text-xs font-semibold px-3 py-2 text-slate-900 disabled:opacity-60"
                      disabled={adminTestLoading || !adminTestQuery.trim()}
                      onClick={handleAdminTestQuery}
                    >
                      {adminTestLoading ? 'Running...' : 'Run'}
                    </button>
                  </div>
                  {adminTestResponse && (
                    <pre className="mt-2 text-xs bg-slate-900 text-emerald-100 rounded-md p-3 overflow-auto max-h-64">
                      {JSON.stringify(adminTestResponse, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Upload modal */}
      {showUpload && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6">
            <h2 className="text-lg font-semibold mb-3">Upload documents</h2>
            <p className="text-xs text-slate-500 mb-3">
              Files will be parsed by Jericho and added to the knowledge base.
            </p>
            <div className="space-y-3">
              <div>
                <input
                  type="file"
                  multiple
                  className="block w-full text-sm text-slate-700"
                  onChange={(e) => setUploadFiles(e.target.files)}
                />
              </div>
              <label className="inline-flex items-center gap-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  className="rounded border-slate-300"
                  checked={uploadPrivate}
                  onChange={(e) => setUploadPrivate(e.target.checked)}
                />
                <span>Private upload (visible only to you)</span>
              </label>
              {uploadStatus && (
                <div className="text-xs text-slate-600 bg-slate-100 rounded px-3 py-2">
                  {uploadStatus}
                </div>
              )}
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                className="px-3 py-2 text-sm rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100"
                onClick={() => {
                  setShowUpload(false)
                }}
                disabled={uploadLoading}
              >
                Close
              </button>
              <button
                className="px-4 py-2 text-sm rounded-md bg-amber-500 text-slate-900 font-semibold hover:bg-amber-400 disabled:opacity-60"
                onClick={handleUpload}
                disabled={uploadLoading}
              >
                {uploadLoading ? 'Uploading...' : 'Upload'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Rename modal */}
      {showRenameModal && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-sm p-5">
            <h2 className="text-base font-semibold mb-3">Rename chat</h2>
            <input
              type="text"
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-amber-400"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
            />
            <div className="flex justify-end gap-2">
              <button
                className="px-3 py-1.5 text-xs rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100"
                onClick={() => setShowRenameModal(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-1.5 text-xs rounded-md bg-amber-500 text-slate-900 font-semibold hover:bg-amber-400"
                onClick={handleRenameSubmit}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
