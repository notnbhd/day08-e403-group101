const messagesEl = document.getElementById("messages")
const formEl = document.getElementById("chatForm")
const inputEl = document.getElementById("messageInput")
const sendButton = document.getElementById("sendButton")
const resetButton = document.getElementById("resetButton")
const newChatButton = document.getElementById("newChatButton")
const topKInput = document.getElementById("topKInput")
const rewriteInput = document.getElementById("rewriteInput")
const logsInput = document.getElementById("logsInput")
const healthStatus = document.getElementById("healthStatus")
const sessionLabel = document.getElementById("sessionLabel")
const conversationList = document.getElementById("conversationList")

let sessionId = localStorage.getItem("rag_session_id") || null

function updateSessionLabel () {
    sessionLabel.textContent = sessionId ? `${sessionId.slice(0, 8)}...` : "new"
}

function escapeHtml (value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;")
}

function renderAnswer (text) {
    const escaped = escapeHtml(text)
    const withCitations = escaped.replace(/\[([^\[\]\n]{1,180})\]/g, (match) => {
        return `<span class="citation">${match}</span>`
    })
    return withCitations.replaceAll("\n", "<br />")
}

function scrollToBottom () {
    messagesEl.scrollTop = messagesEl.scrollHeight
}

function autosizeInput () {
    inputEl.style.height = "auto"
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`
}

function emptyState () {
    messagesEl.innerHTML = `
        <section class="empty-state">
            <h2>Hỏi về pháp luật và tin tức ma túy</h2>
            <p>Hệ thống trả lời bằng RAG, có citation, nhớ ngữ cảnh hội thoại và hiển thị source documents.</p>
        </section>
    `
}

function appendMessage (role, text, extra = {}) {
    const existingEmpty = messagesEl.querySelector(".empty-state")
    if (existingEmpty) existingEmpty.remove()

    const article = document.createElement("article")
    article.className = `message ${role}`

    const avatar = document.createElement("div")
    avatar.className = "avatar"
    avatar.textContent = role === "user" ? "U" : "AI"

    const body = document.createElement("div")
    body.className = "message-body"

    const bubble = document.createElement("div")
    bubble.className = "bubble"
    bubble.innerHTML = role === "assistant" ? renderAnswer(text) : escapeHtml(text)
    body.appendChild(bubble)

    if (role === "assistant") {
        body.appendChild(renderMeta(extra))

        if (extra.sources && extra.sources.length) {
            body.appendChild(renderSources(extra.sources))
        }

        if (extra.suggested_questions && extra.suggested_questions.length) {
            body.appendChild(renderSuggestions(extra.suggested_questions))
        }

        if (extra.logs) {
            body.appendChild(renderLogs(extra.logs))
        }
    }

    article.appendChild(avatar)
    article.appendChild(body)
    messagesEl.appendChild(article)
    scrollToBottom()
    return article
}

function renderMeta (data = {}) {
    const meta = document.createElement("div")
    meta.className = "meta-row"

    const items = [
        `retrieval: ${data.retrieval_source || "none"}`,
        `time: ${Number(data.elapsed_seconds || 0).toFixed(2)}s`,
        `sources: ${(data.sources || []).length}`,
    ]

    if (data.citations && data.citations.length) {
        items.push(`citations: ${data.citations.length}`)
    }

    if (data.standalone_question && data.standalone_question !== data.original_question) {
        items.push(`query: ${data.standalone_question}`)
    }

    items.forEach((item) => {
        const pill = document.createElement("span")
        pill.className = "pill"
        pill.textContent = item
        meta.appendChild(pill)
    })

    return meta
}

function normalizeHistorySources (sources = []) {
    return sources.map((src, index) => {
        const metadata = src.metadata || {}
        const content = src.content || ""
        return {
            index: index + 1,
            content,
            preview: src.preview || content.slice(0, 420).replaceAll("\n", " ").trim(),
            score: Number(src.score || 0),
            retrieval_source: src.retrieval_source || src.source || "unknown",
            source_name: src.source_name || metadata.source || `Source ${index + 1}`,
            doc_type: src.doc_type || metadata.type || metadata.doc_type || "unknown",
            chunk_index: src.chunk_index ?? metadata.chunk_index ?? null,
        }
    })
}

function renderSources (sources) {
    const details = document.createElement("details")
    details.className = "details source-details"

    const summary = document.createElement("summary")
    summary.textContent = `Sources used (${sources.length})`
    details.appendChild(summary)

    sources.forEach((src) => {
        const card = document.createElement("div")
        card.className = "source-card"

        const head = document.createElement("div")
        head.className = "source-head"
        head.innerHTML = `
            <span>#${src.index} · ${escapeHtml(src.source_name)}</span>
            <span>${Number(src.score || 0).toFixed(3)}</span>
        `
        card.appendChild(head)

        const meta = document.createElement("div")
        meta.className = "source-meta"
        meta.textContent = [
            `type=${src.doc_type || "unknown"}`,
            `retrieval=${src.retrieval_source || "unknown"}`,
            src.chunk_index !== null && src.chunk_index !== undefined ? `chunk=${src.chunk_index}` : null,
        ]
            .filter(Boolean)
            .join(" · ")
        card.appendChild(meta)

        const preview = document.createElement("div")
        preview.className = "source-preview"
        preview.textContent = src.preview || ""
        card.appendChild(preview)

        details.appendChild(card)
    })

    return details
}

function renderSuggestions (questions) {
    const wrap = document.createElement("div")
    wrap.className = "suggestions"

    questions.forEach((question) => {
        const button = document.createElement("button")
        button.type = "button"
        button.className = "suggestion"
        button.textContent = question
        button.addEventListener("click", () => {
            inputEl.value = question
            autosizeInput()
            inputEl.focus()
        })
        wrap.appendChild(button)
    })

    return wrap
}

function renderLogs (logs) {
    const details = document.createElement("details")
    details.className = "details"

    const summary = document.createElement("summary")
    summary.textContent = "Pipeline logs"
    details.appendChild(summary)

    const pre = document.createElement("div")
    pre.className = "log-box"
    pre.textContent = logs
    details.appendChild(pre)

    return details
}

function appendLoading () {
    return appendMessage("assistant", "Đang retrieval + generation...")
}

async function refreshSessions () {
    try {
        const response = await fetch("/api/sessions")
        if (!response.ok) throw new Error("cannot load sessions")
        const data = await response.json()
        renderSessionList(data.sessions || [])
    } catch (_) {
        renderSessionList([])
    }
}

function renderSessionList (sessions) {
    conversationList.innerHTML = ""
    if (!sessions.length) {
        const empty = document.createElement("div")
        empty.className = "history-empty"
        empty.textContent = "Chưa có hội thoại"
        conversationList.appendChild(empty)
        return
    }

    sessions.forEach((session) => {
        const button = document.createElement("button")
        button.type = "button"
        button.className = `history-item${session.session_id === sessionId ? " active" : ""}`
        button.innerHTML = `
            <span>${escapeHtml(session.title || "Hội thoại mới")}</span>
            <small>${Number(session.turn_count || 0)} lượt</small>
        `
        button.addEventListener("click", () => loadSession(session.session_id))
        conversationList.appendChild(button)
    })
}

async function loadSession (nextSessionId) {
    try {
        const response = await fetch(`/api/session/${encodeURIComponent(nextSessionId)}`)
        if (!response.ok) throw new Error("cannot load session")
        const data = await response.json()

        sessionId = data.session_id
        localStorage.setItem("rag_session_id", sessionId)
        updateSessionLabel()
        messagesEl.innerHTML = ""

        const history = data.history || []
        if (!history.length) {
            emptyState()
        }

        history.forEach((turn) => {
            appendMessage("user", turn.user || "")
            appendMessage("assistant", turn.assistant || "", {
                original_question: turn.user || "",
                standalone_question: turn.standalone_question || "",
                sources: normalizeHistorySources(turn.sources || []),
                citations: [],
                retrieval_source: "history",
                elapsed_seconds: 0,
            })
        })

        await refreshSessions()
    } catch (error) {
        appendMessage("assistant", `Không tải được session: ${error.message}`, {
            retrieval_source: "error",
            elapsed_seconds: 0,
            sources: [],
            citations: [],
        })
    }
}

async function sendMessage (message) {
    appendMessage("user", message)

    inputEl.value = ""
    autosizeInput()
    inputEl.focus()
    sendButton.disabled = true
    inputEl.disabled = true

    const loadingEl = appendLoading()

    try {
        const payload = {
            message,
            session_id: sessionId,
            top_k: Number(topKInput.value || 5),
            rewrite_followup: rewriteInput.checked,
            include_logs: logsInput.checked,
        }

        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        })

        if (!response.ok) {
            const errorText = await response.text()
            throw new Error(errorText)
        }

        const data = await response.json()

        sessionId = data.session_id
        localStorage.setItem("rag_session_id", sessionId)
        updateSessionLabel()

        loadingEl.remove()
        appendMessage("assistant", data.answer, data)
        await refreshSessions()
    } catch (error) {
        loadingEl.remove()
        appendMessage("assistant", `Không gọi được API: ${error.message}`, {
            retrieval_source: "error",
            elapsed_seconds: 0,
            sources: [],
            citations: [],
            suggested_questions: [],
            logs: "",
        })
    } finally {
        sendButton.disabled = false
        inputEl.disabled = false
        scrollToBottom()
    }
}

function startNewChat () {
    sessionId = null
    localStorage.removeItem("rag_session_id")
    updateSessionLabel()
    emptyState()
    refreshSessions()
    inputEl.focus()
}

formEl.addEventListener("submit", async (event) => {
    event.preventDefault()
    const message = inputEl.value.trim()
    if (!message) return
    await sendMessage(message)
})

inputEl.addEventListener("input", autosizeInput)

inputEl.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault()
        formEl.requestSubmit()
    }
})

newChatButton.addEventListener("click", startNewChat)

resetButton.addEventListener("click", async () => {
    try {
        await fetch("/api/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        })
    } catch (_) {
        // Local UI reset still works even if API reset fails.
    }

    startNewChat()
})

async function checkHealth () {
    try {
        const response = await fetch("/health")
        if (!response.ok) throw new Error("health check failed")
        const data = await response.json()
        healthStatus.textContent = `${data.status} · ${data.app}`
    } catch (_) {
        healthStatus.textContent = "API chưa sẵn sàng"
        healthStatus.classList.add("error")
    }
}

updateSessionLabel()
emptyState()
autosizeInput()
checkHealth()
refreshSessions()
if (sessionId) {
    loadSession(sessionId)
}
