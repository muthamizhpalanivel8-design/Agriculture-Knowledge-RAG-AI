const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const messages = document.getElementById("messages");

function addMessage(text, role, sources = []) {
    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();

    const wrap = document.createElement("div");
    wrap.className = `message ${role}`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);

    if (sources.length) {
        const source = document.createElement("div");
        source.className = "sources";
        source.textContent = "Sources: " + sources.join(", ");
        wrap.appendChild(source);
    }

    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
}

uploadBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
        statusEl.textContent = "Select at least one agriculture document first.";
        return;
    }

    const fd = new FormData();
    for (const file of fileInput.files) {
        fd.append("files", file);
    }

    uploadBtn.disabled = true;
    statusEl.textContent = "Extracting text and building RAG index...";

    try {
        const res = await fetch("/upload", {
            method: "POST",
            body: fd
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Upload failed.");

        statusEl.textContent =
            `${data.files.length} file(s) processed • ${data.chunks} chunks indexed`;

        addMessage(
            `Successfully indexed ${data.files.length} agriculture document(s). You can now ask questions.`,
            "bot"
        );
    } catch (err) {
        statusEl.textContent = err.message;
    } finally {
        uploadBtn.disabled = false;
    }
});

clearBtn.addEventListener("click", async () => {
    if (!confirm("Clear all uploaded agriculture documents and the RAG index?")) {
        return;
    }

    const res = await fetch("/clear", { method: "POST" });
    const data = await res.json();

    statusEl.textContent = data.message || "Cleared.";
    messages.innerHTML = "";
    addMessage("All agriculture documents and the RAG index were cleared.", "bot");
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const q = question.value.trim();
    if (!q) return;

    addMessage(q, "user");
    question.value = "";

    const thinking = document.createElement("div");
    thinking.className = "message bot";
    thinking.innerHTML =
        '<div class="bubble">Searching your agriculture documents…</div>';

    messages.appendChild(thinking);
    messages.scrollTop = messages.scrollHeight;

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q })
        });

        const data = await res.json();
        thinking.remove();

        if (!res.ok) throw new Error(data.error || "Request failed.");

        addMessage(data.answer, "bot", data.sources || []);
    } catch (err) {
        thinking.remove();
        addMessage(err.message, "bot");
    }
});
