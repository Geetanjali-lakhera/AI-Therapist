document.addEventListener("DOMContentLoaded", function () {
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("user-input");
  const chatContainer = document.getElementById("chat-container");
  const clearBtn = document.getElementById("clear-btn");
  const saveBtn = document.getElementById("save-btn");

  // Send on button click
  sendBtn.addEventListener("click", sendMessage);

  // Send on Enter key
  userInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage("user", message);
    userInput.value = "";

    fetch("/get_response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    })
      .then((res) => res.json())
      .then((data) => {
        addMessage("bot", data.response);
      })
      .catch(() => {
        addMessage("bot", "⚠️ Error: could not reach server.");
      });
  }

  function addMessage(sender, text) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.innerText = text;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  clearBtn.addEventListener("click", function () {
    chatContainer.innerHTML = "";
  });

  saveBtn.addEventListener("click", function () {
    const text = [...chatContainer.querySelectorAll(".message")]
      .map((el) => el.innerText)
      .join("\n\n");

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = "chat_log.txt";
    link.href = url;
    link.click();
  });
});
