import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

export default function ChatScreen() {
  const [messages, setMessages] = useState([
    { sender: "assistant", text: "Hello! How can I help you today?" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Replace this with your AI API call
  async function getChatResponse(userMessage) {
    // Placeholder: you should call your backend/ChatGPT API here
    console.log(userMessage )
    return new Promise((resolve) =>
      setTimeout(() => resolve("This is a sample response from the assistant."), 600)
    );
  }

  async function handleSendMessage(e) {
    e.preventDefault();
    if (inputValue.trim() === "") return;

    const newUserMessage = { sender: "user", text: inputValue };
    setMessages((prev) => [...prev, newUserMessage]);
    setInputValue("");
    setIsLoading(true);

    const assistantReply = await getChatResponse(inputValue);

    setMessages((prev) => [
      ...prev,
      { sender: "assistant", text: assistantReply },
    ]);
    setIsLoading(false);
  }

  return (
    <div className="container py-4">
      <div className="card shadow-sm" style={{ height: "80vh" }}>
        <div className="card-body d-flex flex-column">
          <div
            className="flex-grow-1 overflow-auto mb-3"
            style={{ maxHeight: "70vh" }}
          >
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`mb-2 d-flex ${
                  msg.sender === "user" ? "justify-content-end" : "justify-content-start"
                }`}
              >
                <div
                  className={`p-2 rounded ${
                    msg.sender === "user"
                      ? "bg-primary text-white"
                      : "bg-light text-dark"
                  }`}
                  style={{ maxWidth: "75%" }}
                >
                  {msg.text}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="text-muted small">Assistant is typing...</div>
            )}
          </div>

          <form onSubmit={handleSendMessage}>
            <div className="input-group">
              <input
                className="form-control"
                type="text"
                placeholder="Type your message..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
              />
              <button className="btn btn-primary" type="submit">
                Send
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
