// Utility functions for the chat interface

export const generateSessionId = () => {
  return "session_" + Math.random().toString(36).substr(2, 9);
};

export const formatTimestamp = (timestamp) => {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
};

export const scrollToBottom = (elementRef) => {
  elementRef.current?.scrollIntoView({ behavior: "smooth" });
};

export const autoResizeTextarea = (textarea) => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
};

export const getStatusBadgeClasses = (status) => {
  const baseClasses = "ml-2 px-1.5 py-0.5 rounded text-xs";

  switch (status) {
    case "ready_for_execution":
      return `${baseClasses} bg-yellow-100 text-yellow-800`;
    case "completed":
      return `${baseClasses} bg-green-100 text-green-800`;
    case "error":
      return `${baseClasses} bg-red-100 text-red-800`;
    case "collecting_data":
      return `${baseClasses} bg-blue-100 text-blue-800`;
    default:
      return `${baseClasses} bg-gray-100 text-gray-600`;
  }
};
