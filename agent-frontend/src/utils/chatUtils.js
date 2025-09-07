// Utility functions for the chat interface (mirrored from frontend/src/utils/chatUtils.js)

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

import React from 'react';

export const renderMarkdown = (text) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let key = 0;

  lines.forEach((line) => {
    // Headers
    if (line.startsWith('### ')) {
      elements.push(
        React.createElement('h3', {
          key: key++,
          className: "text-lg font-semibold text-gray-900 dark:text-white mb-2 mt-4"
        }, line.slice(4))
      );
    } else if (line.startsWith('## ')) {
      elements.push(
        React.createElement('h2', {
          key: key++,
          className: "text-xl font-semibold text-gray-900 dark:text-white mb-3 mt-6"
        }, line.slice(3))
      );
    } else if (line.startsWith('# ')) {
      elements.push(
        React.createElement('h1', {
          key: key++,
          className: "text-2xl font-bold text-gray-900 dark:text-white mb-4 mt-6"
        }, line.slice(2))
      );
    }
    // Lists - handle nested lists properly
    else if (line.match(/^[\s]*\* /)) {
      const spaces = line.match(/^(\s*)/)[1].length;
      const content = line.replace(/^[\s]*\* /, '');
      const isNested = spaces >= 4;
      
      elements.push(
        React.createElement('div', {
          key: key++,
          className: `${isNested ? 'ml-8' : 'ml-4'} mb-1 text-gray-800 dark:text-gray-200 flex items-start`
        }, [
          React.createElement('span', { key: 'bullet', className: 'mr-2' }, '\u2022'),
          React.createElement('span', { key: 'content' }, processInlineFormatting(content))
        ])
      );
    }
    // Numbered lists
    else if (line.match(/^\d+\.\s+/)) {
      const match = line.match(/^(\d+)\.\s+(.*)/);
      if (match) {
        elements.push(
          React.createElement('div', {
            key: key++,
            className: "ml-4 mb-1 text-gray-800 dark:text-gray-200 flex items-start"
          }, [
            React.createElement('span', { key: 'number', className: 'mr-2' }, `${match[1]}.`),
            React.createElement('span', { key: 'content' }, processInlineFormatting(match[2]))
          ])
        );
      }
    }
    // Regular paragraphs
    else if (line.trim()) {
      elements.push(
        React.createElement('p', {
          key: key++,
          className: "mb-2 text-gray-800 dark:text-gray-200"
        }, processInlineFormatting(line.trim()))
      );
    }
    // Empty lines
    else {
      elements.push(React.createElement('br', { key: key++ }));
    }
  });

  // Handle bold and italic within the line
  function processInlineFormatting(text) {
    const parts = [];
    let partKey = 0;

    // Process bold text **text**
    const boldRegex = /\*\*(.*?)\*\*/g;
    let lastIndex = 0;
    let match;

    while ((match = boldRegex.exec(text)) !== null) {
      // Add text before the bold part
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }
      
      // Add the bold part
      parts.push(
        React.createElement('strong', {
          key: `bold-${partKey++}`,
          className: "font-semibold text-gray-900 dark:text-white"
        }, match[1])
      );
      
      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    // If no bold formatting found, return the original text
    return parts.length > 0 ? parts : text;
  }

  return React.createElement('div', {
    className: "prose prose-sm max-w-none"
  }, elements);
};
