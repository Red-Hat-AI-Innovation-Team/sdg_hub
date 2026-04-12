"use client";

import { useState, useEffect, useCallback, useRef } from "react";

const phrases = [
  "Distill tool-use trajectories from MCP servers",
  "Generate QA pairs grounded in your documents",
  "Evaluate agent responses with LLM judges",
  "Build red-team datasets for safety testing",
  "Chain blocks into reproducible YAML flows",
];

export function TypewriterSubtitle() {
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [text, setText] = useState("");
  const pauseTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const tick = useCallback(() => {
    const currentPhrase = phrases[phraseIndex];

    if (!isDeleting) {
      // Typing
      if (charIndex < currentPhrase.length) {
        setText(currentPhrase.slice(0, charIndex + 1));
        setCharIndex((prev) => prev + 1);
      } else {
        // Pause at end, then start deleting
        pauseTimer.current = setTimeout(() => setIsDeleting(true), 2000);
        return;
      }
    } else {
      // Deleting
      if (charIndex > 0) {
        setText(currentPhrase.slice(0, charIndex - 1));
        setCharIndex((prev) => prev - 1);
      } else {
        setIsDeleting(false);
        setPhraseIndex((prev) => (prev + 1) % phrases.length);
        return;
      }
    }
  }, [charIndex, isDeleting, phraseIndex]);

  useEffect(() => {
    const speed = isDeleting ? 30 : 50;
    const timer = setTimeout(tick, speed);
    return () => {
      clearTimeout(timer);
      clearTimeout(pauseTimer.current);
    };
  }, [tick, isDeleting]);

  return (
    <p className="mt-5 h-8 text-lg text-text-1" style={{ fontFamily: "var(--font-mono)" }}>
      <span>{text}</span>
      <span className="typewriter-cursor" />
    </p>
  );
}
