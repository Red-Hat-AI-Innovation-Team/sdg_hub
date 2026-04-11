"use client";

import { useState, useEffect, useCallback } from "react";

const phrases = [
  "Generate training data from documents",
  "Build QA pairs with LLM pipelines",
  "Filter and validate synthetic datasets",
  "Chain blocks into reproducible flows",
  "Scale data generation with async execution",
];

export function TypewriterSubtitle() {
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [text, setText] = useState("");

  const tick = useCallback(() => {
    const currentPhrase = phrases[phraseIndex];

    if (!isDeleting) {
      // Typing
      if (charIndex < currentPhrase.length) {
        setText(currentPhrase.slice(0, charIndex + 1));
        setCharIndex((prev) => prev + 1);
      } else {
        // Pause at end, then start deleting
        setTimeout(() => setIsDeleting(true), 2000);
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
    return () => clearTimeout(timer);
  }, [tick, isDeleting]);

  return (
    <p className="mt-5 h-8 text-lg text-text-1" style={{ fontFamily: "var(--font-mono)" }}>
      <span>{text}</span>
      <span className="typewriter-cursor" />
    </p>
  );
}
