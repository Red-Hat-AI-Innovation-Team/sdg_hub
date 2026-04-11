"use client";

import { useEffect, useRef } from "react";

const steps = [
  { label: "dataset", type: "input" },
  { label: "PromptBuilder", type: "block" },
  { label: "LLMChat", type: "block" },
  { label: "TagParser", type: "block" },
  { label: "Filter", type: "block" },
  { label: "enriched", type: "output" },
];

const SEGMENT_DURATION = 800; // ms per connector segment
const PAUSE_DURATION = 200; // ms pause/glow at each block
const TOTAL_SEGMENTS = steps.length - 1; // 5 connectors
const CYCLE_DURATION =
  TOTAL_SEGMENTS * SEGMENT_DURATION + (TOTAL_SEGMENTS - 1) * PAUSE_DURATION; // ~9.1s total

export function AnimatedPipeline() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const packets = container.querySelectorAll<HTMLElement>(".packet");
    const nodes = container.querySelectorAll<HTMLElement>(".pipe-block");
    const timeouts: ReturnType<typeof setTimeout>[] = [];

    function hideAllPackets() {
      packets.forEach((pkt) => {
        pkt.style.opacity = "0";
        pkt.style.animation = "none";
      });
    }

    function runCycle() {
      hideAllPackets();

      for (let i = 0; i < TOTAL_SEGMENTS; i++) {
        const startDelay = i * (SEGMENT_DURATION + PAUSE_DURATION);

        // Animate packet across connector i
        timeouts.push(
          setTimeout(() => {
            // Hide all packets first, then show only the current one
            hideAllPackets();
            const pkt = packets[i];
            if (!pkt) return;
            pkt.style.animation = "none";
            pkt.offsetHeight; // force reflow
            pkt.style.animation = `packet-move ${SEGMENT_DURATION}ms ease-in-out forwards`;
          }, startDelay)
        );

        // Glow the destination node when packet arrives
        timeouts.push(
          setTimeout(() => {
            const targetNode = nodes[i + 1];
            if (!targetNode) return;
            targetNode.style.animation = "none";
            targetNode.offsetHeight;
            targetNode.style.animation = "node-glow 0.6s ease-out";
          }, startDelay + SEGMENT_DURATION - 100)
        );

        // Hide packet after it finishes (before next segment starts)
        timeouts.push(
          setTimeout(() => {
            const pkt = packets[i];
            if (pkt) pkt.style.opacity = "0";
          }, startDelay + SEGMENT_DURATION)
        );
      }
    }

    runCycle();
    const interval = setInterval(runCycle, CYCLE_DURATION + 1500); // 1.5s rest between cycles
    return () => {
      clearInterval(interval);
      timeouts.forEach(clearTimeout);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="mt-16 flex flex-wrap items-center justify-center"
      style={{ fontFamily: "var(--font-mono)", gap: 0 }}
    >
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-center">
          {/* Node */}
          <div
            className={`pipe-block rounded-md px-3 py-1.5 text-xs sm:text-sm whitespace-nowrap ${
              step.type === "block" ? "pipe-block-active" : ""
            }`}
            style={{
              background:
                step.type === "block"
                  ? "rgba(232, 151, 93, 0.1)"
                  : step.type === "output"
                    ? "rgba(125, 170, 140, 0.1)"
                    : "var(--color-bg-2)",
              color:
                step.type === "block"
                  ? "var(--color-accent)"
                  : step.type === "output"
                    ? "var(--color-green)"
                    : "var(--color-text-2)",
              boxShadow:
                step.type === "block"
                  ? "0 0 0 1px rgba(232, 151, 93, 0.2)"
                  : step.type === "output"
                    ? "0 0 0 1px rgba(125, 170, 140, 0.2)"
                    : "0 0 0 1px var(--color-border)",
            }}
          >
            {step.label}
          </div>

          {/* Connector with packet */}
          {i < steps.length - 1 && (
            <div className="relative mx-1 sm:mx-2 hidden sm:block" style={{ width: 36, height: 2 }}>
              {/* Line */}
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 2,
                  background: "var(--color-bg-3)",
                }}
              />
              {/* Arrow */}
              <div
                style={{
                  position: "absolute",
                  top: -3,
                  right: -1,
                  width: 0,
                  height: 0,
                  borderLeft: "5px solid var(--color-bg-3)",
                  borderTop: "4px solid transparent",
                  borderBottom: "4px solid transparent",
                }}
              />
              {/* Animated packet dot */}
              <div
                className="packet"
                style={{
                  position: "absolute",
                  top: -3,
                  left: 0,
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  background: "var(--color-accent)",
                  boxShadow: "0 0 8px rgba(232, 151, 93, 0.6)",
                  opacity: 0,
                }}
              />
            </div>
          )}

          {/* Mobile arrow (no animation) */}
          {i < steps.length - 1 && (
            <span className="sm:hidden text-text-3 mx-1 text-xs">&rarr;</span>
          )}
        </div>
      ))}

      <style>{`
        @keyframes packet-move {
          0% { left: 0; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { left: calc(100% - 7px); opacity: 0; }
        }
        @keyframes node-glow {
          0% { box-shadow: 0 0 0 1px rgba(232, 151, 93, 0.2); }
          50% { box-shadow: 0 0 16px 2px rgba(232, 151, 93, 0.25), 0 0 0 1px rgba(232, 151, 93, 0.4); }
          100% { box-shadow: 0 0 0 1px rgba(232, 151, 93, 0.2); }
        }
      `}</style>
    </div>
  );
}
