import React from "react";
import Markdoc, { type RenderableTreeNode } from "@markdoc/markdoc";

function Callout({
  type = "info",
  children,
}: {
  type?: "info" | "warning" | "error";
  children: React.ReactNode;
}) {
  return <div className={`callout callout-${type}`}>{children}</div>;
}

const components = {
  Callout,
};

export function MarkdocRenderer({
  content,
}: {
  content: RenderableTreeNode;
}) {
  return (
    <div className="prose">
      {Markdoc.renderers.react(content, React, { components })}
    </div>
  );
}
