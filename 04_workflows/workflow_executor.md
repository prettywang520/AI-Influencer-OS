# Workflow Executor

Version: 1.0

---

Purpose

Execute workflow according to the execution pipeline.

Execution Order

Receive Task

↓

Load Workflow

↓

Load Context

↓

Execute Agent 1

↓

Handoff

↓

Execute Agent 2

↓

Handoff

↓

Execute Agent N

↓

QA

↓

Save Memory

↓

Publish

↓

Archive

---

Execution Rules

Never skip an Agent.

Always validate outputs.

Always update workflow status.

Always record execution logs.