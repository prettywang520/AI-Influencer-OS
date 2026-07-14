# Workflow Architecture

Version: 1.0

---

## Purpose

This document defines the architecture of all workflows.

Every workflow must follow the same structure.

Workflow

↓

Context

↓

Memory

↓

Agent Chain

↓

Quality Check

↓

Output

---

Workflow Components

1. Workflow Definition

2. Context Loading

3. Memory Loading

4. Agent Execution

5. Handoff

6. Quality Assurance

7. Publishing

8. Archive

---

Workflow Rules

Every workflow must:

• Read Persona

• Read Brand

• Read Workflow

• Read Memory

• Execute Agents

• Pass QA

• Update Memory

• Archive Results