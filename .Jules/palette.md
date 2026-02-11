## 2026-02-08 - Headless UX with Single-File Dashboards
**Learning:** UX for headless backend systems can be dramatically improved by embedding a simple, dependency-free HTML/JS dashboard served directly by the API.
**Action:** Always consider adding a '/dashboard' endpoint with a single-file HTML template using CDN-based Alpine.js/Tailwind for immediate visualization of backend state.

## 2026-02-08 - Active Control in Dashboards
**Learning:** Users prefer active control over passive monitoring. Adding "Dry Run" and "Pause/Resume" buttons directly to the dashboard significantly increases its utility.
**Action:** When building dashboards, always include "Emergency Stop" and "Test Mode" controls if the system is autonomous.

## 2026-02-08 - Securing Headless UIs
**Learning:** Even internal dashboards need security. Implementing a simple client-side API Key modal (stored in localStorage) provides a massive security upgrade over open endpoints without requiring complex backend session management.
**Action:** Always secure control endpoints with at least an API Key, and provide a way for the UI to input it.

## 2026-02-08 - Secrets Management
**Learning:** Never print sensitive keys to stdout, even in "helpful" startup banners. CodeQL flags this immediately.
**Action:** Always redact or omit secrets in logs.

## 2026-02-08 - Automated Workflows vs. Agent Constraints
**Learning:** Users may expect agents to behave like stateless bots (opening new PRs on command). Agents bound to stateful sessions must communicate their limitations clearly and persist on the current branch.
**Action:** When asked to "open a new PR", politely explain the single-session constraint and continue working on the current branch.
