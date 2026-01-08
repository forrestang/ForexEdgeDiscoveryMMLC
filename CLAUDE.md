# Claude Instructions for ForexEdgeDiscoveryMMLC

## Port Requirements (STRICT)

- **Frontend**: Must run on port **5173** only
- **Backend**: Must run on port **8000** only

Do NOT use any other ports for this project.

## If Ports Are In Use

1. Kill whatever process is using the required port
2. Restart the service on the correct port (5173 for frontend, 8000 for backend)
3. If you cannot kill the blocking process, notify the user BEFORE attempting any alternative

Never silently switch to a different port.

## Backend Development Workflow

When Claude is running the backend as a background task:
- Do NOT rely on `--reload` - it is unreliable on Windows
- After making ANY changes to backend Python files, ALWAYS:
  1. Kill the running backend background task
  2. Restart with: `cd backend && python -m uvicorn app.main:app --port 8000`
  3. Wait for "Application startup complete" before telling the user to test

This only applies when Claude has started the backend as a background task. If the user is running servers in their own terminal, notify them to restart manually.
