# Rosetta UI

A minimal web page for asking Rosetta questions. It calls the API's `POST /questions` and shows the answer, the
explanation (what was counted, how dates were read), the result table, and, on request, how it was calculated:
definitions used, values matched, the query and the agent's steps.

## Run

1. Start the API from the repository root (it must allow this page's address, see `CORS_ORIGINS` in `.env`):

   ```
   .venv/Scripts/python.exe -m uvicorn rosetta.api.main:app --port 8000
   ```

2. Start the UI:

   ```
   cd ui
   cp .env.example .env      # VITE_API_URL points at the API, http://localhost:8000 by default
   npm install
   npm run dev
   ```

3. Open http://localhost:5173.

The "Today is" date is optional: it sets what relative dates such as "last month" mean. Recent questions are kept in
this browser only.
