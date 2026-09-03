# Deployment Guide — Getting a Live Public URL

This project is containerized (see each task's `Dockerfile` +
`docker-compose.yml` at the repo root). This guide covers the fastest
path to a **public, working deployed URL** for Task 1 (the Streamlit
demo) using **Hugging Face Spaces** — free, Docker-native, and gives you
a URL in a few minutes. An alternative (Render.com) is included as a
backup in case Spaces isn't available to you.

Task 2 is a CLI/agent, not a web app (see `task2_mcp_agent/Dockerfile`'s
docstring) — it's containerized for reproducibility but doesn't have a
browser URL of its own; run it via `docker compose run --rm task2-agent`
or a Docker-hosting CI job instead.

---

## Option A (recommended) — Hugging Face Spaces, Docker SDK

**Why this one:** it deploys the actual Docker image you already built
(satisfies the "containerization" requirement AND the "deployed URL"
requirement with the same artifact — no separate non-Docker deploy path
to maintain), it's free, and it doesn't sleep on a schedule the way some
free tiers do.

### Steps

1. Create a free account at https://huggingface.co/join (if you don't
   have one).
2. Go to https://huggingface.co/new-space
   - **Space name**: e.g. `ai-ml-assessment-task1`
   - **License**: your choice (e.g. MIT)
   - **Select the Space SDK**: choose **Docker**
   - **Visibility**: Public (so the interviewer can open the URL without
     logging in)
   - Click **Create Space**. This creates a git repo for you, with a
     starter `README.md` that already has the required YAML frontmatter
     (`sdk: docker`, etc.) — you don't need to write that by hand.
3. Clone the new Space repo locally (a separate clone, not your
   submission repo):
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/ai-ml-assessment-task1
   cd ai-ml-assessment-task1
   ```
4. Copy **the contents of** `task1_multimodal_retrieval/` (everything
   inside that folder, not the folder itself) into this cloned Space
   repo, **except** don't overwrite the Space's starter `README.md`
   frontmatter — append your project's README content below the
   existing frontmatter block instead:
   ```bash
   # from inside the cloned Space repo
   cp -r /path/to/project/task1_multimodal_retrieval/* .
   # then open README.md and make sure the YAML frontmatter block
   # (the ---...--- section at the very top) is still first, with your
   # project's README content following after it
   ```
5. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy Task 1 multimodal retrieval demo"
   git push
   ```
6. Open the Space page (`https://huggingface.co/spaces/<your-username>/ai-ml-assessment-task1`)
   — it will show a **Building** status while it runs your Dockerfile,
   then switch to **Running** with your live app embedded, at:
   ```
   https://<your-username>-ai-ml-assessment-task1.hf.space
   ```
   That URL is what you send to the interviewer.

### Notes
- The deployed app runs in **offline mock mode** by default (no GPU, no
  API key needed) — appropriate for a public demo. If you want the
  hosted Qwen-VL backend live, add `DASHSCOPE_API_KEY` as a Space
  **secret** (Space Settings → Repository secrets) — never commit it to
  the repo.
- Build takes a few minutes the first time (installing dependencies +
  generating the corpus). Subsequent pushes rebuild faster via layer
  caching.

---

## Option B (backup) — Render.com, Docker Web Service

If Hugging Face Spaces isn't an option, Render.com also builds directly
from a `Dockerfile` and gives a public URL, free tier available.

1. Push your submission repo to a GitHub repository (see root
   `README.md` if you haven't already).
2. Go to https://render.com → New → **Web Service**.
3. Connect your GitHub repo, set:
   - **Root Directory**: `task1_multimodal_retrieval`
   - **Environment**: Docker (it will detect the `Dockerfile`
     automatically)
   - **Instance type**: Free
4. Deploy. Render gives you a URL like
   `https://ai-ml-assessment-task1.onrender.com`.

Free tier note: Render's free web services spin down after inactivity
and take ~30-60s to wake on the next request — mention this to the
interviewer if you use this option, so a slow first load isn't mistaken
for a broken deployment.

---

## Verifying the container locally before deploying

You don't need Docker installed to submit the source, but if you want to
verify the image builds and runs correctly on your own machine before
deploying:

```bash
cd task1_multimodal_retrieval
docker build -t task1-multimodal-retrieval .
docker run -p 8501:8501 task1-multimodal-retrieval
```

Then open `http://localhost:8501`. If that works, the same Dockerfile
will work on Hugging Face Spaces or Render.

Or, to bring up both tasks via the root orchestration file:
```bash
docker compose up --build task1-web
# in another terminal, run the Task 2 demo once:
docker compose run --rm task2-agent
```
