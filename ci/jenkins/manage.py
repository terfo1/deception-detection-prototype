"""Start the isolated local Jenkins lab; trigger, inspect and collect real builds.

Uses only the Python standard library. Password stays in ignored ci/jenkins/.env.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import http.cookiejar
import json
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / "ci/jenkins/.env"
BASE = "http://127.0.0.1:8085"
JOB = "/job/eye-tracking-assignment5"


def settings() -> dict[str, str]:
    return dict(line.split("=", 1) for line in ENV_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


def client():
    password = settings()["JENKINS_ADMIN_PASSWORD"]
    token = base64.b64encode(f"research-admin:{password}".encode()).decode()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("Authorization", f"Basic {token}")]
    return opener


def read_json(opener, path):
    with opener.open(BASE + path, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "build", "status", "collect"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--number", type=int)
    args = parser.parse_args()
    if args.action == "start":
        if not ENV_FILE.exists():
            ENV_FILE.write_text(f"JENKINS_ADMIN_PASSWORD={secrets.token_urlsafe(36)}\nRESEARCH_BRANCH=main\n",
                                encoding="utf-8")
        subprocess.run(["docker", "compose", "-f", "ci/jenkins/compose.yaml", "up", "-d", "--build"],
                       cwd=ROOT, check=True)
        opener = client()
        for _ in range(120):
            try:
                read_json(opener, "/api/json")
                print(f"Jenkins ready: {BASE}{JOB}/ (user research-admin; password in ignored .env)")
                return
            except (OSError, http.client.HTTPException, json.JSONDecodeError):
                time.sleep(2)
        raise TimeoutError("Jenkins did not become ready within four minutes")
    opener = client()
    if args.action == "build":
        crumb = read_json(opener, "/crumbIssuer/api/json")
        request = urllib.request.Request(BASE + JOB + "/build", data=b"", method="POST",
                                         headers={crumb["crumbRequestField"]: crumb["crumb"]})
        with opener.open(request, timeout=30) as response:
            queue_url = response.headers["Location"]
        if not queue_url.startswith(BASE + "/queue/item/"):
            raise ValueError("Unexpected queue location")
        for _ in range(150):
            with opener.open(queue_url + "api/json", timeout=30) as response:
                queued = json.load(response)
            if "executable" in queued:
                print(json.dumps(queued["executable"], indent=2))
                return
            if queued.get("cancelled"):
                raise RuntimeError("Build queue item was cancelled")
            time.sleep(2)
        raise TimeoutError("Build did not leave queue within five minutes")
    number = args.number or read_json(opener, JOB + "/lastBuild/api/json")["number"]
    build_path = f"{JOB}/{number}"
    state = read_json(opener, build_path + "/api/json")
    if args.action == "status":
        print(json.dumps({key: state.get(key) for key in
                          ["number", "building", "result", "url", "duration"]}, indent=2))
        return
    if args.output_dir is None:
        parser.error("collect requires --output-dir (a new directory)")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "build.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    for remote, local in [("consoleText", "console.txt"), ("wfapi/describe", "stages.json"),
                          ("testReport/api/json", "tests.json"), ("artifact/*zip*/archive.zip", "artifacts.zip")]:
        with opener.open(BASE + build_path + "/" + remote, timeout=60) as response:
            (args.output_dir / local).write_bytes(response.read())
    print(f"Collected actual Jenkins build {number}: {state['result']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
