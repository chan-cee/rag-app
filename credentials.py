import subprocess
import json
import os

# CONFIG
AWS_PROFILE = "default"   # your AWS CLI SSO profile name
ACCOUNT_ID = "897189464960"      # your AWS account id
ROLE_NAME = "ccoe-powerusers"    # your role name
REGION = "eu-west-1"
REPO_DIR = "/Users/snowy/rag"  # your local git repo folder with Streamlit app

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}")
        print(result.stderr)
        exit(1)
    return result.stdout.strip()

def aws_sso_login():
    print("Running aws sso login...")
    run_cmd(f"aws sso login --profile {AWS_PROFILE}")

def get_access_token():
    # The access token is cached in ~/.aws/sso/cache/*.json by aws sso login
    import glob
    import time

    cache_files = glob.glob(os.path.expanduser("~/.aws/sso/cache/*.json"))
    if not cache_files:
        print("No AWS SSO cache files found. Run aws sso login first.")
        exit(1)

    latest_file = max(cache_files, key=os.path.getmtime)
    with open(latest_file, "r") as f:
        data = json.load(f)

    expires_at = data["expiresAt"]  # ISO 8601 string
    import datetime
    expire_dt = datetime.datetime.fromisoformat(expires_at.replace("Z","+00:00"))
    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    if expire_dt < now:
        print("SSO token expired, please run 'aws sso login' manually.")
        exit(1)

    return data["accessToken"]

def get_role_credentials(access_token):
    print("Getting AWS role credentials...")
    cmd = (
        f"aws sso get-role-credentials "
        f"--account-id {ACCOUNT_ID} "
        f"--role-name {ROLE_NAME} "
        f"--access-token {access_token} "
        f"--profile {AWS_PROFILE} "
        f"--region {REGION}"
    )
    output = run_cmd(cmd)
    creds = json.loads(output)["roleCredentials"]
    return creds

def write_secrets_toml(creds):
    secrets_path = os.path.join(REPO_DIR, ".streamlit", "secrets.toml")
    os.makedirs(os.path.dirname(secrets_path), exist_ok=True)

    content = f"""
    AWS_ACCESS_KEY_ID = "{creds['accessKeyId']}"
    AWS_SECRET_ACCESS_KEY = "{creds['secretAccessKey']}"
    AWS_SESSION_TOKEN = "{creds['sessionToken']}"
    AWS_DEFAULT_REGION = "{REGION}"
    """.strip()

    with open(secrets_path, "w") as f:
        f.write(content)
    print(f"Updated secrets at {secrets_path}")

def git_commit_and_push():
    print("Committing and pushing updated secrets.toml to GitHub...")
    cmds = [
        f"cd {REPO_DIR}",
        "git add .streamlit/secrets.toml",
        'git commit -m "Automated update AWS creds for Streamlit"',
        "git push origin main"
    ]
    run_cmd(" && ".join(cmds))
    print("Pushed update. Streamlit Cloud should redeploy automatically.")

def main():
    aws_sso_login()
    token = get_access_token()
    creds = get_role_credentials(token)
    write_secrets_toml(creds)
    git_commit_and_push()

if __name__ == "__main__":
    main()
